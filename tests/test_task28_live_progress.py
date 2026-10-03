"""CPU/stub observability checks only; no real training or explanation."""
import ast
import copy
import inspect
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_elliptic_pgexplainer_training as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('phase', ['TRAIN', 'PUBLICATION', 'COMPLETE', 'FAILED'])
def test_phase_contract_and_no_rng_or_state_changes(capsys, phase):
    state = torch.get_rng_state().clone()
    row = {'loss': 0.25, 'epoch': '1/30', 'checkpoint': 'epoch_001.pt'}
    before = copy.deepcopy(row)
    runner.progress('GCN', 42, phase, **row)
    assert capsys.readouterr().out.startswith(f'[TASK28][GCN][seed42][{phase}] ')
    assert row == before
    assert torch.equal(state, torch.get_rng_state())


def test_validation_real_loop_order_health_and_no_metrics(monkeypatch, capsys):
    calls = []

    def explanation(*args, **kwargs):
        calls.append(kwargs['index'])
        return SimpleNamespace(edge_mask=torch.tensor([0.2, 0.8]))

    monkeypatch.setattr(runner, 'computation_subgraph_edge_mask',
                        lambda *args, **kwargs: torch.tensor([True, True]))
    data = SimpleNamespace(x=None, edge_index=None)
    rows, _ = runner.validate_pg_mask_health(
        explainer=explanation, data=data, data_cpu=data, frozen_predictions=torch.tensor([0]),
        val_targets=[SimpleNamespace(node_id=n) for n in [54, 61]], limit=None,
        progress_backbone='GCN', progress_seed=42)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert 'event="START" targets=2' in lines[0]
    assert calls == [54, 61]
    for i, (node, line, row) in enumerate(zip(calls, lines[1:], rows, strict=True), 1):
        assert f'node_id={node}' in line and f'progress="{i}/2"' in line
        assert f'edge_mask_std={row["edge_mask_std"]}' in line
        assert row['candidate_edge_count'] == 2
        assert row['edge_mask_count'] == 2 and row['edge_mask_finite']
        assert not any(key in line for key in ['fid_', 'sparsity', 'jaccard'])


@pytest.mark.parametrize('phase,metric', [('FINAL_EVAL', 'remove_delta'), ('STABILITY', 'jaccard')])
def test_future_result_contract_forwards_only_existing_values(capsys, phase, metric):
    row = {'node_id': 123, metric: 0.125, 'status': 'success'}
    before = copy.deepcopy(row)
    runner.progress_result('GraphSAGE', 0, phase, row, 2, 3, 0.5)
    line = capsys.readouterr().out
    assert f'[{phase}] node_id=123 progress="2/3" {metric}=0.125' in line
    assert 'fid_plus' not in line and row == before


def test_checkpoint_publication_and_train_instrumentation_placement():
    source = inspect.getsource(runner.train_one)
    assert source.index('write_epoch_csv(') < source.index('"TRAIN"')
    assert source.index('"PUBLICATION", event="START"') < source.index('atomic_torch_save(artifact')
    assert source.index('publish_completion(') < source.index('"PUBLICATION", event="DONE"')
    for name in ('progress', 'progress_result'):
        tree = ast.parse(inspect.getsource(getattr(runner, name)))
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert not any('random' in call or 'torch.' in call or 'numpy' in call for call in calls)


def test_main_completion_and_failure(monkeypatch, capsys):
    monkeypatch.setattr(runner, 'parse_args', lambda: SimpleNamespace(backbone='gcn', seed=42))
    monkeypatch.setattr(runner, 'train_one',
                        lambda _: {'status': 'success', 'artifact_path': 'stub.pt'})
    runner.main()
    assert '[COMPLETE] status="success" artifact="stub.pt"' in capsys.readouterr().out
    monkeypatch.setattr(runner, 'train_one',
                        lambda _: {'status': 'already_complete',
                                   'complete_marker': 'COMPLETE.json'})
    runner.main()
    assert '[COMPLETE] status="already_complete"' in capsys.readouterr().out

    def fail(_):
        raise RuntimeError('stub failure')

    monkeypatch.setattr(runner, 'train_one', fail)
    with pytest.raises(RuntimeError, match='stub failure'):
        runner.main()
    assert '[FAILED]' in capsys.readouterr().out


@pytest.mark.parametrize('fail_worker', ['', 'gcn'])
@pytest.mark.skipif(not (ROOT / 'docs/kaggle/task28_kaggle_dual_gpu_run.sh').exists(),
                    reason='External deployment adapter is not part of the scientific clone')
def test_live_stream_concurrency_logs_and_failure(tmp_path, fail_worker):
    source = (ROOT / 'docs/kaggle/task28_kaggle_dual_gpu_run.sh').read_text()
    # Exercise the real launch/join bodies with stub executables, never a GPU child.
    functions = source[source.index('launch()'):source.index('launch 0 gcn reference')]
    stub = tmp_path / 'worker.sh'
    stub.write_text('''#!/usr/bin/env bash
backbone=unknown
while [[ $# -gt 0 ]]; do
 if [[ "$1" == --backbone ]]; then backbone=$2; fi
 shift
done
printf '[TASK28][%s][seed42][TRAIN] epoch=1/30\\n' "$backbone"
printf 'stderr-%s\\n' "$backbone" >&2
date +%s > "$output/$backbone.start"
while [[ ! -f "$output/release" ]]; do sleep 0.05; done
date +%s > "$output/$backbone.end"
[[ "$backbone" != "$FAIL_WORKER" ]]
''', newline='\n')
    output = tmp_path / 'logs'
    output.mkdir()
    (tmp_path / 'locks').mkdir()
    script = tmp_path / 'launch.sh'
    script.write_text('set -euo pipefail\nflock() { return 0; }\n'
                      'mode=primary\n' + functions + '''
launch 0 gcn concurrent; a=$pid
launch 1 graphsage concurrent; b=$pid
failed=0
join "$a" gcn || failed=1
join "$b" graphsage || failed=1
exit "$failed"
''', newline='\n')
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else 'bash'
    # py is an executable shell wrapper which ignores Python argv preceding --backbone.
    env = dict(os.environ, py=stub.as_posix(), output=output.as_posix(),
               execution=tmp_path.as_posix(), FAIL_WORKER=fail_worker)
    if os.name != 'nt':
        stub.chmod(0o755)
    process = subprocess.Popen([bash, script.as_posix()], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
    import queue
    import threading
    lines = queue.Queue()

    def collect():
        for line in process.stdout:
            lines.put(line)

    thread = threading.Thread(target=collect)
    thread.start()
    try:
        observed = [lines.get(timeout=15).decode(), lines.get(timeout=15).decode()]
        assert all('[TRAIN]' in line for line in observed)
        assert process.poll() is None  # output is live, before releasing either worker
        deadline = time.monotonic() + 10
        while len(list(output.glob('*.start'))) != 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(list(output.glob('*.start'))) == 2
    finally:
        (output / 'release').touch()
        process.wait(timeout=20)
        thread.join(timeout=5)
    assert process.returncode == (1 if fail_worker else 0)
    for backbone in ('gcn', 'graphsage'):
        assert (output / f'concurrent_{backbone}.stdout.log').read_text().count('[TRAIN]') == 1
        assert (output / f'concurrent_{backbone}.stderr.log').read_text() == f'stderr-{backbone}\n'
    assert len((output / 'exits.tsv').read_text().splitlines()) == 2
    assert len((output / 'pids.tsv').read_text().splitlines()) == 2
    assert process.stderr.read().count(b'stderr-') == 2


def test_metric_result_json_unchanged(capsys):
    row = {'node_id': 2, 'fid_plus': 1.0, 'fid_minus': 0.0,
           'one_minus_fid_minus': 1.0, 'remove_delta': 0.4,
           'retain_delta': 0.2, 'retain_probability_ratio': 0.8, 'sparsity': 0.8}
    before = json.dumps(row)
    runner.progress_result('GCN', 42, 'FINAL_EVAL', row, 1, 1, 1.5)
    assert json.dumps(row) == before


def test_logging_does_not_modify_serialized_payload(tmp_path, capsys):
    path = tmp_path / 'epoch_001.pt'
    payload = {'epoch_completed': 1, 'algorithm_state_dict': {'weight': torch.ones(2)},
               'optimizer_state_dict': {'step': torch.tensor(1)},
               'rng_state_cpu': torch.get_rng_state(), 'epoch_rows': [{'mean_loss': 0.25}]}
    runner.atomic_torch_save(path, payload)
    original = path.read_bytes()
    runner.progress('GCN', 42, 'TRAIN', loss=payload['epoch_rows'][0]['mean_loss'],
                    checkpoint=path, epoch='1/30')
    runner.atomic_torch_save(path, payload)
    assert path.read_bytes() == original
    runner.exact_state(torch.load(path, weights_only=False), payload, 'fixture')
