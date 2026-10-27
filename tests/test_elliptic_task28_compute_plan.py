import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / 'docs/elliptic_pgexplainer_step04_compute_plan.json').read_text())


def test_compute_plan_schema_and_authorizations():
    assert PLAN['schema_version'] == 1
    assert PLAN['primary_training_authorized'] is True
    assert PLAN['stability_training_authorized'] is False
    assert PLAN['final_evaluation_authorized'] is False
    assert PLAN['epochs'] == 30
    assert (PLAN['train_count'], PLAN['validation_count'], PLAN['final_count']) == (2000, 500, 200)
    assert len(PLAN['implementation_commit']) == 40
    assert len(PLAN['runs']) == 12
    assert {(r['backbone'], r['seed']) for r in PLAN['runs']} == {
        (b, s) for b in ['gcn', 'graphsage'] for s in [42, 0, 1, 2, 3, 4]
    }
    for key in ['artifact', 'run_dir', 'manifest', 'complete']:
        assert len({r[key] for r in PLAN['runs']}) == 12


@pytest.mark.parametrize('run', PLAN['runs'], ids=lambda r: f"{r['backbone']}-{r['seed']}")
def test_compute_plan_commands_and_paths(run):
    args = run['argv']
    assert args == [
        'scripts/run_elliptic_pgexplainer_training.py', '--config',
        'configs/elliptic/elliptic_pgexplainer_training.yaml',
        '--backbone', run['backbone'], '--seed', str(run['seed']), '--device', 'cuda',
    ]
    assert (ROOT / args[0]).is_file()
    assert (ROOT / args[2]).is_file()
    stem = f"elliptic_pg_{run['backbone']}_seed{run['seed']}"
    assert run['artifact'] == f'artifacts/explainers/{stem}.pt'
    assert run['complete'] == f'artifacts/explainers/{stem}_run/COMPLETE.json'
    assert run['command'] in (ROOT / 'docs/elliptic_pgexplainer_step04_runbook.ps1').read_text()


def test_compute_plan_frozen_hashes_and_configuration():
    for name, digest in PLAN['file_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    config = yaml.safe_load(
        (ROOT / 'configs/elliptic/elliptic_pgexplainer_training.yaml').read_text()
    )['task28_training']
    assert config['epochs'] == PLAN['epochs']
    assert config['selected_edge_ratio'] == PLAN['selected_edge_ratio'] == 0.2
    assert config['rounding_rule'] == PLAN['rounding_rule']
    assert config['target_semantics'] == PLAN['target_semantics']


def test_compute_plan_estimates_are_six_plus_six():
    estimate = PLAN['runtime_estimate']
    values = [v * 2000 * 30 / 3600 for v in estimate['mean_step_seconds'].values()]
    assert estimate['primary_training_hours'] == pytest.approx(sum(values))
    assert estimate['all_12_training_hours'] == pytest.approx(6 * sum(values))
    assert estimate['planning_hours_with_50_percent_margin'] == pytest.approx(9 * sum(values))
    disk = PLAN['disk_estimate']
    assert disk['total_12_runs_mib'] == 12 * (
        disk['assumed_mib_per_checkpoint_file'] * disk['files_per_run_including_atomic_temp']
        + disk['logs_and_metadata_mib_per_run']
    )
