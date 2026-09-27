import ast
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest
import torch

import scripts.run_elliptic_pgexplainer_training as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('ending', [b'\n', b'\r\n', b'\r'])
def test_text_line_endings_are_canonical(tmp_path, ending):
    path = tmp_path / 'source.py'
    path.write_bytes(ending.join([b'x = 1', b'# text', b'']))
    assert runner.canonical_text_sha256(path) == hashlib.sha256(b'x = 1\n# text\n').hexdigest()


@pytest.mark.parametrize('changed', [b'x = 2\n', b'x = 1 \n', b'x = 1', b'\xef\xbb\xbfx = 1\n'])
def test_real_text_content_is_not_normalized(tmp_path, changed):
    path = tmp_path / 'config.yaml'
    path.write_bytes(b'x = 1\n')
    first = runner.canonical_text_sha256(path)
    path.write_bytes(changed)
    assert runner.canonical_text_sha256(path) != first


def test_binary_hash_is_still_raw_and_not_accepted_as_text(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    payload = b'\x00\xff\r\n\r binary'
    path.write_bytes(payload)
    assert runner.sha256(path) == hashlib.sha256(payload).hexdigest()
    assert runner.sha256(path) != hashlib.sha256(payload.replace(b'\r\n', b'\n')).hexdigest()
    with pytest.raises(ValueError):
        runner.canonical_text_sha256(path)


def test_candidate_mapping_observed_newline_diagnosis():
    path = Path(inspect.getfile(runner.computation_subgraph_edge_mask))
    raw = path.read_bytes()
    lf = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    assert hashlib.sha256(lf).hexdigest() == (
        '8a39548748d7c97d3b29288683837687267dd7d2f03ed02dc4a48876e44b1391')
    assert hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest() == (
        'ab5d7809fe235c1d5a5c1e083e4c77d5a19acfadd0d70eb38d2ba396972c7bc5')


def test_local_hard_compatibility_matches_plan():
    plan = json.loads((ROOT / 'docs/elliptic_pgexplainer_step04_compute_plan.json').read_text())
    actual = runner.hard_compatibility(torch.device('cuda'))  # identity only, no CUDA work
    assert actual.pop('git_commit') == runner.git_value('rev-parse', 'HEAD')
    assert actual == plan['hard_compatibility_without_git']
    assert actual['source_hash_policy'] == 'utf8-crlf-cr-to-lf-v1'


def test_training_and_integrity_functions_ast_unchanged():
    original = subprocess.check_output(
        ['git', 'show', '61c554bef5f462610ff8c2765ac8488d72b9e48d:'
         'scripts/run_elliptic_pgexplainer_training.py'], cwd=ROOT)

    def definitions(source):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name not in ('canonical_text_sha256', 'hard_compatibility')}

    assert definitions(original) == definitions(Path(runner.__file__).read_bytes())
