"""Path/import invariants for the organized release; no experiment execution."""

import json
from pathlib import Path

import pytest

from gnnexplainer_benchmark.models import GCN, GraphSAGE
from gnnexplainer_benchmark.models.elliptic import EllipticGCN, EllipticSAGE
from gnnexplainer_benchmark.models.synthetic import GCN as SyntheticGCN
from gnnexplainer_benchmark.models.synthetic import GraphSAGE as SyntheticGraphSAGE
from gnnexplainer_benchmark.utils.paths import resolve_repository_path
from scripts.prepare_elliptic_task26 import EllipticGCN as HistoricalEllipticGCN
from scripts.prepare_elliptic_task26 import EllipticSAGE as HistoricalEllipticSAGE

ROOT = Path(__file__).resolve().parents[1]


def test_backbone_imports_are_identical_and_families_separate():
    assert GCN is SyntheticGCN and GraphSAGE is SyntheticGraphSAGE
    assert EllipticGCN is HistoricalEllipticGCN and EllipticSAGE is HistoricalEllipticSAGE
    assert GCN is not EllipticGCN and GraphSAGE is not EllipticSAGE


def test_every_former_published_artifact_resolves_to_one_current_file():
    catalog = json.loads((ROOT / "docs/provenance/external_assets.json").read_text("utf-8"))
    for entry in catalog["required_artifacts"]:
        old = entry.get("previous_published_relative_path")
        if old:
            expected = ROOT / entry["published_relative_path"]
            assert resolve_repository_path(old, ROOT) == expected
            assert resolve_repository_path(ROOT / old, ROOT) == expected
            assert expected.is_file() and not (ROOT / old).exists()


def test_custom_absolute_input_is_not_rewritten(tmp_path):
    path = tmp_path / "custom.pt"
    assert resolve_repository_path(path, ROOT) == path


def test_catalog_cannot_redirect_outside_repository(tmp_path):
    folder = tmp_path / "docs/provenance"
    folder.mkdir(parents=True)
    (folder / "repository_paths.json").write_text(json.dumps({"paths": {"old.pt": "../x"}}))
    with pytest.raises(ValueError, match="escapes"):
        resolve_repository_path("old.pt", tmp_path)


def test_canonical_api_exports_refer_to_one_implementation():
    from gnnexplainer_benchmark.explain import run_gnnexplainer_for_node as public
    from gnnexplainer_benchmark.explainers.gnnexplainer import run_gnnexplainer_for_node

    assert public is run_gnnexplainer_for_node
    assert GCN is SyntheticGCN and GraphSAGE is SyntheticGraphSAGE
