from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest() -> dict:
    return json.loads(
        (ROOT / "docs/elliptic_gnnexplainer_freeze.json").read_text(encoding="utf-8")
    )


def test_task27_manifest_freezes_elliptic_gnnexplainer_scope() -> None:
    data = manifest()

    assert data["task"] == "TASK27 Elliptic GNNExplainer"
    assert data["status"] == "success"
    assert data["inputs"]["target_count"] == 200
    assert data["protocol"]["primary_seed"] == 42
    assert data["protocol"]["stability_seeds"] == [0, 1, 2, 3, 4]
    assert data["protocol"]["selected_edge_ratio"] == 0.2
    assert data["protocol"]["rounding_rule"] == (
        "max(1, ceil(candidate_edge_count * 0.20))"
    )
    assert data["protocol"]["explanation_type"] == "phenomenon"


def test_task27_disables_elliptic_edge_ground_truth_metrics() -> None:
    data = manifest()

    assert data["protocol"]["disabled_explanation_metrics"] == ["Edge AUROC", "Edge AP"]
    assert data["acceptance"]["no_auroc_ap_explanation_metrics"] is True
    for result in data["backbones"].values():
        metric_names = {row["metric"] for row in result["primary_summary"]}
        assert "edge_auroc" not in metric_names
        assert "edge_ap" not in metric_names


def test_task27_primary_and_stability_counts_are_complete() -> None:
    data = manifest()

    for result in data["backbones"].values():
        assert result["primary_success_count"] == 200
        assert result["primary_failure_count"] == 0
        assert result["stability_seed_success_count"] == 1000
        assert result["stability_seed_failure_count"] == 0
        assert result["stability_node_success_count"] == 200
        assert result["stability_pairwise_rows"] == 2000
        assert result["checkpoint_unchanged"] is True
        assert result["model_weights_unchanged"] is True


def test_task27_summary_schema_and_hashes_are_recorded() -> None:
    data = manifest()

    expected_metrics = {
        "candidate_edge_count",
        "selected_edge_count",
        "fid_plus",
        "fid_minus",
        "one_minus_fid_minus",
        "remove_probability_delta",
        "retain_probability",
        "retain_probability_delta",
        "retain_probability_ratio",
        "sparsity",
        "generation_runtime_seconds",
        "evaluation_runtime_seconds",
    }
    for result in data["backbones"].values():
        metric_names = {row["metric"] for row in result["primary_summary"]}
        assert expected_metrics <= metric_names
    assert data["output_hashes"]
    assert all(len(value) == 64 for value in data["output_hashes"].values())
