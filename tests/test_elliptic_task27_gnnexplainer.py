from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest() -> dict:
    return json.loads(
        (ROOT / "docs/elliptic_gnnexplainer_freeze.json").read_text(encoding="utf-8")
    )


def raw_inventory() -> dict:
    return json.loads(
        (ROOT / "docs/elliptic_gnnexplainer_raw_artifact_inventory.json").read_text(
            encoding="utf-8"
        )
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
    assert data["protocol"]["probability_reference_class"] == "original predicted class"


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


def test_task27_git_provenance_uses_non_self_referential_schema() -> None:
    data = manifest()

    assert data["git"] == {
        "branch": "main",
        "execution_start_commit": "a68b933490ef6b94d92dd3b55f2bddc8bf139226",
        "task27_initial_result_commit": "940ca4c51b24c845d3a94477f63280103825c1e4",
    }
    assert "commit" not in data["git"]


def test_task27_optimization_target_is_original_prediction() -> None:
    data = manifest()
    target = data["protocol"]["optimization_target"]

    assert target["explanation_type"] == "phenomenon"
    assert "prediction_vector(model, data)" in target["target_source"]
    assert "target=target_vector" in target["code_path"]
    assert target["target_dtype"] == "torch.int64 / long class-index tensor from argmax"
    assert "not chosen from ground-truth" in target["conclusion"]
    for audit in target["by_backbone"].values():
        assert audit["checked_rows"] == 1200
        assert audit["target_prediction_matches"] == 1200
        assert audit["target_prediction_mismatches"] == 0
        assert audit["target_ground_truth_mismatches"] > 0


def test_task27_raw_inventory_covers_all_existing_edge_masks() -> None:
    inventory = raw_inventory()

    assert inventory["status"] == "success"
    assert len(inventory["runs"]) == 2400
    for audit in inventory["aggregate"].values():
        assert audit["expected_raw_runs"] == 1200
        assert audit["actual_raw_runs"] == 1200
        assert audit["primary_count"] == 200
        assert audit["stability_count"] == 1000
        assert audit["missing_count"] == 0
        assert audit["duplicate_count"] == 0
        assert audit["failed_count"] == 0
        assert audit["candidate_score_mismatches"] == 0
        assert audit["full_score_edge_index_mismatches"] == 0
        assert audit["selected_count_violations"] == 0
        assert audit["nan_inf_violations"] == 0
        assert audit["selected_outside_candidate_violations"] == 0


def test_task27_raw_inventory_preserves_continuous_candidate_scores() -> None:
    inventory = raw_inventory()

    for row in inventory["runs"]:
        assert row["status"] == "success"
        assert row["continuous_score_count"] == row["candidate_edge_count"]
        assert row["selected_edge_count"] == row["expected_selected_count"]
        assert row["selected_edge_ratio"] == 0.2
        assert row["edge_score_nan_count"] == 0
        assert row["edge_score_inf_count"] == 0
        assert row["candidate_score_length_mismatch"] is False
        assert row["candidate_edge_duplicate_count"] == 0
        assert row["selected_edge_duplicate_count"] == 0
        assert len(row["raw_artifact_sha256"]) == 64
        assert len(row["candidate_edge_ids_sha256"]) == 64


def test_task27_metrics_csv_audit_and_identities_pass() -> None:
    data = manifest()

    for audit in data["metrics_csv_audit"].values():
        files = audit["files"]
        assert files["per_node_metrics"]["row_count"] == 200
        assert files["stability_seed_metrics"]["row_count"] == 1000
        assert files["stability_pairwise_jaccard"]["row_count"] == 2000
        assert files["stability_per_node_jaccard"]["row_count"] == 200
        assert files["summary_metrics"]["row_count"] == 14
        assert files["per_node_metrics"]["nan_inf"] == 0
        assert files["stability_seed_metrics"]["nan_inf"] == 0
        assert all(file_audit["duplicates"] == 0 for file_audit in files.values())

        identities = audit["metric_identity_checks"]
        assert identities["checked_rows"] == 1200
        assert identities["violation_count"] == 0
        assert identities["maximum_absolute_error"] <= 1e-9
        assert identities["required_metric_null_nan_inf_count"] == 0


def test_task27_stability_pair_structure_is_frozen() -> None:
    data = manifest()

    for audit in data["metrics_csv_audit"].values():
        stability = audit["stability_pair_audit"]
        assert stability["target_count"] == 200
        assert stability["seed_set"] == [0, 1, 2, 3, 4]
        assert stability["uses_seed42_as_stability_seed"] is False
        assert stability["pair_rows"] == 2000
        assert stability["expected_pair_rows"] == 2000
        assert stability["per_node_rows"] == 200
        assert stability["self_pair_violations"] == 0
        assert stability["duplicate_unordered_pair_violations"] == 0
        assert stability["wrong_seed_violations"] == 0
        assert stability["out_of_range_violations"] == 0
        assert stability["nodes_with_wrong_pair_count"] == 0


def test_task27_runtime_decomposition_keeps_stage_boundaries() -> None:
    data = manifest()

    runtime = data["runtime_decomposition"]
    assert runtime["aggregate"]["batch_wall_clock_total_seconds"] == data[
        "total_runtime_seconds"
    ]
    assert runtime["aggregate"]["sum_recorded_stage_runtimes_seconds"] > 0
    assert runtime["aggregate"]["unattributed_difference_seconds"] > 0
    for audit in runtime["by_backbone"].values():
        assert audit["primary_generation"]["n"] == 200
        assert audit["primary_evaluation"]["n"] == 200
        assert audit["stability_generation"]["n"] == 1000
        assert audit["stability_evaluation"]["n"] == 1000
        assert audit["primary_generation"]["boundary"] == "generation_runtime_seconds"
        assert audit["primary_evaluation"]["boundary"] == "evaluation_runtime_seconds"
        assert audit["primary_generation"]["mean"] != audit["primary_evaluation"]["mean"]
