from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/elliptic_pgexplainer_step03_evidence_expansion_manifest.json"
ORIGINAL_MANIFEST_PATH = ROOT / "docs/elliptic_pgexplainer_step03_pilot_manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_step03_evidence_manifest_required_sections() -> None:
    manifest = load_manifest()

    for key in [
        "scope",
        "git",
        "original_pilot_artifacts",
        "new_instrumentation_script",
        "frozen_input_config_hashes",
        "exact_pilot_node_identities",
        "backbones",
        "cpu_resume_fixture",
        "pool_and_leakage",
        "path_isolation",
        "validation_results",
        "limitations",
        "remaining_blockers",
        "stop_boundary",
    ]:
        assert key in manifest


def test_step03_evidence_uses_exact_original_pilot_nodes() -> None:
    manifest = load_manifest()
    original = json.loads(ORIGINAL_MANIFEST_PATH.read_text(encoding="utf-8"))

    for backbone in ["gcn", "graphsage"]:
        identity = manifest["exact_pilot_node_identities"][backbone]
        assert identity["old_train"] == original["backbones"][backbone]["pilot_train_nodes"]
        assert identity["new_train"] == original["backbones"][backbone]["pilot_train_nodes"]
        assert identity["old_validation"] == original["backbones"][backbone]["pilot_val_nodes"]
        assert identity["new_validation"] == original["backbones"][backbone]["pilot_val_nodes"]
        assert identity["train_mismatches"] == 0
        assert identity["validation_mismatches"] == 0


def test_step03_evidence_target_semantics_and_parameters_pass() -> None:
    manifest = load_manifest()

    for item in manifest["backbones"].values():
        assert item["target_evidence"]["train"]["target_prediction_mismatch_count"] == 0
        assert item["target_evidence"]["validation"]["target_prediction_mismatch_count"] == 0
        assert item["target_evidence"]["train"]["ground_truth_used_as_target_source"] is False
        assert item["target_evidence"]["validation"]["ground_truth_used_as_target_source"] is False
        assert item["pgexplainer_parameter_update"]["pass"] is True
        assert item["pgexplainer_parameter_update"]["changed_tensor_count"] > 0
        assert item["pgexplainer_parameter_update"]["maximum_absolute_difference"] > 0
        assert item["backbone_immutability"]["pass"] is True
        assert item["backbone_immutability"]["changed_tensor_count"] == 0
        assert item["backbone_immutability"]["maximum_absolute_difference"] == 0
        assert item["backbone_immutability"]["gradient_present_tensor_count"] == 0


def test_step03_evidence_optimizer_reload_resume_and_leakage_pass() -> None:
    manifest = load_manifest()

    for item in manifest["backbones"].values():
        optimizer = item["optimizer_isolation"]["after_training"]
        assert item["optimizer_isolation"]["pass"] is True
        assert (
            optimizer["optimizer_pgexplainer_shared_parameter_id_count"]
            == optimizer["optimizer_parameter_tensor_count"]
        )
        assert optimizer["optimizer_backbone_shared_parameter_id_count"] == 0
        assert optimizer["cross_backbone_shared_trainable_parameter_id_count"] == 0
        assert item["reload_parity"]["pass"] is True
        assert item["reload_parity"]["parameter_comparison"]["mismatch_tensor_count"] == 0
        assert item["reload_parity"]["parameter_comparison"]["maximum_absolute_difference"] == 0
        assert item["reload_parity"]["optimizer_comparison"]["maximum_absolute_difference"] == 0
        assert item["resume_parity"]["pass"] is True
        assert item["resume_parity"]["maximum_absolute_loss_difference"] <= 1e-7
        assert (
            item["resume_parity"]["pgexplainer_tensor_comparison"][
                "maximum_absolute_difference"
            ]
            <= 1e-7
        )
        assert (
            item["resume_parity"]["optimizer_tensor_comparison"][
                "maximum_absolute_difference"
            ]
            <= 1e-7
        )
        assert (
            item["resume_parity"]["checkpoint_reload_algorithm_state_exact"][
                "maximum_absolute_difference"
            ]
            == 0
        )
        assert (
            item["resume_parity"]["checkpoint_reload_optimizer_state_exact"][
                "maximum_absolute_difference"
            ]
            == 0
        )

    cpu_resume = manifest["cpu_resume_fixture"]
    assert cpu_resume["pass"] is True
    assert cpu_resume["final_pg_parameter_comparison"]["maximum_absolute_difference"] == 0
    assert cpu_resume["final_optimizer_state_comparison"]["maximum_absolute_difference"] == 0
    assert cpu_resume["maximum_absolute_loss_difference"] == 0

    leakage = manifest["pool_and_leakage"]
    assert leakage["all_train_ids_belong_to_frozen_train_pool"] is True
    assert leakage["all_validation_ids_belong_to_frozen_validation_pool"] is True
    assert leakage["train_validation_pilot_overlap"] == 0
    assert leakage["pilot_final_target_intersection"] == 0
    assert leakage["out_of_pool_id_count"] == 0
    assert leakage["final_target_metadata"]["final_target_model_outputs_accessed"] is False
    assert leakage["final_target_metadata"]["final_target_explanations_accessed"] is False
    assert leakage["final_target_metadata"]["final_target_metrics_accessed"] is False


def test_step03_evidence_path_isolation() -> None:
    manifest = load_manifest()
    isolation = manifest["path_isolation"]

    assert isolation["canonical_epoch_30_checkpoint_created"] is False
    assert isolation["canonical_completion_marker_created"] is False
    assert isolation["pilot_checkpoint_cannot_satisfy_full_run_config_identity"] is True
    assert "task28_step03_evidence_expansion" in isolation["pilot_outputs_root"]
