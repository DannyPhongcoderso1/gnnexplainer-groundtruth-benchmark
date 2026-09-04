from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "audit" / "backbone_fairness_matrix.csv"
REPORT_PATH = ROOT / "docs" / "audit" / "backbone_architecture_training_audit.md"


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_audit03_deliverables_exist_and_cover_all_backbones() -> None:
    assert CSV_PATH.exists()
    assert REPORT_PATH.exists()

    rows = load_rows()
    assert len(rows) == 6
    assert {(row["dataset"], row["backbone"]) for row in rows} == {
        ("BA-House", "GCN"),
        ("BA-House", "GraphSAGE"),
        ("ShapeGGen", "GCN"),
        ("ShapeGGen", "GraphSAGE"),
        ("Elliptic", "GCN"),
        ("Elliptic", "GraphSAGE"),
    }
    assert len(rows[0]) == len(set(rows[0]))


def test_audit03_required_expansion_columns_exist() -> None:
    rows = load_rows()
    required = {
        "bias",
        "bias_tensor_names",
        "bias_tensor_count",
        "bias_parameter_elements",
        "class_weight_source",
        "class_weight_formula",
        "class_weight_vector",
        "class_weight_population",
        "class_weight_same_within_dataset",
        "best_checkpoint_epoch",
        "training_stop_epoch",
        "updates_per_epoch",
        "updates_in_selected_checkpoint",
        "total_executed_training_updates",
        "discarded_updates_after_best",
        "hyperparameter_search_updates",
        "programmatic_test_based_selection",
        "test_metric_computed_per_trial",
        "test_metric_logged_or_displayed_per_trial",
        "human_observable_test_exposure",
        "evidence_that_human_used_test_for_tuning",
        "hyperparameter_provenance_classification",
        "calibration_ece_bin_count",
        "calibration_ece_bin_policy",
        "calibration_brier_definition",
        "calibration_nll_definition",
        "target_selection_flow_status",
    }
    assert required <= set(rows[0])


def test_audit03_backbone_comparisons_are_depth_matched_capacity_unmatched() -> None:
    rows = load_rows()
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["backbone"]] = row

    for pair in by_dataset.values():
        assert pair["GCN"]["layers"] == pair["GraphSAGE"]["layers"]
        assert pair["GCN"]["parameter_count"] != pair["GraphSAGE"]["parameter_count"]


def test_audit03_elliptic_is_not_hyperparameter_or_update_matched() -> None:
    rows = {
        row["backbone"]: row
        for row in load_rows()
        if row["dataset"] == "Elliptic"
    }

    assert rows["GCN"]["hidden_size"] != rows["GraphSAGE"]["hidden_size"]
    assert rows["GCN"]["dropout"] != rows["GraphSAGE"]["dropout"]
    assert rows["GCN"]["learning_rate"] != rows["GraphSAGE"]["learning_rate"]
    assert (
        rows["GCN"]["total_updates_to_selected_checkpoint"]
        != rows["GraphSAGE"]["total_updates_to_selected_checkpoint"]
    )
    assert rows["GCN"]["output_classes"] == "2"
    assert rows["GraphSAGE"]["output_classes"] == "2"


def test_audit03_bias_and_class_weight_evidence_is_explicit() -> None:
    rows = load_rows()
    for row in rows:
        assert row["bias"] in {"TRUE", "FALSE"}
        assert row["bias"] != "PyG defaults"
        assert int(row["bias_tensor_count"]) >= 0
        assert int(row["bias_parameter_elements"]) >= 0
        assert row["class_weight_vector"]
        assert row["class_weight_vector"] != ""

    synthetic = [row for row in rows if row["dataset"] != "Elliptic"]
    assert all(row["class_weight_source"].endswith("::_class_weights") for row in synthetic)
    assert all(row["class_weight_vector"] != "not_used" for row in synthetic)

    elliptic = [row for row in rows if row["dataset"] == "Elliptic"]
    assert all(row["class_weight_vector"] == "not_used" for row in elliptic)


def test_audit03_update_accounting_is_separated() -> None:
    rows = {(row["dataset"], row["backbone"]): row for row in load_rows()}

    assert rows[("BA-House", "GCN")]["best_checkpoint_epoch"] == "10"
    assert rows[("BA-House", "GCN")]["training_stop_epoch"] == "70"
    assert rows[("BA-House", "GCN")]["updates_in_selected_checkpoint"] == "10"
    assert rows[("BA-House", "GCN")]["total_executed_training_updates"] == "70"

    assert rows[("ShapeGGen", "GraphSAGE")]["best_checkpoint_epoch"] == "263"
    assert rows[("ShapeGGen", "GraphSAGE")]["training_stop_epoch"] == "300"
    assert rows[("ShapeGGen", "GraphSAGE")]["discarded_updates_after_best"] == "37"

    assert rows[("Elliptic", "GCN")]["hyperparameter_search_updates"] == "UNKNOWN"
    assert rows[("Elliptic", "GraphSAGE")]["hyperparameter_search_updates"] == "UNKNOWN"


def test_audit03_elliptic_test_exposure_boundary_is_precise() -> None:
    rows = [row for row in load_rows() if row["dataset"] == "Elliptic"]
    assert rows
    for row in rows:
        assert row["programmatic_test_based_selection"] == "FALSE"
        assert row["test_metric_computed_per_trial"] == "TRUE"
        assert row["test_metric_logged_or_displayed_per_trial"] == "TRUE"
        assert row["human_observable_test_exposure"] == "TRUE"
        assert row["evidence_that_human_used_test_for_tuning"] == "UNKNOWN"
        trace = json.loads(row["elliptic_objective_trace"])
        assert trace["metric_returned_to_optuna"] == "validation_auprc"
        assert trace["test_metric_enters_objective"] == "FALSE"
        assert trace["test_metric_enters_pruning"] == "FALSE"


def test_audit03_hyperparameter_provenance_uses_allowed_vocabulary() -> None:
    allowed = {
        "PAPER_REPORTED_EXACT_MATCH",
        "INHERITED_BASELINE_CONFIGURATION",
        "LOCALLY_TUNED_BY_OPTUNA",
        "FINAL_REFIT_DERIVED",
        "PROJECT_DEFINED",
        "UNKNOWN",
        "not_applicable",
    }
    for row in load_rows():
        payload = json.loads(row["hyperparameter_provenance_classification"])
        for value in payload.values():
            classification = str(value).split(":", 1)[0]
            assert classification in allowed


def test_audit03_calibration_and_target_boundaries_are_explicit() -> None:
    rows = load_rows()
    for row in rows:
        assert row["calibration_probability_source"] == "softmax(logits)"
        assert row["calibration_ece_bin_count"] == "10"
        assert row["calibration_ece_bin_policy"] == "equal_width_bins_over_predicted_confidence"
        assert "one_hot_label" in row["calibration_brier_definition"]
        assert row["target_selection_flow_status"] == "PENDING_AUDIT04"


def test_audit03_report_keeps_stop_boundary() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "AUDIT03 VERDICT: CONDITIONAL PASS" in report
    assert "Task28 Step04: NOT AUTHORIZED" in report
    assert "Task29: NOT AUTHORIZED" in report
    assert "Do not claim a pure causal effect" in report
    assert "Exact target eligibility and sampling flow: `PENDING_AUDIT04`" in report
    assert "no programmatic test-based selection" in report
    assert "observational test exposure" in report
