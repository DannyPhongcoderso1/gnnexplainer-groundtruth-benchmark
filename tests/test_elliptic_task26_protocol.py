import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def node_ids(path: str) -> set[str]:
    return {row["node_id"] for row in read_csv(path)}


def freeze() -> dict:
    return json.loads((ROOT / "docs/elliptic_targets_freeze.json").read_text(encoding="utf-8"))


def test_elliptic_positive_class_and_no_gt_explanation_metrics() -> None:
    data = freeze()

    mapping = data["positive_class_semantic_numeric_mapping"]
    assert mapping == {
        "semantic_positive_class": "ILLICIT transaction",
        "positive_label_index": 1,
        "licit_label_index": 0,
        "unknown_label_index": 2,
    }
    assert data["metric_formulas"]["disabled_explanation_metrics"] == [
        "Edge AUROC",
        "Edge AP",
    ]
    assert data["selected_edge_protocol"]["selected_edge_ratio"] == 0.2
    assert data["selected_edge_protocol"]["selected_edge_count"] == (
        "max(1, ceil(candidate_edge_count * 0.20))"
    )


def test_elliptic_model_parity_and_perturbation_smoke_pass() -> None:
    data = freeze()

    parity = data["parity_tolerances_and_results"]
    assert parity["atol"] == 1e-6
    assert parity["rtol"] == 0.0
    assert parity["gcn"]["pass"] is True
    assert parity["graphsage"]["pass"] is True
    assert parity["gcn"]["prediction_mismatch_count"] == 0
    assert parity["graphsage"]["prediction_mismatch_count"] == 0
    assert parity["gcn"]["state_dict_hash"]["sha256"]
    assert parity["graphsage"]["state_dict_hash"]["sha256"]

    smoke = data["perturbation_smoke_evidence"]
    assert smoke["gcn"]["pass"] is True
    assert smoke["graphsage"]["pass"] is True
    assert smoke["edge_input_hash_before"] != smoke["edge_input_hash_after_remove"]
    assert smoke["edge_input_hash_before"] != smoke["edge_input_hash_after_retain"]


def test_elliptic_target_counts_hashes_and_disjointness_are_frozen() -> None:
    data = freeze()
    final = data["final_target_rule_seed_count_hash_distributions"]
    train = data["pg_common_train_rule_seed_count_hash_distributions"]
    val = data["pg_common_val_rule_seed_count_hash_distributions"]

    assert final["count"] == 200
    assert train["count"] == 2000
    assert val["count"] == 500
    assert all(value == 0 for value in data["set_intersection_audit"].values())
    assert final["hash"]
    assert train["hash"]
    assert val["hash"]


def test_elliptic_unknown_labels_are_excluded_from_target_files() -> None:
    data = freeze()
    for key, item in data["artifact_inventory"].items():
        if not key.startswith("configs"):
            continue
        rows = read_csv(item["path"])
        assert len(rows) == item["row_count"]
        assert {row["label"] for row in rows} <= {"0", "1"}
        assert all(row["candidate_mapping_valid"] == "True" for row in rows)


def test_elliptic_common_pg_pool_identity_files_match_backbone_eligibility() -> None:
    common_train = node_ids("configs/targets/elliptic_pg_common_train.csv")
    common_val = node_ids("configs/targets/elliptic_pg_common_val.csv")
    gcn_train = node_ids("configs/targets/elliptic_pg_gcn_train_eligible.csv")
    sage_train = node_ids("configs/targets/elliptic_pg_graphsage_train_eligible.csv")
    gcn_val = node_ids("configs/targets/elliptic_pg_gcn_val_eligible.csv")
    sage_val = node_ids("configs/targets/elliptic_pg_graphsage_val_eligible.csv")

    assert common_train <= gcn_train
    assert common_train <= sage_train
    assert common_val <= gcn_val
    assert common_val <= sage_val
    assert not (common_train & common_val)


def test_elliptic_dataset_split_semantics_and_required_freeze_keys() -> None:
    data = freeze()
    dataset = data["dataset_split_hashes_and_semantics"]["semantics"]

    assert data["task_id"] == "TASK26"
    assert data["status"] == "PASS"
    assert data["task27_authorized"] is False
    assert data["no_final_explanations_run"] is True
    assert dataset["total_nodes"] == 203769
    assert dataset["feature_count"] == 165
    assert dataset["total_directed_edge_entries"] == 468710
    assert dataset["temporal_semantics_pass"] is True
    assert dataset["splits"]["train"]["timestep_max"] < 30
    assert dataset["splits"]["validation"]["timestep_min"] >= 30
    assert dataset["splits"]["validation"]["timestep_max"] < 40
    assert dataset["splits"]["test"]["timestep_min"] >= 40


def test_elliptic_metric_formula_definitions_are_frozen() -> None:
    formulas = freeze()["metric_formulas"]

    assert formulas["probability_reference_class"] == "original predicted class"
    assert formulas["fid_plus"].startswith("1 when removing selected edges")
    assert formulas["fid_minus"].startswith("1 when retaining only selected edges")
    assert formulas["one_minus_fid_minus"] == "1 - Fid-"
    assert formulas["remove_probability_delta"] == (
        "original_probability - remove_probability"
    )
    assert formulas["retain_probability_delta"] == (
        "original_probability - retain_probability"
    )
    assert "original_probability == 0" in formulas["retain_probability_ratio"]
    assert formulas["runtime_boundaries"]["GNNExplainer"]
    assert formulas["stability_sources"]["PGExplainer"]
