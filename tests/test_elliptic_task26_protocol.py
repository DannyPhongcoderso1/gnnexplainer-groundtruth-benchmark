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

    assert data["positive_class_semantics"] == {
        "semantic_positive_class": "ILLICIT transaction",
        "positive_label_index": 1,
        "licit_label_index": 0,
        "unknown_label_index": 2,
    }
    assert "disabled" in data["metric_protocol"]["edge_auroc"]
    assert "disabled" in data["metric_protocol"]["edge_ap"]
    assert data["metric_protocol"]["selected_edge_ratio"] == 0.2


def test_elliptic_model_parity_and_perturbation_smoke_pass() -> None:
    data = freeze()

    assert data["model_parity"]["gcn"]["passes"] is True
    assert data["model_parity"]["graphsage"]["passes"] is True
    assert data["model_parity"]["gcn"]["prediction_mismatch_count"] == 0
    assert data["model_parity"]["graphsage"]["prediction_mismatch_count"] == 0
    assert data["perturbation_smoke"]["gcn_compatible_cached_false"]["passes"] is True
    assert data["perturbation_smoke"]["graphsage_wrapper"]["passes"] is True


def test_elliptic_target_counts_hashes_and_disjointness_are_frozen() -> None:
    data = freeze()
    targets = data["targets"]

    assert targets["counts"]["final_test"] == 200
    assert targets["counts"]["common_train"] == 2000
    assert targets["counts"]["common_val"] == 500
    assert all(value == 0 for value in targets["disjointness"].values())
    for item in targets["files"].values():
        assert item["sha256"]
        assert item["count"] > 0


def test_elliptic_unknown_labels_are_excluded_from_target_files() -> None:
    data = freeze()
    for item in data["targets"]["files"].values():
        rows = read_csv(item["path"])
        assert len(rows) == item["count"]
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
