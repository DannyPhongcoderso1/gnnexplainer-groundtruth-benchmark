from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "docs/audit/dataset_split_preprocessing_inventory.json"


def load_inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_audit02_inventory_required_expansion_keys() -> None:
    inventory = load_inventory()
    elliptic = inventory["datasets"]["elliptic"]

    for key in [
        "elliptic_temporal_exposure",
        "edge_boundary_matrix",
        "time_transition_summary",
        "receptive_field_exposure",
        "symmetrization_provenance",
        "feature_provenance",
        "pending_audit04",
    ]:
        assert key in elliptic

    assert "split_rationale" in inventory
    assert "elliptic_future_information_leakage_observed" not in json.dumps(inventory)


def test_audit02_elliptic_edge_boundary_totals_reconcile() -> None:
    elliptic = load_inventory()["datasets"]["elliptic"]
    boundary = elliptic["edge_boundary_matrix"]

    assert boundary["stored_directed_edge_entries"] == 468710
    assert boundary["stored_directed_matrix_total"] == 468710
    assert boundary["canonical_unique_undirected_relations"] == 234355
    assert boundary["canonical_unique_relation_matrix_total"] == 234355
    assert boundary["boundary_counts_unique_relations"]["train_validation"] == 0
    assert boundary["boundary_counts_unique_relations"]["train_test"] == 0
    assert boundary["boundary_counts_unique_relations"]["validation_test"] == 0


def test_audit02_elliptic_time_transition_totals_reconcile() -> None:
    elliptic = load_inventory()["datasets"]["elliptic"]
    time_summary = elliptic["time_transition_summary"]

    assert time_summary["canonical_unique_relations_checked"] == 234355
    assert time_summary["same_time_relations"] == 234355
    assert time_summary["cross_time_relations"] == 0
    assert time_summary["stored_directed_entries"]["total"] == 468710
    assert time_summary["stored_directed_entries"]["same_time"] == 468710
    assert time_summary["absolute_time_gap"]["maximum"] == 0


def test_audit02_elliptic_temporal_exposure_schema_is_specific() -> None:
    elliptic = load_inventory()["datasets"]["elliptic"]
    exposure = elliptic["elliptic_temporal_exposure"]
    receptive = elliptic["receptive_field_exposure"]

    assert exposure["future_label_leakage_observed"] is False
    assert exposure["future_feature_exposure_observed"] is False
    assert exposure["future_topology_exposure_observed"] is False
    assert exposure["inductive_hidden_future_isolation_proven"] is False
    assert exposure["training_mode"] == "transductive_full_graph_temporal_supervision"

    assert receptive["train_nodes_checked"] == 26381
    for hop in ["1", "2", "3"]:
        assert receptive["cumulative_within_k_hop_counts"][hop]["fraction_any_future"] == 0


def test_audit02_unknown_labels_and_target_pool_boundary() -> None:
    elliptic = load_inventory()["datasets"]["elliptic"]

    assert elliptic["known_unknown"]["unknown_nodes_in_any_supervised_mask"] == 0
    assert elliptic["pending_audit04"]["exact_target_eligibility_filter_sampling_flow"] == (
        "PENDING_AUDIT04"
    )
    assert elliptic["pending_audit04"]["exact_pg_pool_eligibility_filter_sampling_flow"] == (
        "PENDING_AUDIT04"
    )


def test_audit02_symmetrization_raw_to_stored_counts() -> None:
    elliptic = load_inventory()["datasets"]["elliptic"]
    raw = elliptic["symmetrization_provenance"]["raw_edge_audit"]

    assert raw["raw_direction_comparison"] == "READ_LOCAL_RAW_CSV"
    assert raw["raw_directed_rows"] == 234355
    assert raw["raw_valid_non_self_loop_unique_directed_pairs"] == 234355
    assert raw["raw_reverse_directed_pairs_originally_present_count"] == 0
    assert raw["reverse_pairs_added_by_undirected_preprocessing"] == 234355
    assert raw["stored_directed_pairs"] == 468710
    assert raw["stored_equals_raw_plus_reverse_after_dedup"] is True
