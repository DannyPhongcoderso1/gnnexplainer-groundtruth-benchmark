from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch

GROUPS = ["TRAIN_KNOWN", "VAL_KNOWN", "TEST_KNOWN", "UNKNOWN"]
UNKNOWN_PERIOD_GROUPS = [
    "UNKNOWN_TRAIN_PERIOD",
    "UNKNOWN_VAL_PERIOD",
    "UNKNOWN_TEST_PERIOD",
]
ORIGINAL_AUDIT_REPORT_SHA256 = (
    "47a3406ef1006c81be49f019167aa04adbd74a4c938e6c7797da5be6b7228b00"
)
ORIGINAL_AUDIT_INVENTORY_SHA256 = (
    "099e42b9f3d5a355da255a95a92f320325c594a0b12a476e0873552be5c9c876"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_data(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    return [(int(src), int(dst)) for src, dst in edge_index.t().tolist()]


def canonical_pairs(pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in pairs if pair[0] != pair[1]}


def matrix_template() -> dict[str, dict[str, int]]:
    return {row: {col: 0 for col in GROUPS} for row in GROUPS}


def matrix_total(matrix: dict[str, dict[str, int]]) -> int:
    return sum(sum(row.values()) for row in matrix.values())


def group_nodes(data: Any) -> tuple[list[str], list[str]]:
    labels = data.y.cpu()
    time_step = data.time_step.cpu()
    train = data.train_mask.cpu().bool()
    val = data.val_mask.cpu().bool()
    test = data.test_mask.cpu().bool()

    groups: list[str] = []
    period_groups: list[str] = []
    for idx in range(int(data.num_nodes)):
        time_value = int(time_step[idx].item())
        if bool(train[idx].item()):
            groups.append("TRAIN_KNOWN")
        elif bool(val[idx].item()):
            groups.append("VAL_KNOWN")
        elif bool(test[idx].item()):
            groups.append("TEST_KNOWN")
        else:
            if int(labels[idx].item()) != 2:
                raise AssertionError(f"Known node {idx} is outside all supervised masks")
            groups.append("UNKNOWN")

        if int(labels[idx].item()) == 2:
            if time_value < 30:
                period_groups.append("UNKNOWN_TRAIN_PERIOD")
            elif time_value < 40:
                period_groups.append("UNKNOWN_VAL_PERIOD")
            else:
                period_groups.append("UNKNOWN_TEST_PERIOD")
        else:
            period_groups.append(groups[-1])

    return groups, period_groups


def edge_boundary_audit(data: Any) -> dict[str, Any]:
    groups, period_groups = group_nodes(data)
    pairs = edge_pairs(data.edge_index)
    canonical = sorted(canonical_pairs(pairs))

    directed_matrix = matrix_template()
    for src, dst in pairs:
        directed_matrix[groups[src]][groups[dst]] += 1

    unique_upper = matrix_template()
    cross_counts = Counter()
    for src, dst in canonical:
        left = groups[src]
        right = groups[dst]
        left_idx = GROUPS.index(left)
        right_idx = GROUPS.index(right)
        row, col = (left, right) if left_idx <= right_idx else (right, left)
        unique_upper[row][col] += 1
        cross_counts["--".join((row, col))] += 1

    period_counts = Counter(period_groups)
    group_counts = Counter(groups)
    return {
        "node_group_counts": {key: int(group_counts[key]) for key in GROUPS},
        "unknown_period_group_counts": {
            key: int(period_counts[key]) for key in UNKNOWN_PERIOD_GROUPS
        },
        "stored_directed_edge_entries": len(pairs),
        "stored_directed_endpoint_matrix": directed_matrix,
        "stored_directed_matrix_total": matrix_total(directed_matrix),
        "canonicalization_rule": (
            "sort endpoint ids within each non-self-loop edge and de-duplicate"
        ),
        "canonical_unique_undirected_relations": len(canonical),
        "canonical_unique_relation_endpoint_matrix_upper_triangular": unique_upper,
        "canonical_unique_relation_matrix_total": matrix_total(unique_upper),
        "boundary_counts_unique_relations": {
            "within_train": int(cross_counts["TRAIN_KNOWN--TRAIN_KNOWN"]),
            "train_validation": int(cross_counts["TRAIN_KNOWN--VAL_KNOWN"]),
            "train_test": int(cross_counts["TRAIN_KNOWN--TEST_KNOWN"]),
            "train_unknown": int(cross_counts["TRAIN_KNOWN--UNKNOWN"]),
            "validation_test": int(cross_counts["VAL_KNOWN--TEST_KNOWN"]),
            "validation_unknown": int(cross_counts["VAL_KNOWN--UNKNOWN"]),
            "test_unknown": int(cross_counts["TEST_KNOWN--UNKNOWN"]),
        },
    }


def quantiles(values: list[int]) -> dict[str, float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return {}

    def pct(p: float) -> float:
        pos = (len(sorted_values) - 1) * p
        lower = int(pos)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = pos - lower
        return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)

    return {"q25": pct(0.25), "q50": pct(0.5), "q75": pct(0.75), "q90": pct(0.9)}


def time_transition_audit(data: Any) -> dict[str, Any]:
    pairs = edge_pairs(data.edge_index)
    canonical = sorted(canonical_pairs(pairs))
    time_step = data.time_step.cpu()

    directed_relation = Counter()
    for src, dst in pairs:
        src_time = int(time_step[src].item())
        dst_time = int(time_step[dst].item())
        if src_time < dst_time:
            directed_relation["earlier_to_later"] += 1
        elif src_time > dst_time:
            directed_relation["later_to_earlier"] += 1
        else:
            directed_relation["same_time"] += 1

    gaps = [abs(int(time_step[src].item()) - int(time_step[dst].item())) for src, dst in canonical]
    same_time = sum(1 for gap in gaps if gap == 0)
    cross_time = len(gaps) - same_time
    train_val_boundary = 0
    val_test_boundary = 0
    train_test_period = 0
    for src, dst in canonical:
        src_time = int(time_step[src].item())
        dst_time = int(time_step[dst].item())
        lo, hi = sorted((src_time, dst_time))
        if lo <= 29 and hi >= 30:
            train_val_boundary += 1
        if lo <= 39 and hi >= 40:
            val_test_boundary += 1
        if lo <= 29 and hi >= 40:
            train_test_period += 1

    return {
        "canonical_unique_relations_checked": len(canonical),
        "same_time_relations": int(same_time),
        "cross_time_relations": int(cross_time),
        "absolute_time_gap": {
            "minimum": int(min(gaps)),
            "maximum": int(max(gaps)),
            "mean": float(statistics.fmean(gaps)),
            "median": float(statistics.median(gaps)),
            "quantiles": quantiles(gaps),
        },
        "relations_crossing_train_validation_boundary_29_to_30_plus": train_val_boundary,
        "relations_crossing_validation_test_boundary_39_to_40_plus": val_test_boundary,
        "relations_connecting_train_period_to_test_period": train_test_period,
        "stored_directed_entries": {
            "earlier_to_later": int(directed_relation["earlier_to_later"]),
            "later_to_earlier": int(directed_relation["later_to_earlier"]),
            "same_time": int(directed_relation["same_time"]),
            "total": len(pairs),
            "direction_interpretation": (
                "Stored graph is symmetrized, so earlier_to_later and later_to_earlier "
                "entries are stored message-passing directions, not necessarily raw "
                "transaction directions."
            ),
        },
    }


def build_adjacency(num_nodes: int, canonical: set[tuple[int, int]]) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in canonical:
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    return adjacency


def receptive_field_audit(data: Any, max_hops: int = 3) -> dict[str, Any]:
    groups, period_groups = group_nodes(data)
    train_nodes = torch.nonzero(data.train_mask.cpu().bool(), as_tuple=False).view(-1).tolist()
    canonical = canonical_pairs(edge_pairs(data.edge_index))
    adjacency = build_adjacency(int(data.num_nodes), canonical)

    future_groups = {"VAL_KNOWN", "TEST_KNOWN", "UNKNOWN_VAL_PERIOD", "UNKNOWN_TEST_PERIOD"}
    exact = {str(hop): Counter() for hop in range(1, max_hops + 1)}
    cumulative = {str(hop): Counter() for hop in range(1, max_hops + 1)}

    for node in train_nodes:
        visited = {int(node)}
        frontier = {int(node)}
        cumulative_reached: set[int] = set()
        for hop in range(1, max_hops + 1):
            next_frontier: set[int] = set()
            for current in frontier:
                next_frontier.update(adjacency[current])
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
            cumulative_reached.update(frontier)

            exact_future = {period_groups[target] for target in frontier} & future_groups
            cumulative_future = (
                {period_groups[target] for target in cumulative_reached} & future_groups
            )
            if exact_future:
                exact[str(hop)]["any_future"] += 1
                for group in exact_future:
                    exact[str(hop)][group] += 1
            if cumulative_future:
                cumulative[str(hop)]["any_future"] += 1
                for group in cumulative_future:
                    cumulative[str(hop)][group] += 1

    def pack(counter_by_hop: dict[str, Counter]) -> dict[str, dict[str, Any]]:
        result = {}
        for hop, counts in counter_by_hop.items():
            row = {key: int(value) for key, value in sorted(counts.items())}
            row["fraction_any_future"] = (
                float(row.get("any_future", 0) / len(train_nodes)) if train_nodes else 0.0
            )
            result[hop] = row
        return result

    return {
        "train_nodes_checked": len(train_nodes),
        "max_hops_checked": max_hops,
        "definition": (
            "exact hop frontier and cumulative reachable-within-k-hop over canonical "
            "undirected relations"
        ),
        "future_groups_counted": sorted(future_groups),
        "exact_shortest_frontier_counts": pack(exact),
        "cumulative_within_k_hop_counts": pack(cumulative),
    }


def raw_edge_audit(raw_dir: Path, data: Any) -> dict[str, Any]:
    edge_file = raw_dir / "elliptic_txs_edgelist.csv"
    feature_file = raw_dir / "elliptic_txs_features.csv"
    if not edge_file.exists() or not feature_file.exists():
        return {"raw_direction_comparison": "UNKNOWN", "missing_files": True}

    tx_ids: list[int] = []
    with feature_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            tx_ids.append(int(row[0]))
    id_map = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    raw_pairs: list[tuple[int, int]] = []
    with edge_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            src = id_map.get(int(row["txId1"]))
            dst = id_map.get(int(row["txId2"]))
            if src is not None and dst is not None and src != dst:
                raw_pairs.append((src, dst))

    raw_pair_set = set(raw_pairs)
    raw_reverse_originally_present = sum(
        1 for src, dst in raw_pair_set if (dst, src) in raw_pair_set
    )
    stored_pairs = set(edge_pairs(data.edge_index))
    canonical = canonical_pairs(list(raw_pair_set))
    added_reverse_count = sum(1 for src, dst in raw_pair_set if (dst, src) not in raw_pair_set)
    return {
        "raw_direction_comparison": "READ_LOCAL_RAW_CSV",
        "raw_directed_rows": len(raw_pairs),
        "raw_valid_non_self_loop_unique_directed_pairs": len(raw_pair_set),
        "raw_duplicate_rows_removed_by_set": len(raw_pairs) - len(raw_pair_set),
        "raw_reverse_directed_pairs_originally_present_count": int(raw_reverse_originally_present),
        "reverse_pairs_added_by_undirected_preprocessing": int(added_reverse_count),
        "stored_directed_pairs": len(stored_pairs),
        "stored_canonical_unique_relations": len(canonical),
        "stored_equals_raw_plus_reverse_after_dedup": stored_pairs
        == (raw_pair_set | {(dst, src) for src, dst in raw_pair_set}),
        "code_path": (
            "D:/CS/KLTN/elliptic-gnn-xai/data/dataset.py "
            "BCDataset._load_elliptic(directed=False)"
        ),
        "symmetrization_code_symbol": "edges_rev plus pandas.concat followed by drop_duplicates",
    }


def update_inventory(
    existing_path: Path,
    output_path: Path,
    elliptic_data: Any,
    raw_dir: Path,
) -> dict[str, Any]:
    inventory = json.loads(existing_path.read_text(encoding="utf-8"))
    elliptic = inventory["datasets"]["elliptic"]

    edge_boundary = edge_boundary_audit(elliptic_data)
    if edge_boundary["stored_directed_matrix_total"] != 468710:
        raise AssertionError("Directed edge-boundary matrix total does not reconcile")
    if edge_boundary["canonical_unique_relation_matrix_total"] != 234355:
        raise AssertionError("Unique relation boundary matrix total does not reconcile")

    time_transition = time_transition_audit(elliptic_data)
    if time_transition["stored_directed_entries"]["total"] != 468710:
        raise AssertionError("Stored directed time-transition total does not reconcile")
    if time_transition["canonical_unique_relations_checked"] != 234355:
        raise AssertionError("Canonical time-transition total does not reconcile")

    receptive = receptive_field_audit(elliptic_data, max_hops=3)
    raw_edges = raw_edge_audit(raw_dir, elliptic_data)

    future_exposure_observed = any(
        receptive["cumulative_within_k_hop_counts"][str(hop)].get("any_future", 0) > 0
        for hop in range(1, 4)
    )
    elliptic["elliptic_temporal_exposure"] = {
        "future_label_leakage_observed": False,
        "future_feature_exposure_observed": future_exposure_observed,
        "future_topology_exposure_observed": future_exposure_observed,
        "inductive_hidden_future_isolation_proven": False,
        "training_mode": "transductive_full_graph_temporal_supervision",
        "note": (
            "The full graph object is passed to the model, but the frozen Elliptic graph has "
            "zero cross-time relations, so train-period nodes have no observed 1-3 hop "
            "future-period receptive-field exposure."
        ),
        "definitions": {
            "future_label_leakage": (
                "validation/test/future labels used in train loss, target construction, "
                "or model selection"
            ),
            "future_feature_exposure": (
                "train-node representation can receive feature information from nodes "
                "in future validation/test periods"
            ),
            "future_topology_exposure": (
                "train-node representation can depend on connectivity involving future nodes"
            ),
            "inductive_hidden_future_isolation": (
                "future nodes and edges absent from the training computation graph"
            ),
        },
    }
    elliptic["edge_boundary_matrix"] = edge_boundary
    elliptic["time_transition_summary"] = time_transition
    elliptic["receptive_field_exposure"] = receptive
    elliptic["symmetrization_provenance"] = {
        "chronology": "INHERITED_BASELINE_IMPLEMENTATION",
        "raw_relationship_semantics": (
            "directed Bitcoin payment-flow edges per upstream Elliptic source"
        ),
        "repository_symmetrizes": True,
        "raw_edge_audit": raw_edges,
        "rationale": {
            "source": "INFERENCE",
            "text": (
                "Symmetrization supports undirected neighborhood aggregation in GCN and "
                "GraphSAGE over the transaction graph."
            ),
        },
        "benefit": (
            "Both incoming and outgoing transaction neighbors can contribute to "
            "message passing."
        ),
        "trade_off": (
            "Raw transaction direction information is reduced in the stored "
            "message-passing graph."
        ),
        "later_to_earlier_information_flow_possible": True,
    }
    elliptic["feature_provenance"] = {
        "repository_confirmed": {
            "time_step_stored_separately": True,
            "time_step_dropped_from_model_input": True,
            "repository_scaler_or_imputer_fit": False,
        },
        "source_confirmed": {
            "initial_feature_count_including_time_step": 166,
            "local_transaction_features": 94,
            "aggregated_one_hop_features": 72,
            "features_are_standardized_except_id_class_time": (
                "SOURCE_CONFIRMED_BY_KAGGLE_DESCRIPTION"
            ),
        },
        "unknown": {
            "external_raw_feature_scaling_fit_scope": "UNKNOWN",
            "whether_aggregate_features_cross_time_boundaries": "UNKNOWN_FROM_PRIMARY_SOURCE",
        },
        "interpretation": (
            "The project uses the benchmark-provided feature table as frozen input and "
            "does not fit an additional project-level scaler or imputer."
        ),
    }
    elliptic["backbone_training_graph_trace"] = {
        "gcn": {
            "model_layers": 2,
            "edge_index_passed": "full graph edge_index",
            "nodes_passed_to_forward": 203769,
            "edges_passed_to_forward": 468710,
            "loss_mask": (
                "train_mask during Optuna trial, train_mask OR val_mask for final checkpoint"
            ),
            "validation_model_selection_access": (
                "validation labels used only for objective selection"
            ),
            "test_access_during_training": (
                "test metric logged/evaluated but objective_source asserted as validation"
            ),
            "cached_setting": (
                "baseline GCNConv(cached=True); XAI-compatible wrapper uses cached=False"
            ),
            "self_loop_behavior": "PyG GCNConv default add_self_loops behavior unless overridden",
            "train_forward_includes_validation_test_unknown_features": True,
            "train_forward_includes_boundary_edges": True,
        },
        "graphsage": {
            "model_layers": 2,
            "edge_index_passed": "full graph edge_index",
            "nodes_passed_to_forward": 203769,
            "edges_passed_to_forward": 468710,
            "loss_mask": (
                "train_mask during Optuna trial, train_mask OR val_mask for final checkpoint"
            ),
            "validation_model_selection_access": (
                "validation labels used only for objective selection"
            ),
            "test_access_during_training": (
                "test metric logged/evaluated but objective_source asserted as validation"
            ),
            "cached_setting": "not applicable to SAGEConv",
            "self_loop_behavior": (
                "SAGEConv aggregates neighbor messages and root node transform by layer design"
            ),
            "train_forward_includes_validation_test_unknown_features": True,
            "train_forward_includes_boundary_edges": True,
        },
    }
    inventory["split_rationale"] = {
        "ba_house": {
            "split_design": "stratified 60/20/20 node split",
            "actual_source": "project-defined local split in configs/ba_house/ba500/ba_house.yaml",
            "rationale": {
                "source": "INFERENCE",
                "text": (
                    "A deterministic stratified split preserves class proportions in "
                    "a controlled synthetic benchmark."
                ),
            },
            "chronology": "ORIGINALLY_PRE_SPECIFIED",
            "benefit": "Reproducible class-balanced train/validation/test masks.",
            "trade_off": (
                "Random transductive split is not a temporal or deployment-style evaluation."
            ),
        },
        "shapeggen": {
            "split_design": "stratified 60/20/20 node split after official ShapeGGen generation",
            "actual_source": "project-defined local adapter in configs/shapeggen/v1/shapeggen.yaml",
            "rationale": {
                "source": "INFERENCE",
                "text": (
                    "The synthetic graph and labels are generated first, then a deterministic "
                    "stratified split is attached for local backbone training."
                ),
            },
            "chronology": "ORIGINALLY_PRE_SPECIFIED",
            "benefit": (
                "Keeps generator provenance separate from local train/validation/test masks."
            ),
            "trade_off": (
                "Target-pool eligibility counts are later-stage filters, not raw split counts."
            ),
        },
        "elliptic": {
            "split_design": "temporal known-label split: <30, 30-39, >=40",
            "actual_source": "inherited baseline implementation in data/dataset.py",
            "rationale": {
                "source": "INFERENCE_WITH_EXTERNAL_SOURCE_SUPPORT",
                "text": (
                    "Temporal ordering is appropriate for transaction graphs because time steps "
                    "encode chronological blockchain activity and concept drift is possible."
                ),
            },
            "chronology": "INHERITED_BASELINE_IMPLEMENTATION",
            "benefit": "Avoids random mixing of known labels across chronological periods.",
            "trade_off": (
                "This is temporal supervision over a full graph object, not a strict hidden-future "
                "inductive protocol."
            ),
        },
    }
    elliptic["pending_audit04"] = {
        "exact_target_eligibility_filter_sampling_flow": "PENDING_AUDIT04",
        "exact_pg_pool_eligibility_filter_sampling_flow": "PENDING_AUDIT04",
        "audit02_verified_only": [
            "file existence",
            "hash",
            "row count",
            "node uniqueness",
            "split membership",
            "pairwise overlap",
        ],
    }

    inventory["amendment_history"] = {
        "original_audit02_report_sha256_before_expansion": ORIGINAL_AUDIT_REPORT_SHA256,
        "original_audit02_inventory_sha256_before_expansion": ORIGINAL_AUDIT_INVENTORY_SHA256,
        "original_audit02_verdict": (
            "PASS WITH DOCUMENTED UNKNOWN ELLIPTIC FEATURE-PREPROCESSING PROVENANCE"
        ),
        "external_review_verdict": "BLOCKED FOR DETERMINISTIC EVIDENCE EXPANSION",
        "expansion": (
            "Added deterministic Elliptic edge-boundary, time-transition, "
            "receptive-field, symmetrization, feature provenance, and target-pool "
            "pending-audit evidence."
        ),
    }
    inventory["stop_condition_scan"] = {
        "feature_leakage_observed": "NOT_OBSERVED_IN_REPOSITORY_PREPROCESSING_CODE",
        "untraceable_split_creation": False,
        "unexpected_overlap": False,
        "inconsistent_dataset_hashes_observed": False,
        "elliptic_future_label_leakage_observed": False,
        "elliptic_future_feature_exposure_observed": future_exposure_observed,
        "elliptic_future_topology_exposure_observed": future_exposure_observed,
        "unknowns": [
            (
                "Elliptic external raw feature scaling/fitting provenance before the "
                "repository reads CSV files is UNKNOWN."
            ),
            "Exact target/pool eligibility construction is PENDING_AUDIT04.",
        ],
    }

    output_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand Audit02 deterministic evidence.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("docs/audit/dataset_split_preprocessing_inventory.json"),
    )
    parser.add_argument(
        "--elliptic-data",
        type=Path,
        default=Path("D:/CS/KLTN/elliptic-gnn-xai/datasets/elliptic/elliptic_data.pt"),
    )
    parser.add_argument(
        "--elliptic-raw-dir",
        type=Path,
        default=Path("D:/CS/KLTN/elliptic-gnn-xai/datasets/elliptic"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_torch_data(args.elliptic_data)
    update_inventory(args.inventory, args.inventory, data, args.elliptic_raw_dir)


if __name__ == "__main__":
    main()
