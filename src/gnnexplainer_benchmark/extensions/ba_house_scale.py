from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask

SCIENTIFIC_TENSORS = (
    "x",
    "y",
    "edge_index",
    "node_mask",
    "edge_mask",
    "train_mask",
    "val_mask",
    "test_mask",
)
DETERMINISM_TENSORS = (*SCIENTIFIC_TENSORS, "motif_instance_id", "motif_role_in_house")


def frozen_generation_config(
    *, background_nodes: int, num_house_motifs: int, ba_m: int, feature_dim: int, seed: int
) -> dict[str, Any]:
    for name, value in {
        "background_nodes": background_nodes,
        "num_house_motifs": num_house_motifs,
        "ba_m": ba_m,
        "feature_dim": feature_dim,
        "seed": seed,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if not 0 < ba_m < background_nodes:
        raise ValueError("Require 0 < ba_m < background_nodes")
    if not 0 < num_house_motifs <= background_nodes or feature_dim <= 0:
        raise ValueError("Require positive feature_dim and 0 < motifs <= background_nodes")
    return {
        "dataset": {
            "type": "PyG ExplainerDataset",
            "background_graph": "Barabasi-Albert",
            "background_nodes": background_nodes,
            "ba_attachment_parameter": ba_m,
            "motif": "house",
            "num_motifs": num_house_motifs,
            "seed": seed,
            "constant_node_features": True,
            "feature_dim": feature_dim,
            "split": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
        }
    }


def generate_ba_house_scale(
    *, background_nodes: int, num_house_motifs: int, ba_m: int, feature_dim: int, seed: int
) -> Data:
    config = frozen_generation_config(
        background_nodes=background_nodes,
        num_house_motifs=num_house_motifs,
        ba_m=ba_m,
        feature_dim=feature_dim,
        seed=seed,
    )
    # Delegate the entire RNG/generator/split sequence to the frozen implementation.
    data = generate_ba_house_data(config)
    expected = background_nodes + 5 * num_house_motifs
    if data.num_nodes != expected:
        raise RuntimeError("Generator returned an unexpected number of house instances")
    data.motif_instance_id = torch.cat(
        (
            torch.zeros(background_nodes, dtype=torch.long),
            torch.arange(1, num_house_motifs + 1).repeat_interleave(5),
        )
    )
    data.motif_role_in_house = torch.cat(
        (
            torch.full((background_nodes,), -1, dtype=torch.long),
            torch.arange(5).repeat(num_house_motifs),
        )
    )
    return data


def distribution(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return dict.fromkeys(("count", "mean", "std", "min", "Q25", "median", "Q75", "max"), None)
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(array.min()),
        "Q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "Q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def structure(data: Data) -> dict[str, Any]:
    source, destination = data.edge_index
    instance = data.motif_instance_id
    internal = (instance[source] > 0) & (instance[source] == instance[destination])
    bridge = (instance[source] > 0) != (instance[destination] > 0)
    background = (instance[source] == 0) & (instance[destination] == 0)
    edges = list(map(tuple, data.edge_index.t().tolist()))
    edge_set = set(edges)
    degree = torch.bincount(source, minlength=data.num_nodes).tolist()
    splits = {"train": data.train_mask, "validation": data.val_mask, "test": data.test_mask}
    return {
        "num_nodes": int(data.num_nodes),
        "directed_entries": len(edges),
        "undirected_edges": len({tuple(sorted(edge)) for edge in edges}),
        "feature_dim": int(data.x.size(1)),
        "features_all_ones": bool((data.x == 1).all()),
        "average_degree": sum(degree) / len(degree),
        "median_degree": float(np.median(degree)),
        "min_degree": min(degree),
        "max_degree": max(degree),
        "self_loops": sum(u == v for u, v in edges),
        "duplicate_ordered_entries": len(edges) - len(edge_set),
        "reciprocal_complete": all((v, u) in edge_set for u, v in edges),
        "class_counts": dict(sorted(Counter(map(str, data.y.tolist())).items())),
        "motif_instances": int(instance.max()),
        "motif_nodes": int((instance > 0).sum()),
        "background_nodes": int((instance == 0).sum()),
        "motif_fraction": float((instance > 0).double().mean()),
        "background_fraction": float((instance == 0).double().mean()),
        "internal_house_directed_entries": int(internal.sum()),
        "internal_house_undirected_edges": int(internal.sum()) // 2,
        "bridge_directed_entries": int(bridge.sum()),
        "bridge_undirected_edges": int(bridge.sum()) // 2,
        "background_directed_entries": int(background.sum()),
        "background_undirected_edges": int(background.sum()) // 2,
        "other_directed_entries": int((~(internal | bridge | background)).sum()),
        "global_gt_positive_directed_entries": int(data.edge_mask.bool().sum()),
        "global_node_gt_matches_motifs": torch.equal(data.node_mask.bool(), instance > 0),
        "global_edge_gt_matches_internal": torch.equal(data.edge_mask.bool(), internal),
        "bridge_gt_positive_count": int((data.edge_mask.bool() & bridge).sum()),
        "background_gt_positive_count": int((data.edge_mask.bool() & background).sum()),
        "split_counts": {split: int(mask.sum()) for split, mask in splits.items()},
        "split_class_counts": {
            split: dict(sorted(Counter(map(str, data.y[mask].tolist())).items()))
            for split, mask in splits.items()
        },
        "split_disjoint_and_complete": bool(
            torch.all(sum(mask.int() for mask in splits.values()) == 1)
        ),
    }


def structural_audit(data: Data, variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    global_gt = data.edge_mask.bool()
    source, destination = data.edge_index
    instance = data.motif_instance_id
    mask_rows = {
        name: []
        for name in (
            "candidate_edge_mask",
            "target_global_gt_mask",
            "own_house_gt_mask",
            "non_own_house_gt_mask",
        )
    }
    node_ids = torch.where(instance > 0)[0].tolist()
    for node in node_ids:
        own_id = int(instance[node])
        candidate = computation_subgraph_edge_mask(data, node, num_hops=3)
        subset = k_hop_subgraph(node, 3, data.edge_index, num_nodes=data.num_nodes)[0]
        target_gt = candidate & global_gt
        own = target_gt & (instance[source] == own_id) & (instance[destination] == own_id)
        non_own = target_gt & ~own
        gt_instances = sorted(set(instance[source[target_gt]].tolist()))
        count = int(candidate.sum())
        global_count = int(target_gt.sum())
        own_count = int(own.sum())
        non_count = int(non_own.sum())
        clean = gt_instances == [own_id] and global_count == own_count == 12 and non_count == 0
        records.append(
            {
                "dataset_variant": variant,
                "node_id": node,
                "class": int(data.y[node]),
                "own_motif_instance": own_id,
                "candidate_directed_edge_count": count,
                "candidate_node_count": int(subset.numel()),
                "motif_instance_count_3hop": len(gt_instances),
                "global_gt_directed_count": global_count,
                "own_house_gt_directed_count": own_count,
                "non_own_house_gt_directed_count": non_count,
                "global_gt_prevalence": global_count / count,
                "structurally_clean": clean,
            }
        )
        for name, value in zip(mask_rows, (candidate, target_gt, own, non_own), strict=True):
            mask_rows[name].append(value)
    diagnostics = {
        "node_ids": torch.tensor(node_ids),
        **{name: torch.stack(values) for name, values in mask_rows.items()},
        "semantics": "Diagnostic only; primary GT is global_edge_mask restricted to candidate. "
        "Motif count means instances contributing positive GT edges, not vertex presence.",
    }
    return records, diagnostics


def structural_receipt(records: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [r for r in records if r["structurally_clean"]]
    full_fields = (
        "motif_instance_count_3hop",
        "global_gt_directed_count",
        "non_own_house_gt_directed_count",
    )
    clean_fields = ("candidate_directed_edge_count", "candidate_node_count", "global_gt_prevalence")
    return {
        "all_motif_count": len(records),
        "structurally_clean": len(clean),
        "clean_fraction": len(clean) / len(records),
        "all_motif_distributions": {
            key: distribution([r[key] for r in records]) for key in full_fields
        },
        "clean_distributions": {key: distribution([r[key] for r in clean]) for key in clean_fields},
        "motif_instance_bins": {
            "exactly_1": sum(r["motif_instance_count_3hop"] == 1 for r in records),
            "exactly_2": sum(r["motif_instance_count_3hop"] == 2 for r in records),
            "at_least_3": sum(r["motif_instance_count_3hop"] >= 3 for r in records),
        },
        "decomposition_valid": all(
            r["global_gt_directed_count"]
            == r["own_house_gt_directed_count"] + r["non_own_house_gt_directed_count"]
            for r in records
        ),
        "finite_complete": bool(records)
        and all(
            np.isfinite(float(value))
            for row in records
            for value in row.values()
            if isinstance(value, (int, float))
        ),
        "summary_convention": "Sample standard deviation ddof=1; linear interpolated quartiles.",
    }
