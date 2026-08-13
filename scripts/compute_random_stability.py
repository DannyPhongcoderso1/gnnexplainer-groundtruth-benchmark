from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch
import yaml

from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.evaluation.stability import (
    expected_pair_count,
    pairwise_jaccard_rows,
)
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.training.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Phase 4D Random Jaccard stability.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/data/ba_house.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house_gcn_best.pt"),
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=Path("results/metrics/explainer_test_targets.csv"),
    )
    parser.add_argument(
        "--phase4d-config",
        type=Path,
        default=Path("configs/evaluation/phase4d.yaml"),
    )
    parser.add_argument(
        "--random-metrics",
        type=Path,
        default=Path("results/metrics/phase4d_random_baseline/random_baseline_per_node.csv"),
    )
    parser.add_argument(
        "--random-explanation-root",
        type=Path,
        default=Path("results/explanations/phase4d_random_baseline"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metrics/phase4d_random_baseline"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/metrics/phase4d_comparison.csv"),
    )
    parser.add_argument(
        "--gnn-stability-summary",
        type=Path,
        default=Path("results/metrics/phase4d_stability/gnnexplainer_stability_summary.csv"),
    )
    return parser.parse_args()


def load_phase4d_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "phase4d" not in config:
        raise ValueError("Phase 4D config must contain a top-level phase4d mapping")
    return config["phase4d"]


def read_target_ids(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [int(row["node_id"]) for row in csv.DictReader(handle)]


def read_random_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_selected_edge_set(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        edge_indices = [int(row["global_edge_index"]) for row in csv.DictReader(handle)]
    edge_set = set(edge_indices)
    if len(edge_set) != len(edge_indices):
        raise ValueError(f"Duplicate selected edges found in {path}")
    return edge_set


def summarize_values(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    lookup = {}
    for row in rows:
        if row["status"] != "success":
            continue
        key = (int(row["node_id"]), int(row["seed"]))
        if key in lookup:
            raise ValueError(f"Duplicate Random metric row for node/seed {key}")
        lookup[key] = row
    return lookup


def update_comparison_stability(path: Path, random_stability: float) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError("Comparison table has no header")
    for row in rows:
        if row["method"] == "Random":
            row["stability"] = str(random_stability)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_gnn_stability(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "mean_pairwise_jaccard":
                return float(row["mean"])
    return None


def main() -> None:
    args = parse_args()
    phase4d = load_phase4d_config(args.phase4d_config)
    seeds = [int(seed) for seed in phase4d["random_baseline_seeds"]]
    selected_edge_ratio = float(phase4d["selected_edge_ratio"])
    target_ids = read_target_ids(args.target_file)
    random_rows = read_random_metric_rows(args.random_metrics)
    random_lookup = metric_lookup(random_rows)
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    checkpoint = load_checkpoint(args.checkpoint)
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])

    pairwise_rows: list[dict[str, Any]] = []
    per_node_rows: list[dict[str, Any]] = []
    candidate_fingerprints: dict[int, tuple[int, ...]] = {}
    all_seed_sets_differ_count = 0

    for node_id in target_ids:
        candidate_edge_mask = computation_subgraph_edge_mask(data, node_id, num_hops)
        candidate_indices = tuple(
            int(edge_idx)
            for edge_idx in torch.nonzero(candidate_edge_mask, as_tuple=False).view(-1).tolist()
        )
        candidate_fingerprints[node_id] = candidate_indices
        candidate_edge_count = len(candidate_indices)
        expected_selected_count = max(1, ceil(candidate_edge_count * selected_edge_ratio))
        seed_to_edges: dict[int, set[int]] = {}
        status = "success"
        errors = []
        for seed in seeds:
            metric_row = random_lookup.get((node_id, seed))
            if metric_row is None:
                status = "failed"
                errors.append(f"missing metric row for seed {seed}")
                continue
            selected_edge_count = int(metric_row["selected_edge_count"])
            if selected_edge_count != expected_selected_count:
                status = "failed"
                errors.append(
                    f"seed {seed} selected_edge_count={selected_edge_count}, "
                    f"expected={expected_selected_count}"
                )
            selected_path = (
                args.random_explanation_root
                / f"seed_{seed}"
                / f"node_{node_id}"
                / "selected_edges.csv"
            )
            if not selected_path.exists():
                status = "failed"
                errors.append(f"missing selected_edges for seed {seed}")
                continue
            selected_edges = read_selected_edge_set(selected_path)
            if len(selected_edges) != expected_selected_count:
                status = "failed"
                errors.append(
                    f"seed {seed} selected set size={len(selected_edges)}, "
                    f"expected={expected_selected_count}"
                )
            if not selected_edges.issubset(set(candidate_indices)):
                status = "failed"
                errors.append(f"seed {seed} selected edges outside candidate set")
            seed_to_edges[seed] = selected_edges

        node_pairwise = pairwise_jaccard_rows(
            node_id=node_id,
            seed_to_edges=seed_to_edges,
            selected_edge_count=expected_selected_count,
        )
        if len(node_pairwise) != expected_pair_count(len(seeds)):
            status = "failed"
            errors.append(
                f"pair_count={len(node_pairwise)}, expected={expected_pair_count(len(seeds))}"
            )
        pairwise_rows.extend(node_pairwise)
        values = [float(row["jaccard"]) for row in node_pairwise]
        unique_sets = {tuple(sorted(edges)) for edges in seed_to_edges.values()}
        if len(unique_sets) > 1:
            all_seed_sets_differ_count += 1
        if values:
            summary = summarize_values(values)
            per_node_rows.append(
                {
                    "node_id": node_id,
                    "candidate_edge_count": candidate_edge_count,
                    "selected_edge_count": expected_selected_count,
                    "num_seeds": len(seed_to_edges),
                    "pair_count": len(values),
                    "mean_jaccard": summary["mean"],
                    "std_jaccard": summary["std"],
                    "median_jaccard": summary["median"],
                    "min_jaccard": summary["min"],
                    "max_jaccard": summary["max"],
                    "status": status,
                    "error": " | ".join(errors),
                }
            )

    success_rows = [row for row in per_node_rows if row["status"] == "success"]
    overall_values = [float(row["mean_jaccard"]) for row in success_rows]
    overall_summary = summarize_values(overall_values)
    summary_rows = [
        {
            "metric": "random_stability_mean_jaccard",
            "n_targets": len(overall_values),
            "mean": overall_summary["mean"],
            "std": overall_summary["std"],
            "median": overall_summary["median"],
            "min": overall_summary["min"],
            "max": overall_summary["max"],
        }
    ]

    pairwise_path = args.output_dir / "random_stability_pairwise.csv"
    per_node_path = args.output_dir / "random_stability_per_node.csv"
    summary_path = args.output_dir / "random_stability_summary.csv"
    write_rows(
        pairwise_path,
        pairwise_rows,
        [
            "node_id",
            "seed_a",
            "seed_b",
            "selected_edge_count",
            "intersection_count",
            "union_count",
            "jaccard",
        ],
    )
    write_rows(
        per_node_path,
        per_node_rows,
        [
            "node_id",
            "candidate_edge_count",
            "selected_edge_count",
            "num_seeds",
            "pair_count",
            "mean_jaccard",
            "std_jaccard",
            "median_jaccard",
            "min_jaccard",
            "max_jaccard",
            "status",
            "error",
        ],
    )
    write_rows(
        summary_path,
        summary_rows,
        ["metric", "n_targets", "mean", "std", "median", "min", "max"],
    )
    update_comparison_stability(args.comparison, float(overall_summary["mean"]))
    manifest = {
        "target_file": str(args.target_file),
        "target_count": len(target_ids),
        "success_count": len(success_rows),
        "fail_count": len(per_node_rows) - len(success_rows),
        "random_seeds": seeds,
        "pairwise_comparisons_per_target": expected_pair_count(len(seeds)),
        "total_pairwise_comparisons": len(pairwise_rows),
        "selected_edge_ratio": selected_edge_ratio,
        "rounding_rule": phase4d["rounding_rule"],
        "candidate_edge_fingerprint_count": len(candidate_fingerprints),
        "nodes_with_nonidentical_seed_sets": all_seed_sets_differ_count,
        "random_artifacts_reused": True,
        "random_metrics_recomputed": False,
        "gnnexplainer_stability": read_gnn_stability(args.gnn_stability_summary),
        "random_stability_summary": str(summary_path),
        "random_stability_per_node": str(per_node_path),
        "random_stability_pairwise": str(pairwise_path),
        "comparison": str(args.comparison),
    }
    (args.output_dir / "random_stability_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Random stability complete | "
        f"targets={len(target_ids)} | success={len(success_rows)} | "
        f"failed={len(per_node_rows) - len(success_rows)} | "
        f"pairs={len(pairwise_rows)} | mean={overall_summary['mean']}"
    )
    print(f"pairwise={pairwise_path}")
    print(f"per_node={per_node_path}")
    print(f"summary={summary_path}")
    print(f"comparison={args.comparison}")


if __name__ == "__main__":
    main()
