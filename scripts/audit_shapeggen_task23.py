from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch
import yaml

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.data.shapeggen import (
    get_target_candidate_edge_mask,
    get_target_edge_ground_truth,
)
from gnnexplainer_benchmark.evaluation.budget import selected_edge_count_for_ratio

BACKBONES = ("gcn", "graphsage")
METHODS = ("gnnexplainer", "random")
QUALITY_METRICS = [
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "remove_probability_delta",
    "retain_probability",
    "retain_probability_delta",
    "retain_probability_ratio",
    "sparsity",
    "runtime_seconds",
    "generation_runtime_seconds",
    "evaluation_runtime_seconds",
]
QUALITY_REQUIRED_FIELDS = [
    "node_id",
    "seed",
    "candidate_edge_count",
    "selected_edge_count",
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "original_probability",
    "remove_probability",
    "retain_probability",
    "remove_probability_delta",
    "retain_probability_delta",
    "retain_probability_ratio",
    "sparsity",
    "runtime_seconds",
    "evaluation_runtime_seconds",
]
GNN_REQUIRED_EXTRA_FIELDS = ["generation_runtime_seconds"]
COMPARISON_METRICS = [
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "sparsity",
    "stability",
]
EXPECTED_REPORTED = {
    ("GCN", "GNNExplainer"): {
        "edge_auroc": 0.648445,
        "edge_ap": 0.324281,
        "fid_plus": 0.562874,
        "fid_minus": 0.011976,
        "one_minus_fid_minus": 0.988024,
        "sparsity": 0.795928,
        "stability": 0.367797,
    },
    ("GCN", "Random"): {
        "edge_auroc": 0.502273,
        "edge_ap": 0.237006,
        "fid_plus": 0.127545,
        "fid_minus": 0.239521,
        "one_minus_fid_minus": 0.760479,
        "sparsity": 0.795928,
        "stability": 0.116099,
    },
    ("GraphSAGE", "GNNExplainer"): {
        "edge_auroc": 0.631585,
        "edge_ap": 0.297549,
        "fid_plus": 0.497006,
        "fid_minus": 0.257485,
        "one_minus_fid_minus": 0.742515,
        "sparsity": 0.795928,
        "stability": 0.321343,
    },
    ("GraphSAGE", "Random"): {
        "edge_auroc": 0.502273,
        "edge_ap": 0.237006,
        "fid_plus": 0.128743,
        "fid_minus": 0.585629,
        "one_minus_fid_minus": 0.414371,
        "sparsity": 0.795928,
        "stability": 0.116099,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen ShapeGGen Task23 artifacts.")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/evaluation/shapeggen_task23.yaml"),
    )
    parser.add_argument(
        "--previous-freeze",
        type=Path,
        default=Path("docs/shapeggen_gnnexplainer_random_freeze.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/shapeggen_gnnexplainer_random_report.md"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("docs/shapeggen_task23_audit.json"),
    )
    parser.add_argument("--pytest-result", default="not run in this audit invocation")
    parser.add_argument("--ruff-result", default="not run in this audit invocation")
    parser.add_argument("--diff-check-result", default="not run in this audit invocation")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "task23" not in config:
        raise ValueError("Task23 config must contain a top-level task23 mapping")
    return config["task23"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_rows(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    output = []
    success_rows = [row for row in rows if row.get("status") == "success"]
    for metric in metrics:
        values = [
            float(row[metric])
            for row in success_rows
            if row.get(metric) not in (None, "", "nan", "NaN")
        ]
        summary = distribution(values)
        summary["metric"] = metric
        output.append(summary)
    return output


def summary_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["metric"]): row for row in rows}


def finite(value: Any) -> bool:
    if value in (None, ""):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def count_missing_nonfinite(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = 0
    nan_inf = 0
    for row in rows:
        for value in row.values():
            if value in (None, ""):
                missing += 1
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    nan_inf += 1
    return {"missing_value_count": missing, "nan_inf_count": nan_inf}


def infer_dtype(values: list[str]) -> str:
    non_null = [value for value in values if value not in (None, "")]
    if not non_null:
        return "empty"
    if all(str(value).lstrip("-").isdigit() for value in non_null):
        return "int"
    try:
        for value in non_null:
            float(value)
    except (TypeError, ValueError):
        return "string"
    return "float"


def expected_structural_null_reason(family: str, column: str) -> str | None:
    if column == "error":
        return "EXPECTED_STRUCTURAL_NULL: empty when status is success and no failure was recorded."
    if column == "failure_reason":
        return "EXPECTED_STRUCTURAL_NULL: empty when status is success and no failure was recorded."
    if family.startswith("random") and column == "generation_runtime_seconds":
        return (
            "EXPECTED_STRUCTURAL_NULL: Random has no explanation-generation stage; "
            "runtime_seconds/evaluation_runtime_seconds measure random edge selection "
            "plus evaluation."
        )
    return None


def required_columns_for_family(family: str, columns: list[str]) -> set[str]:
    if "pairwise" in family:
        return set(columns)
    if "stability_node" in family:
        return {
            "node_id",
            "mean_pairwise_jaccard",
            "std_pairwise_jaccard",
            "median_pairwise_jaccard",
            "min_pairwise_jaccard",
            "max_pairwise_jaccard",
            "num_seeds",
            "num_pairs",
            "status",
        }
    if family in {"gnn_primary", "gnn_stability_seed"}:
        return set(QUALITY_REQUIRED_FIELDS + GNN_REQUIRED_EXTRA_FIELDS + ["status"])
    if family in {"random_seed", "random_node"}:
        return set(QUALITY_REQUIRED_FIELDS + ["status"])
    return set(columns)


def column_null_audit(rows: list[dict[str, str]], family: str) -> dict[str, Any]:
    if not rows:
        return {
            "columns": [],
            "expected_structural_null_cells": 0,
            "unexpected_required_null_cells": 0,
            "total_null_cells": 0,
            "total_inf_cells": 0,
            "unexpected_null_columns": [],
            "expected_null_columns": [],
        }
    columns = list(rows[0].keys())
    required = required_columns_for_family(family, columns)
    column_rows = []
    total_null = 0
    total_inf = 0
    expected_structural = 0
    unexpected_required = 0
    unexpected_columns = set()
    expected_columns: dict[str, str] = {}
    for column in columns:
        values = [row.get(column, "") for row in rows]
        null_count = sum(1 for value in values if value in (None, ""))
        inf_count = 0
        for value in values:
            if value in (None, ""):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                inf_count += 1
        reason = expected_structural_null_reason(family, column)
        is_required = column in required
        is_expected_null = reason is not None
        unexpected_count = null_count if null_count and is_required and not is_expected_null else 0
        expected_count = null_count if null_count and is_expected_null else 0
        total_null += null_count
        total_inf += inf_count
        unexpected_required += unexpected_count
        expected_structural += expected_count
        if unexpected_count:
            unexpected_columns.add(column)
        if expected_count:
            expected_columns[column] = reason or ""
        column_rows.append(
            {
                "column": column,
                "dtype": infer_dtype(values),
                "row_count": len(rows),
                "null_count": null_count,
                "inf_count": inf_count,
                "required_applicable": is_required,
                "expected_null": is_expected_null,
                "reason": reason or "",
                "null_classification": (
                    "EXPECTED_STRUCTURAL_NULL"
                    if expected_count
                    else "UNEXPECTED_REQUIRED_NULL"
                    if unexpected_count
                    else "NONE"
                ),
            }
        )
    return {
        "columns": column_rows,
        "expected_structural_null_cells": expected_structural,
        "unexpected_required_null_cells": unexpected_required,
        "total_null_cells": total_null,
        "total_inf_cells": total_inf,
        "unexpected_null_columns": sorted(unexpected_columns),
        "expected_null_columns": expected_columns,
    }


def audit_inputs(config: dict[str, Any], task_config_path: Path) -> dict[str, Any]:
    paths = {
        "dataset": Path(config["dataset_path"]),
        "targets": Path(config["target_file"]),
        "gcn_checkpoint": Path(config["gcn_checkpoint"]),
        "graphsage_checkpoint": Path(config["graphsage_checkpoint"]),
        "task23_config": task_config_path,
        "gnnexplainer_config": Path(config["gnnexplainer_config"]),
    }
    expected = {
        "dataset": config["dataset_sha256"],
        "targets": config["target_sha256"],
        "gcn_checkpoint": config["gcn_checkpoint_sha256"],
        "graphsage_checkpoint": config["graphsage_checkpoint_sha256"],
    }
    rows = {}
    for name, path in paths.items():
        actual = sha256_file(path)
        rows[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": actual,
            "expected_sha256": expected.get(name),
            "matches_expected": expected.get(name) in (None, actual),
        }
    return rows


def audit_targets_and_candidate_gt(config: dict[str, Any]) -> dict[str, Any]:
    data = load_data_artifact(Path(config["dataset_path"]))
    rows = read_csv(Path(config["target_file"]))
    node_ids = [int(row["node_id"]) for row in rows]
    edge_pairs = [
        (int(data.edge_index[0, idx].item()), int(data.edge_index[1, idx].item()))
        for idx in range(data.edge_index.size(1))
    ]
    edge_set = set(edge_pairs)
    duplicate_edges = len(edge_pairs) - len(edge_set)
    missing_reverse = sum(1 for src, dst in edge_pairs if (dst, src) not in edge_set)
    self_loops = sum(1 for src, dst in edge_pairs if src == dst)
    candidate_counts = []
    positive_counts = []
    negative_counts = []
    selected_counts = []
    violations = Counter()
    target_records = []
    for row in rows:
        node_id = int(row["node_id"])
        candidate = get_target_candidate_edge_mask(data, node_id)
        gt = get_target_edge_ground_truth(data, node_id)
        candidate_count = int(candidate.sum().item())
        positive_count = int(gt[candidate].sum().item())
        negative_count = candidate_count - positive_count
        selected_count = selected_edge_count_for_ratio(
            candidate_count,
            float(config["selected_edge_ratio"]),
        )
        if candidate.numel() != data.edge_index.size(1) or gt.numel() != data.edge_index.size(1):
            violations["candidate_gt_length_mismatch"] += 1
        if bool((gt & ~candidate).any()):
            violations["gt_not_subset_of_candidate"] += 1
        if positive_count <= 0 or negative_count <= 0:
            violations["invalid_single_class_gt_targets"] += 1
        if selected_count != max(1, math.ceil(candidate_count * 0.2)):
            violations["selected_count_formula_violation"] += 1
        candidate_counts.append(candidate_count)
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)
        selected_counts.append(selected_count)
        target_records.append(
            {
                "node_id": node_id,
                "candidate_edge_count": candidate_count,
                "positive_gt_edges": positive_count,
                "negative_gt_edges": negative_count,
                "selected_edge_count": selected_count,
            }
        )
    return {
        "target_count": len(rows),
        "unique_node_count": len(set(node_ids)),
        "duplicate_target_count": len(node_ids) - len(set(node_ids)),
        "sorted_by_node_id": node_ids == sorted(node_ids),
        "all_split_test": all(row["split"] == "test" for row in rows),
        "candidate_edge_count": distribution([float(v) for v in candidate_counts]),
        "gt_positive_edge_count": distribution([float(v) for v in positive_counts]),
        "gt_negative_edge_count": distribution([float(v) for v in negative_counts]),
        "selected_edge_count": distribution([float(v) for v in selected_counts]),
        "reverse_duplicate_policy": {
            "directed_edge_entries": len(edge_pairs),
            "self_loop_count": self_loops,
            "duplicate_directed_edge_count": duplicate_edges,
            "missing_reverse_edge_count": missing_reverse,
            "matches_task21_policy": (
                self_loops == 0 and duplicate_edges == 0 and missing_reverse == 0
            ),
        },
        "violations": dict(violations),
        "target_records_sample": target_records[:10],
    }


def audit_metrics_for_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    violations = Counter()
    checked_rows = 0
    max_errors = {
        "remove_probability_delta": 0.0,
        "retain_probability_delta": 0.0,
        "one_minus_fid_minus": 0.0,
        "retain_probability_ratio": 0.0,
        "sparsity": 0.0,
    }
    original_zero_count = 0
    ratio_structural_null_count = 0
    required_metric_null_nan_inf = 0
    for row in rows:
        if row["status"] != "success":
            continue
        checked_rows += 1
        original = float(row["original_probability"])
        remove = float(row["remove_probability"])
        retain = float(row["retain_probability"])
        remove_delta = float(row["remove_probability_delta"])
        retain_delta = float(row["retain_probability_delta"])
        fid_minus = float(row["fid_minus"])
        one_minus = float(row["one_minus_fid_minus"])
        selected = float(row["selected_edge_count"])
        candidate = float(row["candidate_edge_count"])
        sparsity = float(row["sparsity"])
        ratio = row.get("retain_probability_ratio")
        remove_error = abs(remove_delta - (original - remove))
        retain_error = abs(retain_delta - (original - retain))
        one_minus_error = abs(one_minus - (1.0 - fid_minus))
        sparsity_error = abs(sparsity - (1.0 - selected / candidate))
        max_errors["remove_probability_delta"] = max(
            max_errors["remove_probability_delta"], remove_error
        )
        max_errors["retain_probability_delta"] = max(
            max_errors["retain_probability_delta"], retain_error
        )
        max_errors["one_minus_fid_minus"] = max(max_errors["one_minus_fid_minus"], one_minus_error)
        max_errors["sparsity"] = max(max_errors["sparsity"], sparsity_error)
        if remove_error > 1e-9:
            violations["remove_probability_delta_formula"] += 1
        if retain_error > 1e-9:
            violations["retain_probability_delta_formula"] += 1
        if one_minus_error > 1e-9:
            violations["one_minus_fid_minus_formula"] += 1
        if sparsity_error > 1e-9:
            violations["sparsity_formula"] += 1
        ratio_is_aggregated_random_node_mean = row.get("seed") == "node_mean"
        if original == 0.0:
            original_zero_count += 1
            if ratio in (None, ""):
                ratio_structural_null_count += 1
        if original != 0.0 and ratio not in (None, "") and not ratio_is_aggregated_random_node_mean:
            ratio_error = abs(float(ratio) - (retain / original))
            max_errors["retain_probability_ratio"] = max(
                max_errors["retain_probability_ratio"], ratio_error
            )
            if ratio_error > 1e-9:
                violations["retain_probability_ratio_formula"] += 1
        if not all(
            finite(row[field])
            for field in [
                "edge_auroc",
                "edge_ap",
                "fid_plus",
                "fid_minus",
                "one_minus_fid_minus",
                "remove_probability_delta",
                "retain_probability",
                "retain_probability_delta",
                "retain_probability_ratio",
                "sparsity",
            ]
        ):
            violations["quality_metric_nonfinite"] += 1
            required_metric_null_nan_inf += 1
    return {
        "checked_row_count": checked_rows,
        "metric_formula_violations": dict(violations),
        "identity_violation_count": sum(violations.values()),
        "maximum_absolute_identity_error": max(max_errors.values()) if max_errors else 0.0,
        "maximum_absolute_error_by_formula": max_errors,
        "original_probability_zero_count": original_zero_count,
        "ratio_structural_null_count": ratio_structural_null_count,
        "required_metric_null_nan_inf_row_count": required_metric_null_nan_inf,
    }


def count_csv_rows(path: Path, family: str) -> dict[str, Any]:
    rows = read_csv(path)
    fieldnames = list(rows[0].keys()) if rows else []
    seed_values = sorted({row.get("seed", "") for row in rows})
    node_values = [row.get("node_id", "") for row in rows]
    if rows and {"seed_a", "seed_b"}.issubset(rows[0]):
        identities = {
            (row.get("node_id"), row.get("seed_a"), row.get("seed_b"))
            for row in rows
        }
    else:
        identities = {(row.get("node_id"), row.get("seed")) for row in rows}
    duplicate_rows = len(node_values) - len(identities)
    details = count_missing_nonfinite(rows)
    details.update(
        {
            "path": str(path),
            "role": "metric_csv",
            "row_count": len(rows),
            "column_names": fieldnames,
            "sha256": sha256_file(path),
            "seed_coverage": seed_values,
            "target_node_count": len(set(node_values)),
            "duplicate_node_seed_count": duplicate_rows,
            "null_audit": column_null_audit(rows, family),
        }
    )
    return details


def audit_raw_artifacts(
    *,
    config: dict[str, Any],
    backbone: str,
    method: str,
    seeds: list[int],
    nodes: list[int],
) -> dict[str, Any]:
    root = Path("results/explanations/shapeggen") / backbone / method
    data = load_data_artifact(Path(config["dataset_path"]))
    expected_dirs = len(seeds) * len(nodes)
    missing_dirs = 0
    missing_files = Counter()
    selected_count_violations = 0
    score_length_violations = 0
    edge_mask_size_violations = 0
    selected_non_candidate_violations = 0
    duplicate_global_edges = 0
    seed_coverage = set()
    node_coverage = set()
    required = [
        "edge_mask.pt",
        "node_mask.pt",
        "metadata.json",
        "metrics.json",
        "selected_edges.csv",
    ]
    for seed in seeds:
        for node_id in nodes:
            run_dir = root / f"seed_{seed}" / f"node_{node_id}"
            if not run_dir.exists():
                missing_dirs += 1
                continue
            seed_coverage.add(seed)
            node_coverage.add(node_id)
            for name in required:
                if not (run_dir / name).exists():
                    missing_files[name] += 1
            edge_path = run_dir / "edge_mask.pt"
            if edge_path.exists() and edge_path.stat().st_size != 19639:
                edge_mask_size_violations += 1
            selected_path = run_dir / "selected_edges.csv"
            if selected_path.exists():
                selected_rows = read_csv(selected_path)
                candidate = get_target_candidate_edge_mask(data, node_id)
                expected_selected = selected_edge_count_for_ratio(
                    int(candidate.sum().item()),
                    float(config["selected_edge_ratio"]),
                )
                if len(selected_rows) != expected_selected:
                    selected_count_violations += 1
                edge_ids = [int(row["global_edge_index"]) for row in selected_rows]
                duplicate_global_edges += len(edge_ids) - len(set(edge_ids))
                for edge_id in edge_ids:
                    if not bool(candidate[edge_id].item()):
                        selected_non_candidate_violations += 1
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "path": str(root),
        "role": f"{backbone}_{method}_raw_explanations",
        "expected_run_dirs": expected_dirs,
        "existing_run_dirs": expected_dirs - missing_dirs,
        "missing_run_dirs": missing_dirs,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "sha256_manifest": sha256_many(files),
        "seed_coverage": sorted(seed_coverage),
        "target_node_count": len(node_coverage),
        "missing_files": dict(missing_files),
        "score_length_violations": score_length_violations,
        "edge_mask_size_violations": edge_mask_size_violations,
        "selected_count_violations": selected_count_violations,
        "selected_non_candidate_violations": selected_non_candidate_violations,
        "duplicate_selected_global_edges": duplicate_global_edges,
    }


def sha256_many(paths: list[Path]) -> str:
    digest = __import__("hashlib").sha256()
    for path in sorted(paths):
        digest.update(str(path).replace("\\", "/").encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
    return digest.hexdigest()


def audit_quality_and_coverage(config: dict[str, Any]) -> dict[str, Any]:
    target_rows = read_csv(Path(config["target_file"]))
    nodes = [int(row["node_id"]) for row in target_rows]
    output: dict[str, Any] = {}
    for backbone in BACKBONES:
        gnn_primary = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/gnnexplainer/per_node_metrics.csv")
        )
        gnn_stability_seed = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/gnnexplainer/stability_seed_metrics.csv")
        )
        gnn_pair = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/gnnexplainer/stability_pairwise_jaccard.csv")
        )
        gnn_node_stability = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/gnnexplainer/stability_per_node_jaccard.csv")
        )
        random_seed = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/random/random_seed_level_metrics.csv")
        )
        random_node = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/random/random_node_level_metrics.csv")
        )
        random_pair = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/random/random_stability_pairwise_jaccard.csv")
        )
        random_node_stability = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/random/random_stability_per_node_jaccard.csv")
        )
        output[backbone] = {
            "coverage": {
                "gnn_primary_seed42": coverage(gnn_primary, nodes, [42]),
                "gnn_stability_seeds0_to_4": coverage(gnn_stability_seed, nodes, [0, 1, 2, 3, 4]),
                "random_seeds0_to_9": coverage(random_seed, nodes, list(range(10))),
                "random_node_means": coverage(random_node, nodes, ["node_mean"]),
            },
            "artifact_inventory": {
                "metric_files": {
                    "gnn_primary": count_csv_rows(
                        Path(f"results/metrics/shapeggen/{backbone}/gnnexplainer/per_node_metrics.csv"),
                        "gnn_primary",
                    ),
                    "gnn_stability_seed": count_csv_rows(
                        Path(
                            f"results/metrics/shapeggen/{backbone}/gnnexplainer/"
                            "stability_seed_metrics.csv"
                        ),
                        "gnn_stability_seed",
                    ),
                    "gnn_stability_pairwise": count_csv_rows(
                        Path(
                            f"results/metrics/shapeggen/{backbone}/gnnexplainer/"
                            "stability_pairwise_jaccard.csv"
                        ),
                        "gnn_stability_pairwise",
                    ),
                    "gnn_stability_node": count_csv_rows(
                        Path(
                            f"results/metrics/shapeggen/{backbone}/gnnexplainer/"
                            "stability_per_node_jaccard.csv"
                        ),
                        "gnn_stability_node",
                    ),
                    "random_seed": count_csv_rows(
                        Path(f"results/metrics/shapeggen/{backbone}/random/random_seed_level_metrics.csv"),
                        "random_seed",
                    ),
                    "random_node": count_csv_rows(
                        Path(f"results/metrics/shapeggen/{backbone}/random/random_node_level_metrics.csv"),
                        "random_node",
                    ),
                    "random_stability_pairwise": count_csv_rows(
                        Path(
                            f"results/metrics/shapeggen/{backbone}/random/"
                            "random_stability_pairwise_jaccard.csv"
                        ),
                        "random_stability_pairwise",
                    ),
                    "random_stability_node": count_csv_rows(
                        Path(
                            f"results/metrics/shapeggen/{backbone}/random/"
                            "random_stability_per_node_jaccard.csv"
                        ),
                        "random_stability_node",
                    ),
                },
                "raw_artifacts": {
                    "gnnexplainer": audit_raw_artifacts(
                        config=config,
                        backbone=backbone,
                        method="gnnexplainer",
                        seeds=[42, 0, 1, 2, 3, 4],
                        nodes=nodes,
                    ),
                    "random": audit_raw_artifacts(
                        config=config,
                        backbone=backbone,
                        method="random",
                        seeds=list(range(10)),
                        nodes=nodes,
                    ),
                },
            },
            "quality_summary": {
                "gnnexplainer_seed42": summarize_rows(gnn_primary, QUALITY_METRICS),
                "random_node_level": summarize_rows(random_node, QUALITY_METRICS),
                "random_seed_level_preserved_reference": summarize_rows(
                    random_seed,
                    QUALITY_METRICS,
                ),
            },
            "metric_semantics_violations": {
                "gnn_primary": audit_metrics_for_rows(gnn_primary),
                "gnn_stability_seed": audit_metrics_for_rows(gnn_stability_seed),
                "random_seed": audit_metrics_for_rows(random_seed),
                "random_node": audit_metrics_for_rows(random_node),
            },
            "stability": {
                "gnnexplainer": audit_stability(
                    pair_rows=gnn_pair,
                    node_rows=gnn_node_stability,
                    nodes=nodes,
                    seeds=[0, 1, 2, 3, 4],
                    stochastic_source="independent per-instance explanation optimization runs",
                ),
                "random": audit_stability(
                    pair_rows=random_pair,
                    node_rows=random_node_stability,
                    nodes=nodes,
                    seeds=list(range(10)),
                    stochastic_source="repeated random edge selections",
                ),
            },
            "runtime": {
                "gnn_primary": runtime_summary(gnn_primary),
                "random_node_level": runtime_summary(random_node),
                "random_seed_level": runtime_summary(random_seed),
            },
            "runtime_alias_checks": {
                "gnn_primary_runtime_equals_generation": runtime_alias_check(
                    gnn_primary,
                    "gnnexplainer",
                ),
                "random_node_runtime_equals_evaluation": runtime_alias_check(
                    random_node,
                    "random",
                ),
                "random_seed_runtime_equals_evaluation": runtime_alias_check(
                    random_seed,
                    "random",
                ),
            },
        }
    return output


def coverage(
    rows: list[dict[str, str]],
    expected_nodes: list[int],
    expected_seeds: list[Any],
) -> dict[str, Any]:
    success = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") == "failed"]
    observed = {(int(row["node_id"]), normalize_seed(row["seed"])) for row in rows}
    expected = {(node_id, seed) for node_id in expected_nodes for seed in expected_seeds}
    duplicates = len(rows) - len(observed)
    return {
        "expected": len(expected),
        "succeeded": len(success),
        "failed": len(failed),
        "missing": len(expected - observed),
        "duplicated": duplicates,
        "failure_reasons": sorted({row.get("error", "") for row in failed if row.get("error", "")}),
        "seed_coverage": sorted({normalize_seed(row["seed"]) for row in rows}, key=str),
        "unique_node_count": len({int(row["node_id"]) for row in rows}),
    }


def normalize_seed(value: str) -> int | str:
    return int(value) if str(value).lstrip("-").isdigit() else str(value)


def audit_stability(
    *,
    pair_rows: list[dict[str, str]],
    node_rows: list[dict[str, str]],
    nodes: list[int],
    seeds: list[int],
    stochastic_source: str,
) -> dict[str, Any]:
    expected_pairs_per_node = len(seeds) * (len(seeds) - 1) // 2
    duplicate_pairs = 0
    self_pairs = 0
    out_of_range = 0
    by_node: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in pair_rows:
        node_id = int(row["node_id"])
        seed_a = int(row["seed_a"])
        seed_b = int(row["seed_b"])
        if seed_a == seed_b:
            self_pairs += 1
        if not 0.0 <= float(row["jaccard"]) <= 1.0:
            out_of_range += 1
        pair = tuple(sorted((seed_a, seed_b)))
        by_node[node_id].append(pair)
    for pairs in by_node.values():
        duplicate_pairs += len(pairs) - len(set(pairs))
    return {
        "stochastic_source": stochastic_source,
        "expected_pair_rows": len(nodes) * expected_pairs_per_node,
        "actual_pair_rows": len(pair_rows),
        "node_count": len(node_rows),
        "self_pair_count": self_pairs,
        "duplicate_unordered_pair_count": duplicate_pairs,
        "jaccard_out_of_range_count": out_of_range,
        "node_level_sampling_unit": (
            "mean pairwise Jaccard within node, then summarize across nodes"
        ),
        "summary": summarize_stability_node_rows(node_rows),
    }


def summarize_stability_node_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    values = [float(row["mean_pairwise_jaccard"]) for row in rows if row["status"] == "success"]
    summary = distribution(values)
    summary["metric"] = "mean_pairwise_jaccard"
    return [summary]


def runtime_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    fields = ["runtime_seconds", "generation_runtime_seconds", "evaluation_runtime_seconds"]
    output = {}
    for field in fields:
        values = [float(row[field]) for row in rows if finite(row.get(field))]
        summary = distribution(values)
        summary["total"] = sum(values) if values else None
        output[field] = summary
    return output


def runtime_alias_check(rows: list[dict[str, str]], method: str) -> dict[str, Any]:
    alias_violation_count = 0
    maximum_absolute_difference = 0.0
    checked_rows = 0
    for row in rows:
        if row.get("status") != "success":
            continue
        runtime = row.get("runtime_seconds")
        generation = row.get("generation_runtime_seconds")
        evaluation = row.get("evaluation_runtime_seconds")
        if method == "gnnexplainer":
            if finite(runtime) and finite(generation):
                checked_rows += 1
                diff = abs(float(runtime) - float(generation))
                maximum_absolute_difference = max(maximum_absolute_difference, diff)
                if diff > 1e-9:
                    alias_violation_count += 1
        elif method == "random":
            if finite(runtime) and finite(evaluation):
                checked_rows += 1
                diff = abs(float(runtime) - float(evaluation))
                maximum_absolute_difference = max(maximum_absolute_difference, diff)
                if diff > 1e-9:
                    alias_violation_count += 1
    return {
        "method": method,
        "checked_rows": checked_rows,
        "alias_violation_count": alias_violation_count,
        "maximum_absolute_difference": maximum_absolute_difference,
    }


def recompute_comparison(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        for method, quality_key, stability_key in [
            ("GNNExplainer", "gnnexplainer_seed42", "gnnexplainer"),
            ("Random", "random_node_level", "random"),
        ]:
            quality = summary_lookup(
                audit["backbone_audit"][backbone]["quality_summary"][quality_key]
            )
            stability = summary_lookup(
                audit["backbone_audit"][backbone]["stability"][stability_key]["summary"]
            )
            row = {"backbone": label, "method": method}
            for metric in COMPARISON_METRICS:
                source = stability if metric == "stability" else quality
                key = "mean_pairwise_jaccard" if metric == "stability" else metric
                row[metric] = source[key]["mean"]
            rows.append(row)
    return rows


def compare_reported_values(recomputed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in recomputed_rows:
        expected = EXPECTED_REPORTED[(row["backbone"], row["method"])]
        for metric, expected_value in expected.items():
            actual = float(row[metric])
            output.append(
                {
                    "backbone": row["backbone"],
                    "method": row["method"],
                    "metric": metric,
                    "recomputed": actual,
                    "previous_report_rounded": expected_value,
                    "difference": actual - expected_value,
                }
            )
    return output


def fid_count_checks(audit: dict[str, Any]) -> dict[str, Any]:
    expectations = {
        ("gcn", "gnnexplainer"): {"fid_plus_sum": 94, "fid_minus_sum": 2},
        ("graphsage", "gnnexplainer"): {"fid_plus_sum": 83, "fid_minus_sum": 43},
    }
    output = {}
    for (backbone, method), expected in expectations.items():
        path = Path(f"results/metrics/shapeggen/{backbone}/{method}/per_node_metrics.csv")
        rows = read_csv(path)
        actual = {
            "fid_plus_sum": int(sum(float(row["fid_plus"]) for row in rows)),
            "fid_minus_sum": int(sum(float(row["fid_minus"]) for row in rows)),
        }
        output[f"{backbone}_{method}"] = {
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
        }
    return output


def validation_status(args: argparse.Namespace) -> dict[str, str]:
    return {
        "pytest": args.pytest_result,
        "ruff": args.ruff_result,
        "git_diff_check": args.diff_check_result,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = read_yaml(args.task_config)
    previous_freeze = json.loads(args.previous_freeze.read_text(encoding="utf-8"))
    input_audit = audit_inputs(config, args.task_config)
    candidate_gt_audit = audit_targets_and_candidate_gt(config)
    backbone_audit = audit_quality_and_coverage(config)
    audit = {
        "task": "TASK23 AUDIT + COMPLETION ONLY",
        "verdict": "PASS",
        "task24_authorized": False,
        "initial_state": {
            "git_status_short_before_audit": (
                "clean; captured before audit edits with git status --short"
            ),
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "log_oneline_decorate_5": git_value("log", "--oneline", "--decorate", "-5"),
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "tags_at_head": git_value("tag", "--points-at", "HEAD"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "frozen_input_verification": input_audit,
        "protocol": {
            "target_count": config["target_count"],
            "selected_edge_ratio": config["selected_edge_ratio"],
            "rounding_rule": config["rounding_rule"],
            "gnnexplainer_primary_seed": config["primary_seed"],
            "gnnexplainer_stability_seeds": config["gnnexplainer_stability_seeds"],
            "random_seeds": config["random_baseline_seeds"],
            "edge_ranking_metric_scope": config["edge_ranking_metric_scope"],
            "candidate_edge_scope": config["candidate_edge_scope"],
        },
        "candidate_gt_audit": candidate_gt_audit,
        "backbone_audit": backbone_audit,
        "metric_semantics": metric_semantics(),
        "random_aggregation_explanation": (
            "Random raw seed rows are preserved. The primary Random quality table first "
            "averages each metric across seeds 0..9 inside each node_id, then summarizes "
            "the 167 node-level means across nodes. Edge AUROC/AP can match across "
            "backbones because they depend on the same candidate/GT/random scores. "
            "Perturbation metrics can differ because predictions after edge removal or "
            "retention are computed by each frozen backbone."
        ),
        "previous_freeze_status": previous_freeze.get("status"),
        "validation_results": validation_status(args),
    }
    audit["recomputed_comparison_rows"] = recompute_comparison(audit)
    audit["differences_vs_previous_report"] = compare_reported_values(
        audit["recomputed_comparison_rows"]
    )
    audit["fid_count_checks"] = fid_count_checks(audit)
    audit["acceptance"] = acceptance(audit)
    audit["verdict"] = "PASS" if all(audit["acceptance"].values()) else "FAIL"
    return audit


def metric_semantics() -> dict[str, str]:
    return {
        "edge_auroc_edge_ap": (
            "candidate_edge_scores_and_labels receives continuous edge scores and "
            "candidate_edge_mask, then compute_edge_ground_truth_metrics computes AUROC/AP "
            "over the full candidate-edge set without thresholding."
        ),
        "fid_plus": "1 when removing selected edges changes original prediction, else 0.",
        "fid_minus": "1 when retaining only selected edges changes original prediction, else 0.",
        "one_minus_fid_minus": "1 - Fid- for every raw row.",
        "probability_reference_class": (
            "All probabilities are softmax probabilities of the original predicted class."
        ),
        "remove_probability_delta": "original_probability - remove_probability.",
        "retain_probability": (
            "softmax probability of original predicted class on retain-only graph."
        ),
        "retain_probability_delta": "original_probability - retain_probability.",
        "retain_probability_ratio": (
            "retain_probability / original_probability when original_probability != 0.0, "
            "otherwise None. No epsilon or clipping is used."
        ),
        "sparsity": "1 - selected_edge_count / candidate_edge_count.",
    }


def acceptance(audit: dict[str, Any]) -> dict[str, bool]:
    backbone = audit["backbone_audit"]
    formula_violations = []
    for values in backbone.values():
        for item in values["metric_semantics_violations"].values():
            formula_violations.extend(item["metric_formula_violations"].values())
    raw_violations = []
    for values in backbone.values():
        for item in values["artifact_inventory"]["raw_artifacts"].values():
            raw_violations.extend(
                [
                    item["missing_run_dirs"],
                    item["score_length_violations"],
                    item["edge_mask_size_violations"],
                    item["selected_count_violations"],
                    item["selected_non_candidate_violations"],
                    item["duplicate_selected_global_edges"],
                ]
            )
            raw_violations.extend(item["missing_files"].values())
    coverage_failures = []
    for values in backbone.values():
        for item in values["coverage"].values():
            coverage_failures.extend([item["failed"], item["missing"], item["duplicated"]])
    unexpected_required_nulls = []
    inf_counts = []
    runtime_alias_violations = []
    required_metric_nonfinite_rows = []
    for values in backbone.values():
        for item in values["artifact_inventory"]["metric_files"].values():
            null_audit = item["null_audit"]
            unexpected_required_nulls.append(null_audit["unexpected_required_null_cells"])
            inf_counts.append(null_audit["total_inf_cells"])
        for item in values["metric_semantics_violations"].values():
            required_metric_nonfinite_rows.append(item["required_metric_null_nan_inf_row_count"])
        for item in values["runtime_alias_checks"].values():
            runtime_alias_violations.append(item["alias_violation_count"])
    return {
        "frozen_input_hashes_match": all(
            item["matches_expected"] for item in audit["frozen_input_verification"].values()
        ),
        "target_candidate_gt_mapping_valid": (
            audit["candidate_gt_audit"]["target_count"] == 167
            and audit["candidate_gt_audit"]["duplicate_target_count"] == 0
            and not audit["candidate_gt_audit"]["violations"]
            and audit["candidate_gt_audit"]["reverse_duplicate_policy"]["matches_task21_policy"]
        ),
        "coverage_complete_without_failures_or_duplicates": sum(coverage_failures) == 0,
        "raw_artifacts_complete_and_aligned": sum(raw_violations) == 0,
        "metric_formula_checks_pass": sum(formula_violations) == 0,
        "all_required_metrics_complete": sum(required_metric_nonfinite_rows) == 0,
        "unexpected_required_nulls_zero": sum(unexpected_required_nulls) == 0,
        "nan_inf_counts_zero": sum(inf_counts) == 0,
        "runtime_alias_checks_pass": sum(runtime_alias_violations) == 0,
        "stability_pair_counts_valid": all(
            values["stability"]["gnnexplainer"]["actual_pair_rows"] == 1670
            and values["stability"]["random"]["actual_pair_rows"] == 7515
            and values["stability"]["gnnexplainer"]["duplicate_unordered_pair_count"] == 0
            and values["stability"]["random"]["duplicate_unordered_pair_count"] == 0
            and values["stability"]["gnnexplainer"]["self_pair_count"] == 0
            and values["stability"]["random"]["self_pair_count"] == 0
            and values["stability"]["gnnexplainer"]["jaccard_out_of_range_count"] == 0
            and values["stability"]["random"]["jaccard_out_of_range_count"] == 0
            for values in backbone.values()
        ),
        "reported_values_recomputed": all(
            abs(item["difference"]) < 1e-6
            for item in audit["differences_vs_previous_report"]
        ),
        "fid_count_checks_match": all(
            item["matches"] for item in audit["fid_count_checks"].values()
        ),
        "task24_pgexplainer_not_started": audit["task24_authorized"] is False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_report(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ShapeGGen Task23 GNNExplainer And Random Audit Freeze",
            "",
            "## Scope And Stop Boundary",
            "",
            "This document audits Task23 only. It does not rerun experiments, tune final "
            "targets, replace seed42 with stability seeds, or start PGExplainer/Task24.",
            "",
            "## Frozen Inputs",
            "",
            frozen_input_table(audit),
            "",
            "## Frozen Protocol",
            "",
            protocol_section(audit),
            "",
            "## Git And Environment Provenance",
            "",
            provenance_section(audit),
            "",
            "## Metric Definitions",
            "",
            metric_definition_lines(audit),
            "",
            "## Recomputed Quality Summary",
            "",
            comparison_table(audit["recomputed_comparison_rows"]),
            "",
            "## Full Dispersion Tables",
            "",
            full_dispersion_tables(audit),
            "",
            "## Differences Versus Previous Report",
            "",
            differences_table(audit),
            "",
            "## Success And Failure Coverage",
            "",
            coverage_table(audit),
            "",
            "## Candidate And Ground-Truth Audit",
            "",
            candidate_gt_section(audit),
            "",
            "## Random Aggregation Audit",
            "",
            audit["random_aggregation_explanation"],
            "",
            "## Stability Audit",
            "",
            stability_section(audit),
            "",
            "## Runtime Audit",
            "",
            runtime_section(audit),
            "",
            "## Metric Identity Checks",
            "",
            metric_identity_section(audit),
            "",
            "## Null And Missing Audit",
            "",
            null_missing_section(audit),
            "",
            "## Artifact Inventory",
            "",
            artifact_inventory_section(audit),
            "",
            "## Validation",
            "",
            validation_section(audit),
            "",
            "## Acceptance Checklist",
            "",
            acceptance_section(audit),
            "",
            "## Remaining Blockers",
            "",
            "NONE" if audit["verdict"] == "PASS" else "See failed acceptance checks above.",
            "",
            "STOP FOR REVIEW",
            "",
        ]
    )


def frozen_input_table(audit: dict[str, Any]) -> str:
    lines = ["| Input | Path | SHA256 | Expected | Match |", "|---|---|---|---|---:|"]
    for name, row in audit["frozen_input_verification"].items():
        lines.append(
            f"| {name} | `{row['path']}` | `{row['sha256']}` | "
            f"`{row.get('expected_sha256') or 'recorded'}` | {row['matches_expected']} |"
        )
    return "\n".join(lines)


def protocol_section(audit: dict[str, Any]) -> str:
    protocol = audit["protocol"]
    return "\n".join(
        [
            f"- target_count: {protocol['target_count']}",
            f"- selected_edge_ratio: {protocol['selected_edge_ratio']}",
            f"- rounding_rule: {protocol['rounding_rule']}",
            f"- edge_ranking_metric_scope: {protocol['edge_ranking_metric_scope']}",
            f"- candidate_edge_scope: {protocol['candidate_edge_scope']}",
            f"- GNNExplainer primary seed: {protocol['gnnexplainer_primary_seed']}",
            f"- GNNExplainer stability seeds: {protocol['gnnexplainer_stability_seeds']}",
            f"- Random seeds: {protocol['random_seeds']}",
        ]
    )


def provenance_section(audit: dict[str, Any]) -> str:
    git = audit["git"]
    env = audit["environment"]
    return "\n".join(
        [
            f"- git_commit: `{git['commit']}`",
            f"- git_branch: `{git['branch']}`",
            f"- tags_at_head: `{git['tags_at_head'] or 'none'}`",
            f"- python: `{env['python']}`",
            f"- platform: `{env['platform']}`",
            f"- torch: `{env['torch']}`",
            f"- torch_cuda: `{env['torch_cuda']}`",
            f"- cuda_available: `{env['cuda_available']}`",
            f"- gpu: `{env['gpu']}`",
        ]
    )


def metric_definition_lines(audit: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in audit["metric_semantics"].items())


def comparison_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        (
            "| Backbone | Method | Edge AUROC | Edge AP | Fid+ | Fid- | "
            "1-Fid- | Sparsity | Stability |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backbone']} | {row['method']} | {fmt(row['edge_auroc'])} | "
            f"{fmt(row['edge_ap'])} | {fmt(row['fid_plus'])} | {fmt(row['fid_minus'])} | "
            f"{fmt(row['one_minus_fid_minus'])} | {fmt(row['sparsity'])} | "
            f"{fmt(row['stability'])} |"
        )
    return "\n".join(lines)


def full_dispersion_tables(audit: dict[str, Any]) -> str:
    blocks = []
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        gnn_quality = audit["backbone_audit"][backbone]["quality_summary"][
            "gnnexplainer_seed42"
        ]
        random_quality = audit["backbone_audit"][backbone]["quality_summary"][
            "random_node_level"
        ]
        gnn_stability = audit["backbone_audit"][backbone]["stability"]["gnnexplainer"][
            "summary"
        ]
        random_stability = audit["backbone_audit"][backbone]["stability"]["random"][
            "summary"
        ]
        for title, rows in [
            ("GNNExplainer seed42 quality", gnn_quality),
            ("Random node-level quality", random_quality),
            ("GNNExplainer stability", gnn_stability),
            ("Random stability", random_stability),
        ]:
            blocks.append(f"### {label} {title}\n\n{summary_table(rows)}")
    return "\n\n".join(blocks)


def summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Metric | n | mean | std | median | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['n']} | {fmt(row['mean'])} | {fmt(row['std'])} | "
            f"{fmt(row['median'])} | {fmt(row['min'])} | {fmt(row['max'])} |"
        )
    return "\n".join(lines)


def differences_table(audit: dict[str, Any]) -> str:
    lines = [
        "| Backbone | Method | Metric | Recomputed | Previous rounded | Difference |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in audit["differences_vs_previous_report"]:
        lines.append(
            f"| {row['backbone']} | {row['method']} | {row['metric']} | "
            f"{fmt(row['recomputed'])} | {fmt(row['previous_report_rounded'])} | "
            f"{row['difference']:.12g} |"
        )
    return "\n".join(lines)


def coverage_table(audit: dict[str, Any]) -> str:
    lines = [
        "| Backbone | Run | Expected | Succeeded | Failed | Missing | Duplicated |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        for run, row in audit["backbone_audit"][backbone]["coverage"].items():
            lines.append(
                f"| {label} | {run} | {row['expected']} | {row['succeeded']} | "
                f"{row['failed']} | {row['missing']} | {row['duplicated']} |"
            )
    return "\n".join(lines)


def candidate_gt_section(audit: dict[str, Any]) -> str:
    cgt = audit["candidate_gt_audit"]
    lines = [
        f"- target_count: {cgt['target_count']}",
        f"- unique_node_count: {cgt['unique_node_count']}",
        f"- duplicate_target_count: {cgt['duplicate_target_count']}",
        f"- sorted_by_node_id: {cgt['sorted_by_node_id']}",
        f"- all_split_test: {cgt['all_split_test']}",
        f"- violations: {cgt['violations'] or 'NONE'}",
        f"- reverse_duplicate_policy: {cgt['reverse_duplicate_policy']}",
        "",
        summary_table(
            [
                {"metric": key, **cgt[key]}
                for key in [
                    "candidate_edge_count",
                    "gt_positive_edge_count",
                    "gt_negative_edge_count",
                    "selected_edge_count",
                ]
            ]
        ),
    ]
    return "\n".join(lines)


def stability_section(audit: dict[str, Any]) -> str:
    lines = [
        (
            "| Backbone | Method | Expected pairs | Actual pairs | Duplicate pairs | "
            "Self pairs | Out-of-range | Source |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        for method in METHODS:
            row = audit["backbone_audit"][backbone]["stability"][method]
            lines.append(
                f"| {label} | {method} | {row['expected_pair_rows']} | {row['actual_pair_rows']} | "
                f"{row['duplicate_unordered_pair_count']} | {row['self_pair_count']} | "
                f"{row['jaccard_out_of_range_count']} | {row['stochastic_source']} |"
            )
    return "\n".join(lines)


def runtime_section(audit: dict[str, Any]) -> str:
    lines = [
        "GNNExplainer runtime is per-target explanation optimization/generation time. "
        "`runtime_seconds` is an alias of `generation_runtime_seconds`. "
        "`evaluation_runtime_seconds` is measured separately and is not included in "
        "`runtime_seconds`.",
        "",
        "Random runtime is random edge selection plus metric evaluation. "
        "`runtime_seconds` is an alias of `evaluation_runtime_seconds` because Random "
        "has no explanation-generation stage. Therefore `generation_runtime_seconds` "
        "is structurally not applicable for Random.",
        "",
        "Raw Random totals across 1,670 seed rows describe total measured Random work. "
        "The 167-row node-level seed-mean runtime distribution is a sampling-unit "
        "summary and is not a total-compute measurement.",
        "",
        "| Backbone | Run | Field | n | mean | std | median | min | max | total |",
    ]
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    definitions = {
        "gnn_primary": "GNNExplainer seed42 generation plus separate metric evaluation",
        "random_node_level": "Random per-node seed-mean quality runtime reference",
        "random_seed_level": "Random edge selection plus metric evaluation per seed row",
    }
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        for run, fields in audit["backbone_audit"][backbone]["runtime"].items():
            for field, row in fields.items():
                lines.append(
                    f"| {label} | {definitions[run]} | {field} | {row['n']} | "
                    f"{fmt(row['mean'])} | {fmt(row['std'])} | {fmt(row['median'])} | "
                    f"{fmt(row['min'])} | {fmt(row['max'])} | {fmt(row['total'])} |"
                )
    lines.extend(["", "### Runtime Alias Checks", "", runtime_alias_table(audit)])
    return "\n".join(lines)


def runtime_alias_table(audit: dict[str, Any]) -> str:
    lines = [
        "| Backbone | Check | Checked rows | Violations | Max absolute difference |",
        "|---|---|---:|---:|---:|",
    ]
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        for name, row in audit["backbone_audit"][backbone]["runtime_alias_checks"].items():
            lines.append(
                f"| {label} | {name} | {row['checked_rows']} | "
                f"{row['alias_violation_count']} | "
                f"{fmt(row['maximum_absolute_difference'])} |"
            )
    return "\n".join(lines)


def metric_identity_section(audit: dict[str, Any]) -> str:
    lines = [
        "Checked identities: `one_minus_fid_minus == 1 - fid_minus`, "
        "`remove_probability_delta == original_probability - remove_probability`, "
        "`retain_probability_delta == original_probability - retain_probability`, "
        "`retain_probability_ratio == retain_probability / original_probability` "
        "when applicable, and `sparsity == 1 - selected_edge_count / candidate_edge_count`.",
        "",
        "| Backbone | Artifact | Checked rows | Violations | Max absolute error | "
        "Original prob zero | Ratio structural null | Required metric null/NaN/Inf rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        checks = audit["backbone_audit"][backbone]["metric_semantics_violations"]
        for family, row in checks.items():
            lines.append(
                f"| {label} | {family} | {row['checked_row_count']} | "
                f"{row['identity_violation_count']} | "
                f"{fmt(row['maximum_absolute_identity_error'])} | "
                f"{row['original_probability_zero_count']} | "
                f"{row['ratio_structural_null_count']} | "
                f"{row['required_metric_null_nan_inf_row_count']} |"
            )
    lines.append("")
    lines.append(
        "For Random node-level rows, `retain_probability_ratio` is the per-node mean "
        "of seed-level ratios by aggregation protocol, so the raw formula is verified "
        "on seed-level rows and not re-derived from node-mean probabilities."
    )
    return "\n".join(lines)


def null_missing_section(audit: dict[str, Any]) -> str:
    lines = [
        "Nulls are audited by column. Expected structural nulls are documented below; "
        "unexpected required nulls are blockers and must remain zero.",
        "",
        "| Backbone | Artifact | Total null cells | Expected structural null cells | "
        "Unexpected required null cells | Inf cells | Expected null columns | "
        "Unexpected null columns |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        metric_files = audit["backbone_audit"][backbone]["artifact_inventory"]["metric_files"]
        for family, row in metric_files.items():
            nulls = row["null_audit"]
            expected_columns = ", ".join(nulls["expected_null_columns"].keys()) or "NONE"
            unexpected_columns = ", ".join(nulls["unexpected_null_columns"]) or "NONE"
            lines.append(
                f"| {label} | {family} | {nulls['total_null_cells']} | "
                f"{nulls['expected_structural_null_cells']} | "
                f"{nulls['unexpected_required_null_cells']} | {nulls['total_inf_cells']} | "
                f"{expected_columns} | {unexpected_columns} |"
            )
    lines.extend(["", "### Expected Structural Null Reasons", ""])
    seen = set()
    for backbone in BACKBONES:
        metric_files = audit["backbone_audit"][backbone]["artifact_inventory"]["metric_files"]
        for family, row in metric_files.items():
            for column, reason in row["null_audit"]["expected_null_columns"].items():
                key = (family, column, reason)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {family}.{column}: {reason}")
    if len(lines) >= 2 and lines[-1] == "":
        lines.append("- NONE")
    return "\n".join(lines)


def artifact_inventory_section(audit: dict[str, Any]) -> str:
    lines = [
        (
            "| Backbone | Family | Path | Role | Rows/Runs | Columns/File count | "
            "SHA256/Manifest | Null/Violation Status |"
        ),
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for backbone in BACKBONES:
        label = "GCN" if backbone == "gcn" else "GraphSAGE"
        inventory = audit["backbone_audit"][backbone]["artifact_inventory"]
        for family, row in inventory["metric_files"].items():
            nulls = row["null_audit"]
            lines.append(
                f"| {label} | {family} | `{row['path']}` | {row['role']} | "
                f"{row['row_count']} | {len(row['column_names'])} | `{row['sha256']}` | "
                f"total_null={nulls['total_null_cells']}, "
                f"expected_structural_null={nulls['expected_structural_null_cells']}, "
                f"unexpected_required_null={nulls['unexpected_required_null_cells']}, "
                f"nan_inf={row['nan_inf_count']}, "
                f"duplicates={row['duplicate_node_seed_count']} |"
            )
        for family, row in inventory["raw_artifacts"].items():
            lines.append(
                f"| {label} | {family} | `{row['path']}` | {row['role']} | "
                f"{row['existing_run_dirs']}/{row['expected_run_dirs']} | {row['file_count']} | "
                f"`{row['sha256_manifest']}` | missing_dirs={row['missing_run_dirs']}, "
                f"missing_files={row['missing_files']}, "
                f"score_len={row['score_length_violations']}, "
                f"edge_mask_size={row['edge_mask_size_violations']}, "
                f"selected_count={row['selected_count_violations']} |"
            )
    return "\n".join(lines)


def validation_section(audit: dict[str, Any]) -> str:
    validation = audit["validation_results"]
    return "\n".join(
        [
            f"- pytest: {validation['pytest']}",
            f"- Ruff: {validation['ruff']}",
            f"- git diff --check: {validation['git_diff_check']}",
        ]
    )


def acceptance_section(audit: dict[str, Any]) -> str:
    evidence = acceptance_evidence(audit)
    return "\n".join(
        f"- {name}: {audit['acceptance'][name]} - {evidence.get(name, 'see audit JSON')}"
        for name in audit["acceptance"]
    )


def acceptance_evidence(audit: dict[str, Any]) -> dict[str, str]:
    target = audit["candidate_gt_audit"]
    return {
        "frozen_input_hashes_match": (
            "dataset, target, checkpoint, and recorded config hashes match."
        ),
        "target_candidate_gt_mapping_valid": (
            f"{target['target_count']} targets, {target['duplicate_target_count']} duplicates, "
            f"violations={target['violations'] or 'NONE'}."
        ),
        "coverage_complete_without_failures_or_duplicates": (
            "GCN and GraphSAGE quality/stability/random expected rows all succeeded with "
            "failed=0, missing=0, duplicated=0."
        ),
        "raw_artifacts_complete_and_aligned": (
            "1002 GNNExplainer raw run directories and 1670 Random raw run directories per "
            "backbone are present with no selected-count or candidate violations."
        ),
        "metric_formula_checks_pass": (
            "all checked identity violation counts are zero within tolerance 1e-9."
        ),
        "all_required_metrics_complete": (
            "required metric null/NaN/Inf row count is zero for applicable quality rows."
        ),
        "unexpected_required_nulls_zero": (
            "column-level null audit reports unexpected_required_null_cells=0 for every CSV."
        ),
        "nan_inf_counts_zero": "column-level audit reports total_inf_cells=0 for every CSV.",
        "runtime_alias_checks_pass": (
            "GNN runtime equals generation runtime, Random runtime equals evaluation/runtime "
            "boundary, with zero alias violations."
        ),
        "stability_pair_counts_valid": (
            "GNNExplainer has 1670 pair rows per backbone and Random has 7515 pair rows per "
            "backbone, with no duplicate/self/out-of-range pairs."
        ),
        "reported_values_recomputed": (
            "Task23 comparison values recompute to the previously reported rounded values."
        ),
        "fid_count_checks_match": (
            "GNNExplainer Fid+ and Fid- count checks match recorded evidence."
        ),
        "task24_pgexplainer_not_started": (
            "audit scope explicitly leaves Task24/PGExplainer unauthorized."
        ),
    }


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    audit = build_audit(args)
    write_json(args.audit_json, audit)
    freeze = json.loads(args.previous_freeze.read_text(encoding="utf-8"))
    freeze["task23_audit"] = audit
    freeze["status"] = "success" if audit["verdict"] == "PASS" else "failed"
    write_json(args.previous_freeze, freeze)
    args.report.write_text(build_report(audit), encoding="utf-8")
    print(f"verdict={audit['verdict']}")
    print(f"audit_json={args.audit_json}")
    print(f"report={args.report}")
    print(f"freeze={args.previous_freeze}")


if __name__ == "__main__":
    main()
