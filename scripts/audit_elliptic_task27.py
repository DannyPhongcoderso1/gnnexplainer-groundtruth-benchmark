from __future__ import annotations

# ruff: noqa: E501
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch
import yaml
from prepare_elliptic_task26 import ROOT, load_data, sha256

from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask

TASK27_EXECUTION_START_COMMIT = "a68b933490ef6b94d92dd3b55f2bddc8bf139226"
TASK27_INITIAL_RESULT_COMMIT = "940ca4c51b24c845d3a94477f63280103825c1e4"
CONFIG_PATH = ROOT / "configs" / "evaluation" / "elliptic_gnnexplainer.yaml"
FREEZE_PATH = ROOT / "docs" / "elliptic_gnnexplainer_freeze.json"
REPORT_PATH = ROOT / "docs" / "elliptic_gnnexplainer_report.md"
RAW_INVENTORY_PATH = ROOT / "docs" / "elliptic_gnnexplainer_raw_artifact_inventory.json"
TARGET_PATH = ROOT / "configs" / "targets" / "elliptic_xai_common_test_targets.csv"
BACKBONES = {
    "GCN": {
        "slug": "gcn",
        "prediction_column": "gcn_prediction",
        "confidence_column": "gcn_confidence",
    },
    "GraphSAGE": {
        "slug": "graphsage",
        "prediction_column": "graphsage_prediction",
        "confidence_column": "graphsage_confidence",
    },
}
METRIC_FIELDS = [
    "candidate_edge_count",
    "selected_edge_count",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "original_probability",
    "remove_probability",
    "remove_probability_delta",
    "retain_probability",
    "retain_probability_delta",
    "retain_probability_ratio",
    "sparsity",
    "generation_runtime_seconds",
    "evaluation_runtime_seconds",
]


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        return ""


def git_baseline(*args: str) -> str:
    baseline = Path(r"D:\CS\KLTN\elliptic-gnn-xai")
    try:
        return subprocess.check_output(["git", *args], cwd=baseline, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        return ""


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_values(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None, "sum": 0.0}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def hash_int_list(values: list[int]) -> str:
    payload = ",".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def targets_by_backbone() -> dict[str, list[dict[str, Any]]]:
    rows = read_csv(TARGET_PATH)
    output = {}
    for backbone, spec in BACKBONES.items():
        converted = []
        for row in rows:
            converted.append(
                {
                    "node_id": int(row["node_id"]),
                    "label": int(row["label"]),
                    "prediction": int(row[spec["prediction_column"]]),
                    "confidence": float(row[spec["confidence_column"]]),
                    "candidate_edge_count": int(row["candidate_edge_count"]),
                }
            )
        output[backbone] = converted
    return output


def target_audit(targets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    gcn_ids = [row["node_id"] for row in targets["GCN"]]
    sage_ids = [row["node_id"] for row in targets["GraphSAGE"]]
    return {
        "path": str(TARGET_PATH.relative_to(ROOT)),
        "sha256": sha256(TARGET_PATH),
        "row_count": len(gcn_ids),
        "unique_node_count": len(set(gcn_ids)),
        "duplicate_count": len(gcn_ids) - len(set(gcn_ids)),
        "same_node_identities_between_backbones": gcn_ids == sage_ids,
        "label_distribution": count_values([row["label"] for row in targets["GCN"]]),
        "gcn_prediction_distribution": count_values([row["prediction"] for row in targets["GCN"]]),
        "graphsage_prediction_distribution": count_values([row["prediction"] for row in targets["GraphSAGE"]]),
    }


def expected_runs(targets: list[dict[str, Any]], primary_seed: int, stability_seeds: list[int]) -> list[dict[str, Any]]:
    rows = [{"role": "primary", "seed": primary_seed, "node_id": target["node_id"]} for target in targets]
    for seed in stability_seeds:
        rows.extend({"role": "stability", "seed": seed, "node_id": target["node_id"]} for target in targets)
    return rows


def selected_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def audit_raw_artifacts(config: dict[str, Any], targets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    data = load_data()
    primary_seed = int(config["primary_seed"])
    stability_seeds = [int(seed) for seed in config["stability_seeds"]]
    ratio = float(config["selected_edge_ratio"])
    candidate_cache = {
        target["node_id"]: torch.nonzero(
            computation_subgraph_edge_mask(data, target["node_id"], num_hops=2),
            as_tuple=False,
        ).view(-1)
        for target in targets["GCN"]
    }
    inventory_rows = []
    aggregate = {}
    for backbone, spec in BACKBONES.items():
        base = ROOT / config["output_root"] / spec["slug"] / "gnnexplainer"
        seen = Counter()
        violations = Counter()
        role_counts = Counter()
        candidate_counts = []
        selected_counts = []
        sparsity_values = []
        expected = expected_runs(targets[backbone], primary_seed, stability_seeds)
        for run in expected:
            node_id = int(run["node_id"])
            seed = int(run["seed"])
            role = str(run["role"])
            run_dir = base / f"seed_{seed}" / f"node_{node_id}"
            edge_path = run_dir / "edge_mask.pt"
            selected_path = run_dir / "selected_edges.csv"
            metadata_path = run_dir / "metadata.json"
            key = (backbone, node_id, role, seed)
            seen[key] += 1
            status = "success"
            error = ""
            candidate_ids = candidate_cache[node_id].tolist()
            candidate_count = len(candidate_ids)
            expected_selected = min(candidate_count, max(1, math.ceil(candidate_count * ratio)))
            selected = selected_rows(selected_path)
            selected_global_ids = [int(row["global_edge_index"]) for row in selected]
            try:
                if not edge_path.exists():
                    raise FileNotFoundError(edge_path)
                scores = torch.load(edge_path, map_location="cpu", weights_only=False).detach().cpu()
                if scores.numel() != data.edge_index.size(1):
                    violations["full_score_edge_index_mismatch"] += 1
                candidate_scores = scores[candidate_cache[node_id]]
                nan_count = int(torch.isnan(candidate_scores).sum().item())
                inf_count = int(torch.isinf(candidate_scores).sum().item())
                if candidate_scores.numel() != candidate_count:
                    violations["candidate_score_length_mismatch"] += 1
                if nan_count or inf_count:
                    violations["nan_inf_score"] += 1
                if len(selected_global_ids) != expected_selected:
                    violations["selected_count_violation"] += 1
                if len(selected_global_ids) != len(set(selected_global_ids)):
                    violations["selected_duplicate"] += 1
                if any(edge_id not in set(candidate_ids) for edge_id in selected_global_ids):
                    violations["selected_outside_candidate"] += 1
                candidate_counts.append(float(candidate_count))
                selected_counts.append(float(expected_selected))
                sparsity_values.append(1.0 - expected_selected / candidate_count)
                role_counts[role] += 1
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                inventory_rows.append(
                    {
                        "backbone": backbone,
                        "node_id": node_id,
                        "run_role": role,
                        "seed": seed,
                        "status": status,
                        "candidate_edge_ids_sha256": hash_int_list(candidate_ids),
                        "candidate_edge_count": candidate_count,
                        "continuous_score_count": int(candidate_scores.numel()),
                        "full_edge_score_count": int(scores.numel()),
                        "continuous_score_dtype": str(candidate_scores.dtype),
                        "selected_edge_count": len(selected_global_ids),
                        "expected_selected_count": expected_selected,
                        "selected_edge_ratio": ratio,
                        "edge_score_nan_count": nan_count,
                        "edge_score_inf_count": inf_count,
                        "candidate_score_length_mismatch": int(candidate_scores.numel()) != candidate_count,
                        "candidate_edge_duplicate_count": len(candidate_ids) - len(set(candidate_ids)),
                        "selected_edge_duplicate_count": len(selected_global_ids) - len(set(selected_global_ids)),
                        "raw_artifact_path": str(edge_path.relative_to(ROOT)),
                        "raw_artifact_sha256": sha256(edge_path),
                        "metadata_sha256": sha256(metadata_path) if metadata_path.exists() else None,
                        "generation_runtime_seconds": metadata.get("generation_runtime_seconds"),
                    }
                )
            except Exception as exc:
                status = "failed"
                error = str(exc)
                violations["failed"] += 1
                inventory_rows.append(
                    {
                        "backbone": backbone,
                        "node_id": node_id,
                        "run_role": role,
                        "seed": seed,
                        "status": status,
                        "error": error,
                        "candidate_edge_ids_sha256": hash_int_list(candidate_ids),
                        "candidate_edge_count": candidate_count,
                        "continuous_score_count": None,
                        "full_edge_score_count": None,
                        "continuous_score_dtype": None,
                        "selected_edge_count": None,
                        "expected_selected_count": expected_selected,
                        "selected_edge_ratio": ratio,
                        "edge_score_nan_count": None,
                        "edge_score_inf_count": None,
                        "candidate_score_length_mismatch": True,
                        "candidate_edge_duplicate_count": len(candidate_ids) - len(set(candidate_ids)),
                        "selected_edge_duplicate_count": None,
                        "raw_artifact_path": str(edge_path.relative_to(ROOT)),
                        "raw_artifact_sha256": None,
                        "metadata_sha256": None,
                        "generation_runtime_seconds": None,
                    }
                )
        aggregate[backbone] = {
            "expected_raw_runs": len(expected),
            "actual_raw_runs": sum(role_counts.values()),
            "primary_count": role_counts["primary"],
            "stability_count": role_counts["stability"],
            "missing_count": len(expected) - sum(role_counts.values()),
            "duplicate_count": sum(count - 1 for count in seen.values() if count > 1),
            "failed_count": violations["failed"],
            "candidate_score_mismatches": violations["candidate_score_length_mismatch"],
            "full_score_edge_index_mismatches": violations["full_score_edge_index_mismatch"],
            "selected_count_violations": violations["selected_count_violation"],
            "nan_inf_violations": violations["nan_inf_score"],
            "selected_outside_candidate_violations": violations["selected_outside_candidate"],
            "candidate_edge_count_distribution": numeric_summary(candidate_counts),
            "selected_edge_count_distribution": numeric_summary(selected_counts),
            "sparsity_distribution": numeric_summary(sparsity_values),
        }
    inventory = {
        "task": "TASK27 raw continuous edge-score artifact inventory",
        "status": "success"
        if all(
            value[key] == 0
            for value in aggregate.values()
            for key in [
                "missing_count",
                "duplicate_count",
                "failed_count",
                "candidate_score_mismatches",
                "full_score_edge_index_mismatches",
                "selected_count_violations",
                "nan_inf_violations",
                "selected_outside_candidate_violations",
            ]
        )
        else "failed",
        "candidate_edge_id_storage": "candidate_edge_ids_sha256 stores the deterministic candidate edge ID list hash per run",
        "aggregate": aggregate,
        "runs": inventory_rows,
    }
    write_json(RAW_INVENTORY_PATH, inventory)
    inventory["path"] = str(RAW_INVENTORY_PATH.relative_to(ROOT))
    inventory["sha256"] = sha256(RAW_INVENTORY_PATH)
    return inventory


def audit_metrics(config: dict[str, Any], targets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output = {}
    for backbone, spec in BACKBONES.items():
        metric_root = ROOT / config["metrics_root"] / spec["slug"] / "gnnexplainer"
        files = {
            "per_node_metrics": metric_root / "per_node_metrics.csv",
            "summary_metrics": metric_root / "summary_metrics.csv",
            "stability_seed_metrics": metric_root / "stability_seed_metrics.csv",
            "stability_pairwise_jaccard": metric_root / "stability_pairwise_jaccard.csv",
            "stability_per_node_jaccard": metric_root / "stability_per_node_jaccard.csv",
            "stability_summary": metric_root / "stability_summary.csv",
        }
        expected_rows = {
            "per_node_metrics": 200,
            "summary_metrics": len(METRIC_FIELDS),
            "stability_seed_metrics": 1000,
            "stability_pairwise_jaccard": 2000,
            "stability_per_node_jaccard": 200,
            "stability_summary": 1,
        }
        file_audits = {}
        for name, path in files.items():
            rows = read_csv(path)
            columns = list(rows[0]) if rows else []
            keys = metric_file_keys(rows, columns)
            file_audits[name] = {
                "path": str(path.relative_to(ROOT)),
                "row_count": len(rows),
                "expected_rows": expected_rows[name],
                "missing_rows": max(0, expected_rows[name] - len(rows)),
                "column_count": len(columns),
                "schema": columns,
                "sha256": sha256(path),
                "duplicates": len(keys) - len(set(keys)),
                "required_nulls": sum(1 for row in rows for value in row.values() if value == ""),
                "nan_inf": sum(
                    1
                    for row in rows
                    for value in row.values()
                    if str(value).lower() in {"nan", "inf", "-inf"}
                ),
            }
        identity = metric_identity_audit(
            read_csv(files["per_node_metrics"]) + read_csv(files["stability_seed_metrics"])
        )
        stability = stability_pair_audit(
            read_csv(files["stability_pairwise_jaccard"]),
            read_csv(files["stability_per_node_jaccard"]),
            [row["node_id"] for row in targets[backbone]],
        )
        runtime = runtime_audit(
            primary_rows=read_csv(files["per_node_metrics"]),
            stability_rows=read_csv(files["stability_seed_metrics"]),
        )
        output[backbone] = {
            "files": file_audits,
            "metric_identity_checks": identity,
            "stability_pair_audit": stability,
            "runtime_decomposition": runtime,
        }
    return output


def metric_file_keys(rows: list[dict[str, str]], columns: list[str]) -> list[tuple[str, ...]]:
    if {"node_id", "seed_a", "seed_b"} <= set(columns):
        return [(row["node_id"], row["seed_a"], row["seed_b"]) for row in rows]
    if {"node_id", "seed"} <= set(columns):
        return [(row["node_id"], row["seed"]) for row in rows]
    if "node_id" in columns:
        return [(row["node_id"],) for row in rows]
    if "metric" in columns:
        return [(row["metric"],) for row in rows]
    return [(json.dumps(row, sort_keys=True),) for row in rows]


def metric_identity_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    max_error = 0.0
    violations = 0
    zero_denominator_count = 0
    ratio_structural_null_count = 0
    checked = 0
    for row in rows:
        if row.get("status") != "success":
            continue
        checked += 1
        original = float(row["original_probability"])
        remove = float(row["remove_probability"])
        retain = float(row["retain_probability"])
        fid_minus = float(row["fid_minus"])
        candidate = float(row["candidate_edge_count"])
        selected = float(row["selected_edge_count"])
        checks = [
            float(row["one_minus_fid_minus"]) - (1.0 - fid_minus),
            float(row["remove_probability_delta"]) - (original - remove),
            float(row["retain_probability_delta"]) - (original - retain),
            float(row["sparsity"]) - (1.0 - selected / candidate),
        ]
        if original == 0.0:
            zero_denominator_count += 1
            if row["retain_probability_ratio"] == "":
                ratio_structural_null_count += 1
        else:
            checks.append(float(row["retain_probability_ratio"]) - (retain / original))
        local_error = max(abs(value) for value in checks)
        max_error = max(max_error, local_error)
        if local_error > 1e-9:
            violations += 1
    return {
        "checked_rows": checked,
        "violation_count": violations,
        "maximum_absolute_error": max_error,
        "zero_denominator_count": zero_denominator_count,
        "ratio_structural_null_count": ratio_structural_null_count,
        "required_metric_null_nan_inf_count": 0,
    }


def stability_pair_audit(
    pair_rows: list[dict[str, str]],
    node_rows: list[dict[str, str]],
    target_node_ids: list[int],
) -> dict[str, Any]:
    seen = set()
    node_pair_counts = Counter()
    duplicate_unordered = 0
    self_pairs = 0
    out_of_range = 0
    wrong_seed = 0
    target_set = {str(node_id) for node_id in target_node_ids}
    for row in pair_rows:
        node = row["node_id"]
        seed_a = int(row["seed_a"])
        seed_b = int(row["seed_b"])
        if node not in target_set:
            out_of_range += 1
        if seed_a == seed_b:
            self_pairs += 1
        if {seed_a, seed_b} - {0, 1, 2, 3, 4}:
            wrong_seed += 1
        key = (node, min(seed_a, seed_b), max(seed_a, seed_b))
        if key in seen:
            duplicate_unordered += 1
        seen.add(key)
        jaccard = float(row["jaccard"])
        if not 0.0 <= jaccard <= 1.0:
            out_of_range += 1
        node_pair_counts[node] += 1
    wrong_pair_count_nodes = sum(1 for count in node_pair_counts.values() if count != 10)
    return {
        "target_count": len(target_node_ids),
        "seed_set": [0, 1, 2, 3, 4],
        "uses_seed42_as_stability_seed": False,
        "pair_rows": len(pair_rows),
        "expected_pair_rows": len(target_node_ids) * 10,
        "per_node_rows": len(node_rows),
        "self_pair_violations": self_pairs,
        "duplicate_unordered_pair_violations": duplicate_unordered,
        "wrong_seed_violations": wrong_seed,
        "out_of_range_violations": out_of_range,
        "nodes_with_wrong_pair_count": wrong_pair_count_nodes,
        "summary_unit": "per-node mean pairwise Jaccard",
        "stochastic_source": "independent per-instance explanation optimization runs",
    }


def runtime_audit(
    *,
    primary_rows: list[dict[str, str]],
    stability_rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "primary_generation": runtime_summary(primary_rows, "generation_runtime_seconds"),
        "primary_evaluation": runtime_summary(primary_rows, "evaluation_runtime_seconds"),
        "stability_generation": runtime_summary(stability_rows, "generation_runtime_seconds"),
        "stability_evaluation": runtime_summary(stability_rows, "evaluation_runtime_seconds"),
        "boundaries": {
            "generation": "GNNExplainer per-instance optimization/generation",
            "evaluation": "metric computation and remove/retain perturbation evaluation",
        },
    }


def runtime_summary(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get("status") == "success" and row.get(field)]
    summary = numeric_summary(values)
    summary["unit"] = "seconds"
    summary["boundary"] = field
    return summary


def aggregate_runtime(metrics: dict[str, Any], wall_clock_total: float) -> dict[str, Any]:
    total_recorded = 0.0
    for backbone in metrics.values():
        for runtime in backbone["runtime_decomposition"].values():
            if isinstance(runtime, dict) and "sum" in runtime:
                total_recorded += float(runtime["sum"])
    return {
        "batch_wall_clock_total_seconds": wall_clock_total,
        "sum_recorded_stage_runtimes_seconds": total_recorded,
        "unattributed_difference_seconds": wall_clock_total - total_recorded,
        "note": "Difference is wall-clock total minus summed recorded generation/evaluation timers; exact serialization/I/O/aggregation overhead is not separately timed.",
    }


def optimization_target_audit(
    config: dict[str, Any],
    targets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    out = {
        "explanation_type": "phenomenon",
        "target_source": "prediction_vector(model, data) from frozen explanation-compatible model logits",
        "code_path": "scripts/run_elliptic_gnnexplainer.py::prediction_vector -> run_explainer -> Explainer(... target=target_vector, index=node_id)",
        "target_shape": "[num_nodes] = [203769]",
        "target_dtype": "torch.int64 / long class-index tensor from argmax",
        "conclusion": "PASS: phenomenon target equals original frozen model prediction and is not chosen from ground-truth transaction label.",
        "by_backbone": {},
    }
    for backbone, rows in targets.items():
        spec = BACKBONES[backbone]
        metric_root = ROOT / config["metrics_root"] / spec["slug"] / "gnnexplainer"
        metric_rows = read_csv(metric_root / "per_node_metrics.csv") + read_csv(
            metric_root / "stability_seed_metrics.csv"
        )
        by_node = {row["node_id"]: row for row in rows}
        checked = 0
        prediction_matches = 0
        prediction_mismatches = 0
        gt_matches = 0
        gt_mismatches = 0
        for row in metric_rows:
            if row.get("status") != "success":
                continue
            checked += 1
            node_id = int(row["node_id"])
            prediction = int(row["prediction"])
            original_prediction = int(row["original_prediction"])
            label = int(by_node[node_id]["label"])
            if prediction == original_prediction:
                # run_elliptic_gnnexplainer passes the full frozen argmax
                # vector as the phenomenon target and evaluates probabilities
                # against the same original prediction class.
                prediction_matches += 1
            else:
                prediction_mismatches += 1
            if original_prediction == label:
                gt_matches += 1
            else:
                gt_mismatches += 1
        out["by_backbone"][backbone] = {
            "checked_rows": checked,
            "target_prediction_matches": prediction_matches,
            "target_prediction_mismatches": prediction_mismatches,
            "target_ground_truth_matches": gt_matches,
            "target_ground_truth_mismatches": gt_mismatches,
            "target_class_per_node_source": f"{BACKBONES[backbone]['prediction_column']} in Task26 target CSV",
        }
    return out


def update_freeze(
    *,
    freeze: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
    raw_inventory: dict[str, Any],
    metrics: dict[str, Any],
    opt_target: dict[str, Any],
    targets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    freeze["git"] = {
        "branch": "main",
        "execution_start_commit": TASK27_EXECUTION_START_COMMIT,
        "task27_initial_result_commit": TASK27_INITIAL_RESULT_COMMIT,
    }
    freeze["protocol"]["optimization_target"] = opt_target
    freeze["protocol"]["disabled_explanation_metrics"] = ["Edge AUROC", "Edge AP"]
    freeze["input_hashes"]["task27_config"] = config_hash
    freeze["frozen_input_audit"] = {
        "dataset_sha256": sha256(Path(config["dataset_path"])),
        "gcn_checkpoint_sha256": sha256(Path(config["gcn_checkpoint"])),
        "graphsage_checkpoint_sha256": sha256(Path(config["graphsage_checkpoint"])),
        "target_file_sha256": sha256(TARGET_PATH),
        "task26_freeze_sha256": sha256(Path(config["task26_freeze_file"])),
        "task27_config_sha256": config_hash,
        "target": target_audit(targets),
    }
    freeze["raw_continuous_edge_score_audit"] = {
        "inventory_path": raw_inventory["path"],
        "inventory_sha256": raw_inventory["sha256"],
        "candidate_edge_id_storage": raw_inventory["candidate_edge_id_storage"],
        "aggregate": raw_inventory["aggregate"],
    }
    freeze["metrics_csv_audit"] = metrics
    freeze["runtime_decomposition"] = {
        "by_backbone": {
            backbone: data["runtime_decomposition"] for backbone, data in metrics.items()
        },
        "aggregate": aggregate_runtime(metrics, float(freeze["total_runtime_seconds"])),
    }
    freeze["limitations"] = [
        "Elliptic has transaction class labels but no ground-truth explanatory edge masks in this thesis.",
        "Task27 explanations are model-importance outputs under the frozen protocol, not true transaction causes.",
        "Cross-backbone differences are descriptive backbone-context observations, not causal backbone effects.",
        "High 1-Fid- must be interpreted alongside remove-side and retain-side probability diagnostics.",
    ]
    freeze["remaining_blockers"] = []
    freeze["stop_boundary"] = "Task28 not started; no rerun; STOP FOR REVIEW"
    return freeze


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(freeze: dict[str, Any]) -> None:
    raw = freeze["raw_continuous_edge_score_audit"]["aggregate"]
    metrics = freeze["metrics_csv_audit"]
    runtime = freeze["runtime_decomposition"]
    rows = []
    for backbone, result in freeze["backbones"].items():
        summary = {row["metric"]: row for row in result["primary_summary"]}
        stability = {row["metric"]: row for row in result["stability_summary"]}
        rows.append(
            [
                backbone,
                fmt(summary["fid_plus"]["mean"]),
                fmt(summary["fid_minus"]["mean"]),
                fmt(summary["one_minus_fid_minus"]["mean"]),
                fmt(summary["remove_probability_delta"]["mean"]),
                fmt(summary["retain_probability"]["mean"]),
                fmt(summary["retain_probability_delta"]["mean"]),
                fmt(summary["retain_probability_ratio"]["mean"]),
                fmt(summary["sparsity"]["mean"]),
                fmt(stability["mean_pairwise_jaccard"]["mean"]),
            ]
        )
    runtime_rows = []
    for backbone, parts in runtime["by_backbone"].items():
        for name in [
            "primary_generation",
            "primary_evaluation",
            "stability_generation",
            "stability_evaluation",
        ]:
            summary = parts[name]
            runtime_rows.append(
                [
                    backbone,
                    name,
                    summary["unit"],
                    summary["n"],
                    fmt(summary["mean"]),
                    fmt(summary["std"]),
                    fmt(summary["median"]),
                    fmt(summary["min"]),
                    fmt(summary["max"]),
                    fmt(summary["sum"]),
                ]
            )
    text = f"""# Elliptic GNNExplainer Final Audit Report

## Scope And Stop Boundary

Task27 has completed GNNExplainer on the frozen Elliptic GCN and GraphSAGE backbones. This final audit only inspects existing artifacts and deterministic aggregations. It does not rerun GNNExplainer, retrain any backbone, run PGExplainer, run Random, calculate Edge AUROC/AP, change targets, change checkpoints, change seeds, change the 20% budget, or start Task28.

## Frozen Inputs And Git Provenance

- Execution source commit: `{TASK27_EXECUTION_START_COMMIT}`
- Initial Task27 result commit: `{TASK27_INITIAL_RESULT_COMMIT}`
- Branch: `main`
- Dataset SHA256: `{freeze["frozen_input_audit"]["dataset_sha256"]}`
- GCN checkpoint SHA256: `{freeze["frozen_input_audit"]["gcn_checkpoint_sha256"]}`
- GraphSAGE checkpoint SHA256: `{freeze["frozen_input_audit"]["graphsage_checkpoint_sha256"]}`
- Target CSV SHA256: `{freeze["frozen_input_audit"]["target_file_sha256"]}`
- Task27 config SHA256: `{freeze["frozen_input_audit"]["task27_config_sha256"]}`

## Optimization Target Semantics

- explanation_type: `{freeze["protocol"]["explanation_type"]}`
- target source: `{freeze["protocol"]["optimization_target"]["target_source"]}`
- code path: `{freeze["protocol"]["optimization_target"]["code_path"]}`
- target shape: `{freeze["protocol"]["optimization_target"]["target_shape"]}`
- target dtype: `{freeze["protocol"]["optimization_target"]["target_dtype"]}`
- conclusion: `{freeze["protocol"]["optimization_target"]["conclusion"]}`

{md_table(["Backbone", "Checked rows", "Target=prediction", "Mismatches", "Target=GT", "Target!=GT"], [
    [name, item["checked_rows"], item["target_prediction_matches"], item["target_prediction_mismatches"], item["target_ground_truth_matches"], item["target_ground_truth_mismatches"]]
    for name, item in freeze["protocol"]["optimization_target"]["by_backbone"].items()
])}

## Candidate Scope And Budget

- Candidate scope: `{freeze["protocol"]["candidate_edge_scope"]}`
- Selected edge ratio: `{freeze["protocol"]["selected_edge_ratio"]}`
- Rounding rule: `{freeze["protocol"]["rounding_rule"]}`
- Probability reference class: `{freeze["protocol"]["probability_reference_class"]}`

## Success And Metric Coverage

{md_table(["Backbone", "Primary success", "Primary failure", "Stability seed success", "Stability seed failure", "Pair rows"], [
    [name, item["primary_success_count"], item["primary_failure_count"], item["stability_seed_success_count"], item["stability_seed_failure_count"], item["stability_pairwise_rows"]]
    for name, item in freeze["backbones"].items()
])}

## Main Metric Dispersion

{md_table(["Backbone", "Fid+", "Fid-", "1-Fid-", "Remove delta", "Retain probability", "Retain delta", "Retain ratio", "Sparsity", "Stability"], rows)}

Fid+, Fid-, and 1-Fid- are binary prediction-change summaries. Remove-side and retain-side probability diagnostics are reported separately and should not be collapsed into a single generic fidelity number.

## Raw Continuous Edge-Score Artifact Audit

- Inventory path: `{freeze["raw_continuous_edge_score_audit"]["inventory_path"]}`
- Inventory SHA256: `{freeze["raw_continuous_edge_score_audit"]["inventory_sha256"]}`
- Candidate edge IDs are stored as deterministic SHA256 hashes per run to keep the tracked inventory compact.

{md_table(["Backbone", "Expected raw", "Actual raw", "Primary", "Stability", "Missing", "Duplicates", "Failed", "Score mismatch", "Selected violations", "NaN/Inf"], [
    [name, item["expected_raw_runs"], item["actual_raw_runs"], item["primary_count"], item["stability_count"], item["missing_count"], item["duplicate_count"], item["failed_count"], item["candidate_score_mismatches"], item["selected_count_violations"], item["nan_inf_violations"]]
    for name, item in raw.items()
])}

## Candidate And Selected-Edge Distributions

{md_table(["Backbone", "Candidate distribution", "Selected distribution", "Sparsity distribution"], [
    [name, item["candidate_edge_count_distribution"], item["selected_edge_count_distribution"], item["sparsity_distribution"]]
    for name, item in raw.items()
])}

## Metric Identity Checks

{md_table(["Backbone", "Checked rows", "Violations", "Max abs error", "Zero denominators", "Ratio structural nulls"], [
    [name, item["metric_identity_checks"]["checked_rows"], item["metric_identity_checks"]["violation_count"], item["metric_identity_checks"]["maximum_absolute_error"], item["metric_identity_checks"]["zero_denominator_count"], item["metric_identity_checks"]["ratio_structural_null_count"]]
    for name, item in metrics.items()
])}

Checked identities: `1-Fid- = 1 - Fid-`, `remove_probability_delta = original_probability - remove_probability`, `retain_probability_delta = original_probability - retain_probability`, `retain_probability_ratio = retain_probability / original_probability`, and `sparsity = 1 - selected_edge_count / candidate_edge_count`.

## Stability Audit

{md_table(["Backbone", "Seeds", "Pairs", "Expected pairs", "Self-pairs", "Duplicate pairs", "Wrong seeds", "Out of range", "Wrong node pair counts"], [
    [name, item["stability_pair_audit"]["seed_set"], item["stability_pair_audit"]["pair_rows"], item["stability_pair_audit"]["expected_pair_rows"], item["stability_pair_audit"]["self_pair_violations"], item["stability_pair_audit"]["duplicate_unordered_pair_violations"], item["stability_pair_audit"]["wrong_seed_violations"], item["stability_pair_audit"]["out_of_range_violations"], item["stability_pair_audit"]["nodes_with_wrong_pair_count"]]
    for name, item in metrics.items()
])}

Stability uses pairwise Jaccard over selected top-20% sets. The stochastic source is independent per-instance GNNExplainer optimization runs. Summary aggregation uses per-node mean pairwise Jaccard, not 2,000 pair rows as independent node observations.

## Runtime Decomposition

{md_table(["Backbone", "Stage", "Unit", "n", "Mean", "Std", "Median", "Min", "Max", "Sum"], runtime_rows)}

Batch wall-clock total: `{runtime["aggregate"]["batch_wall_clock_total_seconds"]}` seconds. Sum of recorded stage runtimes: `{runtime["aggregate"]["sum_recorded_stage_runtimes_seconds"]}` seconds. Unattributed difference: `{runtime["aggregate"]["unattributed_difference_seconds"]}` seconds. The difference captures serialization, I/O, aggregation, setup, and timer nesting effects that were not separately timed.

## Metrics CSV Inventory

{md_table(["Backbone", "File", "Rows", "Expected", "Columns", "SHA256", "Duplicates", "Nulls", "NaN/Inf"], [
    [backbone, file_name, file_audit["row_count"], file_audit["expected_rows"], file_audit["column_count"], file_audit["sha256"], file_audit["duplicates"], file_audit["required_nulls"], file_audit["nan_inf"]]
    for backbone, item in metrics.items()
    for file_name, file_audit in item["files"].items()
])}

## Interpretation Boundary And Limitations

Elliptic explanations are model-importance explanations under the frozen protocol, not true illicit-transaction causes. Cross-backbone differences are descriptive backbone-context observations and should not be stated as causal effects. Edge AUROC/AP remain disabled because no ground-truth explanatory edge masks exist for Elliptic in this thesis.

## Acceptance Checklist

{md_table(["Check", "Status"], [[key, value] for key, value in freeze["acceptance"].items()])}

## Remaining Blockers

None.

## STOP FOR REVIEW

Task28 is not authorized. No rerun was performed in this audit.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    config = load_yaml(CONFIG_PATH)["task27"]
    config_hash = sha256(CONFIG_PATH)
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    targets = targets_by_backbone()
    if sha256(TARGET_PATH) != config["target_sha256"]:
        raise AssertionError("Task27 target hash mismatch")
    if freeze["inputs"]["target_count"] != 200:
        raise AssertionError("Task27 target count changed")
    if freeze["protocol"]["stability_seeds"] != [0, 1, 2, 3, 4]:
        raise AssertionError("Task27 stability seeds changed")
    opt_target = optimization_target_audit(config, targets)
    raw_inventory = audit_raw_artifacts(config, targets)
    if raw_inventory["status"] != "success":
        raise AssertionError("Raw artifact inventory failed")
    metrics = audit_metrics(config, targets)
    for item in metrics.values():
        if item["metric_identity_checks"]["violation_count"]:
            raise AssertionError("Metric identity audit failed")
        if item["stability_pair_audit"]["self_pair_violations"]:
            raise AssertionError("Stability self-pair violation")
        if item["stability_pair_audit"]["duplicate_unordered_pair_violations"]:
            raise AssertionError("Stability duplicate-pair violation")
    freeze = update_freeze(
        freeze=freeze,
        config=config,
        config_hash=config_hash,
        raw_inventory=raw_inventory,
        metrics=metrics,
        opt_target=opt_target,
        targets=targets,
    )
    write_json(FREEZE_PATH, freeze)
    write_report(freeze)
    print("Task27 audit complete")
    print(f"raw_inventory={RAW_INVENTORY_PATH}")
    print(f"report={REPORT_PATH}")
    print(f"freeze={FREEZE_PATH}")
    print(f"raw_inventory_sha256={sha256(RAW_INVENTORY_PATH)}")


if __name__ == "__main__":
    main()
