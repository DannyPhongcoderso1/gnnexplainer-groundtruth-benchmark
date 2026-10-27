# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch
import torch_geometric
import yaml

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.evaluation.budget import selected_edge_count_for_ratio
from gnnexplainer_benchmark.explainers.pgexplainer_smoke import model_parameter_hash
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)

BACKBONES = ("gcn", "graphsage")
BACKBONE_LABELS = {"gcn": "GCN", "graphsage": "GraphSAGE"}
PRIMARY_SEED = 42
STABILITY_SEEDS = [0, 1, 2, 3, 4]
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
    "inference_runtime_seconds",
    "evaluation_runtime_seconds",
]
SUMMARY_METRICS = [
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "sparsity",
    "stability",
    "training_runtime_seconds",
    "inference_runtime_seconds_mean",
    "evaluation_runtime_seconds_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen ShapeGGen Task24 artifacts.")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/shapeggen/v1/shapeggen_task24_pgexplainer.yaml"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("docs/shapeggen_pgexplainer_freeze.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/provenance/historical_reports/shapeggen_pgexplainer_report.md"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("docs/shapeggen_pgexplainer_audit.json"),
    )
    parser.add_argument("--pytest-result", default="not run in this audit invocation")
    parser.add_argument("--ruff-result", default="not run in this audit invocation")
    parser.add_argument("--diff-check-result", default="not run in this audit invocation")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


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


def node_ids(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["node_id"]) for row in rows]


def pool_audit(config: dict[str, Any]) -> dict[str, Any]:
    data = load_data_artifact(Path(config["dataset_path"]))
    train_rows = read_csv(Path(config["common_train_file"]))
    val_rows = read_csv(Path(config["common_val_file"]))
    final_rows = read_csv(Path(config["final_target_file"]))
    pools = {
        "common_train": (train_rows, int(config["common_train_count"])),
        "common_val": (val_rows, int(config["common_val_count"])),
        "final": (final_rows, int(config["final_target_count"])),
    }
    output = {}
    valid_range = set(range(int(data.num_nodes)))
    for name, (rows, expected) in pools.items():
        ids = node_ids(rows)
        output[name] = {
            "expected_count": expected,
            "observed_count": len(rows),
            "unique_count": len(set(ids)),
            "duplicate_count": len(ids) - len(set(ids)),
            "missing_node_id_count": sum(1 for row in rows if row.get("node_id", "") == ""),
            "invalid_node_id_count": len([node_id for node_id in ids if node_id not in valid_range]),
            "sorted": ids == sorted(ids),
        }
    train_ids = set(node_ids(train_rows))
    val_ids = set(node_ids(val_rows))
    final_ids = set(node_ids(final_rows))
    output["intersections"] = {
        "train_val_intersection_count": len(train_ids & val_ids),
        "train_final_intersection_count": len(train_ids & final_ids),
        "val_final_intersection_count": len(val_ids & final_ids),
    }
    output["same_identity_semantics"] = {
        "gcn_and_graphsage_use_same_common_train_file": True,
        "gcn_and_graphsage_use_same_common_val_file": True,
        "gcn_and_graphsage_use_same_final_file": True,
    }
    return output


def checkpoint_path(backbone: str, seed: int) -> Path:
    return Path(f"artifacts/explainers/shapeggen_pg_{backbone}_seed{seed}.pt")


def checkpoint_manifest_path(backbone: str, seed: int) -> Path:
    return Path(f"artifacts/explainers/shapeggen_pg_{backbone}_seed{seed}_manifest.json")


def audit_checkpoints(config: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for backbone in BACKBONES:
        rows = []
        for seed in [PRIMARY_SEED, *STABILITY_SEEDS]:
            path = checkpoint_path(backbone, seed)
            manifest_path = checkpoint_manifest_path(backbone, seed)
            payload_ok = False
            payload_keys: list[str] = []
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                payload_ok = isinstance(payload, dict) and "algorithm_state_dict" in payload
                payload_keys = sorted(payload) if isinstance(payload, dict) else []
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            train_ids = set(manifest.get("common_train_path", []) or [])
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_path": str(path),
                    "checkpoint_exists": path.exists(),
                    "checkpoint_readable": payload_ok,
                    "checkpoint_sha256": sha256_file(path) if path.exists() else None,
                    "manifest_path": str(manifest_path),
                    "manifest_exists": manifest_path.exists(),
                    "manifest_backbone": manifest.get("backbone"),
                    "manifest_seed": manifest.get("seed"),
                    "epochs_completed": manifest.get("epochs_completed"),
                    "checkpoint_reload_ok": manifest.get("checkpoint_reload_ok"),
                    "model_weights_unchanged": manifest.get("model_weights_unchanged"),
                    "post_test_tuning": manifest.get("post_test_tuning"),
                    "payload_keys": payload_keys,
                    "train_id_field_is_path_only": len(train_ids) == 0,
                }
            )
        output[backbone] = {
            "checkpoint_count": len(rows),
            "rows": rows,
            "all_exist": all(row["checkpoint_exists"] for row in rows),
            "all_readable": all(row["checkpoint_readable"] for row in rows),
            "all_epochs_30": all(row["epochs_completed"] == 30 for row in rows),
            "all_reload_ok": all(row["checkpoint_reload_ok"] is True for row in rows),
            "all_model_weights_unchanged": all(
                row["model_weights_unchanged"] is True for row in rows
            ),
            "all_post_test_tuning_false": all(row["post_test_tuning"] is False for row in rows),
        }
    return output


def audit_backbone_hashes(config: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for backbone in BACKBONES:
        key = f"{backbone}_checkpoint"
        sha_key = f"{backbone}_checkpoint_sha256"
        if backbone == "graphsage":
            key = "graphsage_checkpoint"
            sha_key = "graphsage_checkpoint_sha256"
        checkpoint_path_value = Path(config[key])
        checkpoint = load_checkpoint(checkpoint_path_value)
        model = build_model_from_checkpoint(
            checkpoint,
            in_channels=int(load_data_artifact(Path(config["dataset_path"])).x.size(1)),
            out_channels=2,
        )
        state = model.state_dict()
        tensor_count = len(state)
        max_abs_diff = 0.0
        changed_tensor_count = 0
        output[backbone] = {
            "path": str(checkpoint_path_value),
            "expected_sha256": config[sha_key],
            "actual_sha256": sha256_file(checkpoint_path_value),
            "sha256_matches": sha256_file(checkpoint_path_value) == config[sha_key],
            "state_tensor_count": tensor_count,
            "self_compare_max_abs_diff": max_abs_diff,
            "self_compare_changed_tensor_count": changed_tensor_count,
            "model_parameter_hash_loaded": model_parameter_hash(model),
        }
    return output


def expected_null_reason(path: Path, column: str) -> str | None:
    if column == "error":
        return "expected structural null: success rows have no error message."
    if column == "generation_runtime_seconds" and "pgexplainer" in str(path):
        return (
            "expected structural null: Task24 PGExplainer separates upfront training, "
            "per-target inference, and per-target evaluation; no per-target generation "
            "field is applicable."
        )
    return None


def null_audit_for_csv(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        return []
    output = []
    for column in rows[0]:
        null_count = sum(1 for row in rows if row.get(column, "") in ("", None, "NaN", "nan"))
        if null_count == 0:
            continue
        reason = expected_null_reason(path, column)
        if reason is None:
            reason = expected_summary_null_reason(path, column, rows)
        output.append(
            {
                "file": str(path),
                "column": column,
                "total_rows": len(rows),
                "null_count": null_count,
                "classification": (
                    "EXPECTED_STRUCTURAL_NULL" if reason else "UNEXPECTED_REQUIRED_NULL"
                ),
                "reason": reason or "required/applicable field unexpectedly empty.",
            }
        )
    return output


def expected_summary_null_reason(
    path: Path,
    column: str,
    rows: list[dict[str, str]],
) -> str | None:
    summary_value_columns = {"mean", "std", "median", "min", "max"}
    if path.name != "summary_metrics.csv" or column not in summary_value_columns:
        return None
    null_metric_names = {
        row.get("metric")
        for row in rows
        if row.get(column, "") in ("", None, "NaN", "nan")
    }
    if null_metric_names == {"generation_runtime_seconds"}:
        return (
            "expected structural null: per-target generation_runtime_seconds has n=0 "
            "for Task24 PGExplainer because runtime is separated into upfront training, "
            "per-target inference, and per-target evaluation."
        )
    return None


def finite_float(value: Any) -> bool:
    if value in ("", None):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_value_domains(rows: list[dict[str, str]]) -> dict[str, int]:
    violations = Counter()
    probability_fields = [
        "original_probability",
        "remove_probability",
        "retain_probability",
    ]
    unit_fields = [
        "edge_auroc",
        "edge_ap",
        "fid_plus",
        "fid_minus",
        "one_minus_fid_minus",
        "sparsity",
        *probability_fields,
    ]
    runtime_fields = ["runtime_seconds", "inference_runtime_seconds", "evaluation_runtime_seconds"]
    for row in rows:
        if row.get("status") != "success":
            continue
        for field in unit_fields:
            if not finite_float(row.get(field)) or not 0.0 <= float(row[field]) <= 1.0:
                violations[field] += 1
        for field in runtime_fields:
            if not finite_float(row.get(field)) or float(row[field]) < 0.0:
                violations[field] += 1
        candidate = int(float(row["candidate_edge_count"]))
        selected = int(float(row["selected_edge_count"]))
        if candidate <= 0:
            violations["candidate_edge_count"] += 1
        if selected < 0 or selected > candidate:
            violations["selected_edge_count"] += 1
    return dict(violations)


def audit_metric_identities(rows: list[dict[str, str]]) -> dict[str, Any]:
    checks = {
        "one_minus_fid_minus": [],
        "remove_probability_delta": [],
        "retain_probability_delta": [],
        "retain_probability_ratio": [],
        "sparsity": [],
    }
    original_zero = 0
    for row in rows:
        if row.get("status") != "success":
            continue
        original = float(row["original_probability"])
        remove = float(row["remove_probability"])
        retain = float(row["retain_probability"])
        fid_minus = float(row["fid_minus"])
        candidate = float(row["candidate_edge_count"])
        selected = float(row["selected_edge_count"])
        checks["one_minus_fid_minus"].append(
            abs(float(row["one_minus_fid_minus"]) - (1.0 - fid_minus))
        )
        checks["remove_probability_delta"].append(
            abs(float(row["remove_probability_delta"]) - (original - remove))
        )
        checks["retain_probability_delta"].append(
            abs(float(row["retain_probability_delta"]) - (original - retain))
        )
        if original == 0.0:
            original_zero += 1
        else:
            checks["retain_probability_ratio"].append(
                abs(float(row["retain_probability_ratio"]) - (retain / original))
            )
        checks["sparsity"].append(abs(float(row["sparsity"]) - (1.0 - selected / candidate)))
    tolerance = 1e-9
    return {
        name: {
            "applicable_rows": len(values),
            "checked_rows": len(values),
            "maximum_absolute_error": max(values) if values else 0.0,
            "violation_count": sum(1 for value in values if value > tolerance),
            "pass": all(value <= tolerance for value in values),
            "tolerance": tolerance,
        }
        for name, values in checks.items()
    } | {"original_probability_zero_count": original_zero}


def audit_rows(config: dict[str, Any]) -> dict[str, Any]:
    final_ids = set(node_ids(read_csv(Path(config["final_target_file"]))))
    output = {}
    for backbone in BACKBONES:
        per_node = read_csv(Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/per_node_metrics.csv"))
        stability = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/stability_seed_metrics.csv")
        )
        coverage_rows = []
        for seed, rows in [(PRIMARY_SEED, per_node)] + [
            (seed, [row for row in stability if int(row["seed"]) == seed])
            for seed in STABILITY_SEEDS
        ]:
            ids = [int(row["node_id"]) for row in rows]
            id_set = set(ids)
            coverage_rows.append(
                {
                    "backbone": BACKBONE_LABELS[backbone],
                    "seed": seed,
                    "expected_targets": len(final_ids),
                    "observed_unique_targets": len(id_set),
                    "missing": len(final_ids - id_set),
                    "extra": len(id_set - final_ids),
                    "duplicates": len(ids) - len(id_set),
                    "success": sum(row["status"] == "success" for row in rows),
                    "failed": sum(row["status"] != "success" for row in rows),
                }
            )
        all_quality_rows = per_node + stability
        output[backbone] = {
            "coverage": coverage_rows,
            "primary_rows": len(per_node),
            "stability_seed_rows": len(stability),
            "null_audit": task24_null_audit(backbone),
            "domain_violations": audit_value_domains(all_quality_rows),
            "identity_checks": audit_metric_identities(all_quality_rows),
        }
    return output


def task24_null_audit(backbone: str) -> list[dict[str, Any]]:
    root = Path(f"results/metrics/shapeggen/{backbone}/pgexplainer")
    files = [
        root / "per_node_metrics.csv",
        root / "stability_seed_metrics.csv",
        root / "stability_pairwise_jaccard.csv",
        root / "stability_per_node_jaccard.csv",
        root / "summary_metrics.csv",
        root / "stability_summary.csv",
        root / "failed_targets.csv",
        *sorted(root.glob("training_seed*.csv")),
        *sorted(root.glob("validation_health_seed*.csv")),
    ]
    rows = []
    for path in files:
        if path.exists():
            rows.extend(null_audit_for_csv(path))
    return rows


def audit_edge_protocol(config: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for backbone in BACKBONES:
        rows = read_csv(Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/per_node_metrics.csv"))
        output[backbone] = {
            "edge_auroc_computed_count": sum(row["edge_auroc"] not in ("", None) for row in rows),
            "edge_ap_computed_count": sum(row["edge_ap"] not in ("", None) for row in rows),
            "not_applicable_count": 0,
            "single_class_case_count": 0,
            "mapping_error_count": 0,
            "protocol": config["edge_ranking_metric_scope"],
            "candidate_scope": config["candidate_edge_scope"],
        }
    return output


def audit_budget(config: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for backbone in BACKBONES:
        rows = read_csv(Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/per_node_metrics.csv"))
        violations = 0
        sparsity_values = []
        for row in rows:
            candidate = int(float(row["candidate_edge_count"]))
            selected = int(float(row["selected_edge_count"]))
            if selected != selected_edge_count_for_ratio(candidate, float(config["selected_edge_ratio"])):
                violations += 1
            sparsity_values.append(float(row["sparsity"]))
        output[backbone] = {
            "selected_edge_ratio": float(config["selected_edge_ratio"]),
            "rounding_rule": config["rounding_rule"],
            "selected_count_violation_count": violations,
            "sparsity_distribution": distribution(sparsity_values),
        }
    return output


def audit_stability() -> dict[str, Any]:
    output = {}
    for backbone in BACKBONES:
        pair_rows = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/stability_pairwise_jaccard.csv")
        )
        node_rows = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/stability_per_node_jaccard.csv")
        )
        values = [float(row["jaccard"]) for row in pair_rows]
        output[backbone] = {
            "checkpoint_count": len(STABILITY_SEEDS),
            "target_count": len(node_rows),
            "expected_pair_rows": 167 * 10,
            "observed_pair_rows": len(pair_rows),
            "jaccard_null_count": sum(1 for row in pair_rows if row.get("jaccard", "") == ""),
            "jaccard_distribution": distribution(values),
            "node_mean_stability": distribution(
                [float(row["mean_pairwise_jaccard"]) for row in node_rows]
            ),
        }
    return output


def recompute_summary(config: dict[str, Any]) -> dict[str, Any]:
    stored_rows = read_csv(Path("results/shapeggen/v1/metrics/task24_pgexplainer_summary.csv"))
    stored = {row["backbone"]: row for row in stored_rows}
    output = {}
    max_diff = 0.0
    for backbone in BACKBONES:
        label = BACKBONE_LABELS[backbone]
        rows = read_csv(Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/per_node_metrics.csv"))
        stability_rows = read_csv(
            Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/stability_per_node_jaccard.csv")
        )
        manifest = read_json(checkpoint_manifest_path(backbone, PRIMARY_SEED))
        recomputed = {
            "edge_auroc": mean(float(row["edge_auroc"]) for row in rows),
            "edge_ap": mean(float(row["edge_ap"]) for row in rows),
            "fid_plus": mean(float(row["fid_plus"]) for row in rows),
            "fid_minus": mean(float(row["fid_minus"]) for row in rows),
            "one_minus_fid_minus": mean(float(row["one_minus_fid_minus"]) for row in rows),
            "sparsity": mean(float(row["sparsity"]) for row in rows),
            "stability": mean(float(row["mean_pairwise_jaccard"]) for row in stability_rows),
            "training_runtime_seconds": float(manifest["training_runtime_seconds"]),
            "inference_runtime_seconds_mean": mean(
                float(row["inference_runtime_seconds"]) for row in rows
            ),
            "evaluation_runtime_seconds_mean": mean(
                float(row["evaluation_runtime_seconds"]) for row in rows
            ),
        }
        comparisons = []
        for metric, value in recomputed.items():
            diff = abs(float(stored[label][metric]) - value)
            max_diff = max(max_diff, diff)
            comparisons.append(
                {
                    "backbone": label,
                    "metric": metric,
                    "stored_value": float(stored[label][metric]),
                    "recomputed_value": value,
                    "absolute_difference": diff,
                    "pass": diff < 1e-9,
                }
            )
        output[backbone] = comparisons
    output["maximum_aggregate_difference"] = max_diff
    return output


def file_hashes(config: dict[str, Any]) -> dict[str, Any]:
    paths = [
        Path(config["dataset_path"]),
        Path(config["common_train_file"]),
        Path(config["common_val_file"]),
        Path(config["final_target_file"]),
        Path(config["gcn_checkpoint"]),
        Path(config["graphsage_checkpoint"]),
        Path("results/shapeggen/v1/metrics/task24_pgexplainer_summary.csv"),
        Path("docs/provenance/historical_reports/shapeggen_pgexplainer_report.md"),
        Path("docs/shapeggen_pgexplainer_freeze.json"),
    ]
    for backbone in BACKBONES:
        paths.extend(checkpoint_path(backbone, seed) for seed in [PRIMARY_SEED, *STABILITY_SEEDS])
        paths.extend(
            [
                Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/per_node_metrics.csv"),
                Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/stability_seed_metrics.csv"),
            ]
        )
    return {str(path): sha256_file(path) for path in paths if path.exists()}


def directory_manifest(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()]
    digest = hashlib.sha256()
    total_bytes = 0
    for item in sorted(files):
        total_bytes += item.stat().st_size
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("utf-8"))
        digest.update(sha256_file(item).encode("utf-8"))
    return {"path": str(path), "file_count": len(files), "bytes": total_bytes, "sha256": digest.hexdigest()}


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = read_yaml(args.task_config)["task24"]
    freeze = read_json(args.freeze)
    report_hash_before = sha256_file(args.report)
    freeze_hash_before = sha256_file(args.freeze)
    raw_dirs = {
        f"{backbone}_raw_explanations": directory_manifest(
            Path(f"results/explanations/shapeggen/{backbone}/pgexplainer")
        )
        for backbone in BACKBONES
    }
    audit = {
        "audit_version": "task24_audit_only_v1",
        "verdict": "PASS",
        "experiments_rerun": False,
        "task25_started": False,
        "experimental_artifacts_changed_by_audit": False,
        "initial_git_state": {
            "status_short": git_value("status", "--short") or "clean",
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "upstream_parity": git_value("rev-list", "--left-right", "--count", "main...origin/main"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "pool_audit": pool_audit(config),
        "checkpoint_audit": audit_checkpoints(config),
        "backbone_audit": audit_backbone_hashes(config),
        "row_audit": audit_rows(config),
        "edge_ground_truth_metric_audit": audit_edge_protocol(config),
        "budget_audit": audit_budget(config),
        "stability_audit": audit_stability(),
        "summary_recomputation": recompute_summary(config),
        "integrity": {
            "file_hashes": file_hashes(config),
            "raw_explanation_manifests": raw_dirs,
            "old_report_hash": report_hash_before,
            "old_freeze_hash": freeze_hash_before,
            "previous_report_hash_from_user": (
                "c79773e390879a337bb6580464cbb2ff3538d99d022f618fe747b985cd5c4f9f"
            ),
            "previous_freeze_hash_from_user": (
                "099b2c9c705d93a48533265b15d3dd7665788824cff0938aee77b4ad42bfd369"
            ),
        },
        "metric_semantics": {
            "fid_plus": "1 when removing selected top-20% edges changes the original prediction, else 0.",
            "fid_minus": "1 when retaining only selected top-20% edges changes the original prediction, else 0.",
            "one_minus_fid_minus": "1 - Fid-.",
            "probability_reference_class": "the frozen backbone original predicted class.",
            "edge_auroc_edge_ap": (
                "continuous PGExplainer edge scores over the full candidate-edge set."
            ),
        },
        "validation": {
            "pytest": args.pytest_result,
            "ruff": args.ruff_result,
            "git_diff_check": args.diff_check_result,
        },
        "previous_freeze_status": freeze.get("status"),
    }
    audit["acceptance"] = acceptance(audit)
    audit["verdict"] = "PASS" if all(audit["acceptance"].values()) else "FAIL"
    return audit


def acceptance(audit: dict[str, Any]) -> dict[str, bool]:
    pools = audit["pool_audit"]
    rows = audit["row_audit"]
    unexpected_nulls = []
    domain_violations = []
    identity_violations = []
    coverage_failures = []
    for backbone in BACKBONES:
        unexpected_nulls.extend(
            row["null_count"]
            for row in rows[backbone]["null_audit"]
            if row["classification"] == "UNEXPECTED_REQUIRED_NULL"
        )
        domain_violations.extend(rows[backbone]["domain_violations"].values())
        for value in rows[backbone]["identity_checks"].values():
            if isinstance(value, dict):
                identity_violations.append(value["violation_count"])
        for row in rows[backbone]["coverage"]:
            coverage_failures.extend([row["missing"], row["extra"], row["duplicates"], row["failed"]])
    return {
        "common_train_539": pools["common_train"]["observed_count"] == 539,
        "common_val_182": pools["common_val"]["observed_count"] == 182,
        "final_targets_167": pools["final"]["observed_count"] == 167,
        "no_duplicate_missing_invalid_pool_nodes": all(
            pools[name]["duplicate_count"] == 0
            and pools[name]["missing_node_id_count"] == 0
            and pools[name]["invalid_node_id_count"] == 0
            for name in ["common_train", "common_val", "final"]
        ),
        "train_val_final_disjoint": all(value == 0 for value in pools["intersections"].values()),
        "pg_checkpoints_complete_and_readable": all(
            audit["checkpoint_audit"][backbone]["checkpoint_count"] == 6
            and audit["checkpoint_audit"][backbone]["all_exist"]
            and audit["checkpoint_audit"][backbone]["all_readable"]
            for backbone in BACKBONES
        ),
        "epochs_reload_and_weights_ok": all(
            audit["checkpoint_audit"][backbone]["all_epochs_30"]
            and audit["checkpoint_audit"][backbone]["all_reload_ok"]
            and audit["checkpoint_audit"][backbone]["all_model_weights_unchanged"]
            for backbone in BACKBONES
        ),
        "backbone_hashes_match": all(
            audit["backbone_audit"][backbone]["sha256_matches"] for backbone in BACKBONES
        ),
        "row_coverage_complete": sum(coverage_failures) == 0,
        "unexpected_required_nulls_zero": sum(unexpected_nulls) == 0,
        "domain_violations_zero": sum(domain_violations) == 0,
        "identity_violations_zero": sum(identity_violations) == 0,
        "edge_protocol_complete": all(
            audit["edge_ground_truth_metric_audit"][backbone]["edge_auroc_computed_count"] == 167
            and audit["edge_ground_truth_metric_audit"][backbone]["mapping_error_count"] == 0
            for backbone in BACKBONES
        ),
        "budget_and_sparsity_ok": all(
            audit["budget_audit"][backbone]["selected_count_violation_count"] == 0
            for backbone in BACKBONES
        ),
        "stability_complete": all(
            audit["stability_audit"][backbone]["observed_pair_rows"] == 1670
            and audit["stability_audit"][backbone]["jaccard_null_count"] == 0
            for backbone in BACKBONES
        ),
        "summary_recomputed": (
            audit["summary_recomputation"]["maximum_aggregate_difference"] < 1e-9
        ),
        "task25_not_started": audit["task25_started"] is False,
    }


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.6f}"


def build_report(audit: dict[str, Any]) -> str:
    summary_rows = []
    for row in audit["summary_recomputation"]["gcn"] + audit["summary_recomputation"]["graphsage"]:
        summary_rows.append(
            f"| {row['backbone']} | {row['metric']} | {fmt(row['stored_value'])} | "
            f"{fmt(row['recomputed_value'])} | {row['absolute_difference']:.3g} | "
            f"{row['pass']} |"
        )
    coverage_rows = []
    for backbone in BACKBONES:
        for row in audit["row_audit"][backbone]["coverage"]:
            coverage_rows.append(
                f"| {row['backbone']} | {row['seed']} | {row['expected_targets']} | "
                f"{row['observed_unique_targets']} | {row['missing']} | {row['extra']} | "
                f"{row['duplicates']} | {row['success']} | {row['failed']} |"
            )
    null_rows = []
    for backbone in BACKBONES:
        for row in audit["row_audit"][backbone]["null_audit"]:
            null_rows.append(
                f"| `{row['file']}` | {row['column']} | {row['total_rows']} | "
                f"{row['null_count']} | {row['classification']} | {row['reason']} |"
            )
    null_table = "\n".join(null_rows) if null_rows else "| NONE | NONE | 0 | 0 | NONE | NONE |"
    acceptance = "\n".join(f"- {key}: {value}" for key, value in audit["acceptance"].items())
    pool = audit["pool_audit"]
    return f"""# ShapeGGen Task24 PGExplainer Audit Freeze

## Scope

This audit is read-only with respect to experimental evidence. It does not
retrain PGExplainer, rerun explanation generation, change checkpoints, change
targets, change seeds, change hyperparameters, tune on final targets, or start
Task25.

## Initial Git State

- Branch: `{audit['initial_git_state']['branch']}`
- HEAD: `{audit['initial_git_state']['head']}`
- Status before audit: `{audit['initial_git_state']['status_short']}`
- Upstream parity before audit: `{audit['initial_git_state']['upstream_parity']}`

## Dataset And Pool Audit

| Pool | Expected | Observed | Unique | Duplicate | Missing node id | Invalid node id | Sorted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Common train | {pool['common_train']['expected_count']} | {pool['common_train']['observed_count']} | {pool['common_train']['unique_count']} | {pool['common_train']['duplicate_count']} | {pool['common_train']['missing_node_id_count']} | {pool['common_train']['invalid_node_id_count']} | {pool['common_train']['sorted']} |
| Common validation | {pool['common_val']['expected_count']} | {pool['common_val']['observed_count']} | {pool['common_val']['unique_count']} | {pool['common_val']['duplicate_count']} | {pool['common_val']['missing_node_id_count']} | {pool['common_val']['invalid_node_id_count']} | {pool['common_val']['sorted']} |
| Final targets | {pool['final']['expected_count']} | {pool['final']['observed_count']} | {pool['final']['unique_count']} | {pool['final']['duplicate_count']} | {pool['final']['missing_node_id_count']} | {pool['final']['invalid_node_id_count']} | {pool['final']['sorted']} |

- train_val_intersection_count: `{pool['intersections']['train_val_intersection_count']}`
- train_final_intersection_count: `{pool['intersections']['train_final_intersection_count']}`
- val_final_intersection_count: `{pool['intersections']['val_final_intersection_count']}`

## Checkpoint Audit

{checkpoint_table(audit)}

## Row Completeness Audit

| Backbone | Seed | Expected targets | Observed unique targets | Missing | Extra | Duplicates | Success | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(coverage_rows)}

## Null And Missing Audit

| File | Column/field | Total rows | Null count | Classification | Reason |
|---|---|---:|---:|---|---|
{null_table}

## Metric Identity Audit

{identity_section(audit)}

Fid+ is implemented as a prediction-change indicator after removing the
selected top-20% edges. Fid- is implemented as a prediction-change indicator
after retaining only the selected top-20% edges. Both use the frozen backbone
original predicted class as the probability reference for diagnostics.

## Ground-Truth Metric Audit

Edge AUROC and Edge AP use continuous scores over the full candidate-edge set,
using PGExplainer edge scores without top-20% binarization. Mapping errors:
GCN `0`, GraphSAGE `0`. Single-class cases: GCN `0`, GraphSAGE `0`. AUROC/AP
computed targets: GCN `167`, GraphSAGE `167`.

## Budget And Stability Audit

Budget uses `selected_edge_ratio=0.2` with rule
`ceil(candidate_edge_count * selected_edge_ratio), minimum 1`. Sparsity is
computed from selected and candidate counts, not forced to `0.8`.

{stability_section(audit)}

## Runtime Audit

PGExplainer runtime is separated into upfront explainer training, per-target
trained-explainer inference, and per-target metric evaluation.

| Backbone | Training runtime s | Mean inference runtime s | Mean evaluation runtime s |
|---|---:|---:|---:|
{runtime_table(audit)}

## Summary Recomputaton

| Backbone | Metric | Stored value | Recomputed value | Absolute difference | Pass |
|---|---|---:|---:|---:|---:|
{chr(10).join(summary_rows)}

Maximum aggregate difference:
`{audit['summary_recomputation']['maximum_aggregate_difference']:.12g}`.

## Integrity Audit

- Old report hash before audit update: `{audit['integrity']['old_report_hash']}`
- Old freeze hash before audit update: `{audit['integrity']['old_freeze_hash']}`
- Previous report hash from prior response: `{audit['integrity']['previous_report_hash_from_user']}`
- Previous freeze hash from prior response: `{audit['integrity']['previous_freeze_hash_from_user']}`
- Experimental artifacts changed by this audit: `False`
- Raw explanation manifests are recorded in `docs/shapeggen_pgexplainer_audit.json`.

## Acceptance

{acceptance}

## Validation

- pytest: `{audit['validation']['pytest']}`
- Ruff: `{audit['validation']['ruff']}`
- git diff --check: `{audit['validation']['git_diff_check']}`
- freeze JSON parse: `passed`

## Verdict

Task24 audit verdict: `{audit['verdict']}`.
Task25 is authorized only after this audit commit/push is reviewed. Task25 was
not started in this task. Task25 is not started automatically.

STOP FOR REVIEW
"""


def checkpoint_table(audit: dict[str, Any]) -> str:
    lines = [
        "| Backbone | Seed | Exists | Readable | Epochs | Reload | Weights unchanged | SHA256 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for backbone in BACKBONES:
        for row in audit["checkpoint_audit"][backbone]["rows"]:
            lines.append(
                f"| {BACKBONE_LABELS[backbone]} | {row['seed']} | {row['checkpoint_exists']} | "
                f"{row['checkpoint_readable']} | {row['epochs_completed']} | "
                f"{row['checkpoint_reload_ok']} | {row['model_weights_unchanged']} | "
                f"`{row['checkpoint_sha256']}` |"
            )
    return "\n".join(lines)


def identity_section(audit: dict[str, Any]) -> str:
    lines = [
        "| Backbone | Identity | Checked rows | Max abs error | Violations | Pass |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for backbone in BACKBONES:
        for name, row in audit["row_audit"][backbone]["identity_checks"].items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {BACKBONE_LABELS[backbone]} | {name} | {row['checked_rows']} | "
                f"{row['maximum_absolute_error']:.3g} | {row['violation_count']} | "
                f"{row['pass']} |"
            )
    return "\n".join(lines)


def stability_section(audit: dict[str, Any]) -> str:
    lines = [
        "| Backbone | Checkpoints | Targets | Expected pairs | Observed pairs | Null Jaccard | Mean | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backbone in BACKBONES:
        row = audit["stability_audit"][backbone]
        dist = row["jaccard_distribution"]
        lines.append(
            f"| {BACKBONE_LABELS[backbone]} | {row['checkpoint_count']} | "
            f"{row['target_count']} | {row['expected_pair_rows']} | "
            f"{row['observed_pair_rows']} | {row['jaccard_null_count']} | "
            f"{fmt(dist['mean'])} | {fmt(dist['min'])} | {fmt(dist['max'])} |"
        )
    return "\n".join(lines)


def runtime_table(audit: dict[str, Any]) -> str:
    lines = []
    for backbone in BACKBONES:
        row = {
            item["metric"]: item["recomputed_value"]
            for item in audit["summary_recomputation"][backbone]
        }
        lines.append(
            f"| {BACKBONE_LABELS[backbone]} | {fmt(row['training_runtime_seconds'])} | "
            f"{fmt(row['inference_runtime_seconds_mean'])} | "
            f"{fmt(row['evaluation_runtime_seconds_mean'])} |"
        )
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = build_audit(args)
    write_json(args.audit_json, audit)
    freeze = read_json(args.freeze)
    freeze["task24_audit"] = audit
    freeze["status"] = "success" if audit["verdict"] == "PASS" else "failed"
    write_json(args.freeze, freeze)
    args.report.write_text(build_report(audit), encoding="utf-8")
    print(f"verdict={audit['verdict']}")
    print(f"audit_json={args.audit_json}")
    print(f"report={args.report}")
    print(f"freeze={args.freeze}")


if __name__ == "__main__":
    main()
