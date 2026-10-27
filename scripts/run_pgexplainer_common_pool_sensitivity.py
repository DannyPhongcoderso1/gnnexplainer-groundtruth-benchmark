from __future__ import annotations

import csv
import json
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

from gnnexplainer_benchmark.data.serialization import sha256_file
from gnnexplainer_benchmark.explainers.pgexplainer_evaluation import evaluate_pgexplainer_gcn
from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    config_hash,
    load_yaml,
    train_pgexplainer_gcn,
)

COMMON_TRAIN_PATH = Path("configs/ba_house/ba500/targets/ba_house_pg_common_train_targets.csv")
COMMON_VAL_PATH = Path("configs/ba_house/ba500/targets/ba_house_pg_common_val_targets.csv")
FINAL_TARGETS_PATH = Path("configs/ba_house/ba500/targets/ba_house_2x2_targets.csv")
GCN_TRAIN_PATH = Path("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv")
GCN_VAL_PATH = Path("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv")
SAGE_TRAIN_PATH = Path("configs/ba_house/ba500/targets/pgexplainer_graphsage_train_targets.csv")
SAGE_VAL_PATH = Path("configs/ba_house/ba500/targets/pgexplainer_graphsage_val_targets.csv")
COMMON_CONFIG_PATH = Path("configs/ba_house/ba500/pgexplainer_graphsage_common_pool.yaml")
RESULT_ROOT = Path("results/metrics/pgexplainer_common_pool_sensitivity")
GRAPH_METRICS_ROOT = Path("results/metrics/graphsage_pgexplainer_common_pool")
GRAPH_EXPLANATIONS_ROOT = Path("results/explanations/graphsage_pgexplainer_common_pool")
STABILITY_SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    manifest = run_task19b(device=default_device())
    write_freeze_files(manifest)
    common_summary = manifest["graphsage_common_summary"]
    print(
        "Task 19B common-pool sensitivity complete | "
        f"GraphSAGE common AUROC={metric_mean(common_summary, 'edge_auroc'):.6f} | "
        f"AP={metric_mean(common_summary, 'edge_ap'):.6f} | "
        f"stability={manifest['graphsage_common_stability']['mean']:.6f}"
    )


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_task19b(device: str) -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    base_commit = git("rev-parse", "HEAD")
    pool = build_common_pool()
    write_common_pool_targets(pool)
    common_freeze = build_common_pool_freeze(pool, base_commit)
    write_json(Path("docs/ba_house_pg_common_pool_freeze.json"), common_freeze)
    Path("docs/provenance/historical_reports/ba_house_pg_common_pool_freeze.md").write_text(
        build_common_pool_markdown(common_freeze),
        encoding="utf-8",
    )

    verify_gcn_common_reuse(pool)
    verify_graphsage_checkpoint()
    verify_protocol_parity()

    seed42_training = train_graphsage_common(seed=42, device=device)
    seed42_evaluation = evaluate_graphsage_common_seed(
        seed=42,
        device=device,
        metrics_dir=GRAPH_METRICS_ROOT / "seed42",
        explanations_dir=GRAPH_EXPLANATIONS_ROOT / "seed42",
    )
    copy_quality_outputs()

    stability_manifest = train_and_evaluate_stability(device=device)
    comparisons = write_comparison_tables(stability_manifest)
    manifest = build_task19b_manifest(
        base_commit=base_commit,
        device=device,
        common_freeze=common_freeze,
        seed42_training=seed42_training,
        seed42_evaluation=seed42_evaluation,
        stability_manifest=stability_manifest,
        comparisons=comparisons,
    )
    write_json(RESULT_ROOT / "manifest.json", manifest)
    write_json(Path("docs/pgexplainer_common_pool_sensitivity_freeze.json"), manifest)
    Path("docs/provenance/historical_reports/pgexplainer_common_pool_sensitivity_report.md").write_text(
        build_sensitivity_report(manifest),
        encoding="utf-8",
    )
    return manifest


def build_common_pool() -> dict[str, Any]:
    gcn_train = read_csv(GCN_TRAIN_PATH)
    gcn_val = read_csv(GCN_VAL_PATH)
    sage_train = read_csv(SAGE_TRAIN_PATH)
    sage_val = read_csv(SAGE_VAL_PATH)
    final_rows = read_csv(FINAL_TARGETS_PATH)
    gcn_train_ids = node_ids(gcn_train)
    gcn_val_ids = node_ids(gcn_val)
    sage_train_ids = node_ids(sage_train)
    sage_val_ids = node_ids(sage_val)
    final_ids = node_ids(final_rows)
    common_train_ids = sorted(gcn_train_ids & sage_train_ids)
    common_val_ids = sorted(gcn_val_ids & sage_val_ids)
    invariants = {
        "common_train_count_is_48": len(common_train_ids) == 48,
        "common_val_count_is_16": len(common_val_ids) == 16,
        "gcn_train_minus_graphsage_train_empty": not (gcn_train_ids - sage_train_ids),
        "gcn_val_minus_graphsage_val_empty": not (gcn_val_ids - sage_val_ids),
        "common_train_val_disjoint": not (set(common_train_ids) & set(common_val_ids)),
        "common_train_final_disjoint": not (set(common_train_ids) & final_ids),
        "common_val_final_disjoint": not (set(common_val_ids) & final_ids),
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise RuntimeError(f"Common-pool invariant failure: {failed}")
    return {
        "gcn_train_rows": gcn_train,
        "gcn_val_rows": gcn_val,
        "sage_train_rows": sage_train,
        "sage_val_rows": sage_val,
        "final_rows": final_rows,
        "common_train_rows": rows_for_ids(gcn_train, common_train_ids),
        "common_val_rows": rows_for_ids(gcn_val, common_val_ids),
        "counts": {
            "gcn_train": len(gcn_train_ids),
            "graphsage_train": len(sage_train_ids),
            "common_train": len(common_train_ids),
            "gcn_val": len(gcn_val_ids),
            "graphsage_val": len(sage_val_ids),
            "common_val": len(common_val_ids),
            "final": len(final_ids),
            "graphsage_only_train": len(sage_train_ids - gcn_train_ids),
            "graphsage_only_val": len(sage_val_ids - gcn_val_ids),
        },
        "invariants": invariants,
    }


def write_common_pool_targets(pool: dict[str, Any]) -> None:
    write_csv(COMMON_TRAIN_PATH, pool["common_train_rows"])
    write_csv(COMMON_VAL_PATH, pool["common_val_rows"])


def build_common_pool_freeze(pool: dict[str, Any], base_commit: str) -> dict[str, Any]:
    return {
        "status": "success",
        "methodological_status": "post-hoc controlled sensitivity analysis",
        "intersection_rule": (
            "COMMON pool is the symmetric intersection of pre-existing "
            "backbone-specific eligible PGExplainer pools using node_id."
        ),
        "gcn_pool_equality_note": (
            "The fact that the common pool equals the original GCN pool is an "
            "empirical consequence of set inclusion and was not used as a "
            "manual selection rule."
        ),
        "creation_commit_base": base_commit,
        "source_files": {
            "gcn_train": file_record(GCN_TRAIN_PATH),
            "graphsage_train": file_record(SAGE_TRAIN_PATH),
            "gcn_val": file_record(GCN_VAL_PATH),
            "graphsage_val": file_record(SAGE_VAL_PATH),
            "final_targets": file_record(FINAL_TARGETS_PATH),
        },
        "common_files": {
            "train": file_record(COMMON_TRAIN_PATH),
            "validation": file_record(COMMON_VAL_PATH),
        },
        "counts": pool["counts"],
        "disjointness": pool["invariants"],
    }


def verify_gcn_common_reuse(pool: dict[str, Any]) -> None:
    common_train_ids = node_ids(pool["common_train_rows"])
    common_val_ids = node_ids(pool["common_val_rows"])
    if node_ids(pool["gcn_train_rows"]) != common_train_ids:
        raise RuntimeError("Existing GCN PG train identities do not equal common train")
    if node_ids(pool["gcn_val_rows"]) != common_val_ids:
        raise RuntimeError("Existing GCN PG validation identities do not equal common val")
    manifest = read_json(Path("artifacts/explainers/pgexplainer_gcn_ba_house_seed42_manifest.json"))
    hyper = manifest["pgexplainer_hyperparameters"]
    expected = {
        "epochs": 30,
        "learning_rate": 0.003,
        "edge_size": 0.05,
        "edge_ent": 1.0,
        "temperature_start": 5.0,
        "temperature_end": 2.0,
        "bias": 0.01,
    }
    for key, value in expected.items():
        if float(hyper[key]) != float(value):
            raise RuntimeError(f"GCN PG hyperparameter mismatch for {key}: {hyper[key]}")
    if int(manifest["seed"]) != 42:
        raise RuntimeError("GCN seed42 manifest is not seed 42")


def verify_graphsage_checkpoint() -> None:
    dataset_hash = sha256_file(Path("data/ba_house/ba500/graph.pt"))
    if dataset_hash != "4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94":
        raise RuntimeError(f"Unexpected BA-House dataset SHA256: {dataset_hash}")
    task19_hash = read_json(Path("docs/graphsage_explainers_ba_house_freeze.json"))[
        "graphsage_checkpoint_sha256"
    ]
    current_hash = sha256_file(Path("artifacts/checkpoints/ba_house/ba500/backbones/graphsage/seed42.pt"))
    if current_hash != task19_hash:
        raise RuntimeError("GraphSAGE checkpoint SHA256 does not match frozen manifest")


def verify_protocol_parity() -> None:
    gcn_manifest = read_json(
        Path("artifacts/explainers/pgexplainer_gcn_ba_house_seed42_manifest.json")
    )
    sage_manifest = read_json(
        Path("artifacts/explainers/pgexplainer_graphsage_ba_house_seed42_manifest.json")
    )
    common_config = load_yaml(COMMON_CONFIG_PATH)
    sage_config = load_yaml(Path("configs/ba_house/ba500/pgexplainer_graphsage.yaml"))
    if gcn_manifest["pgexplainer_hyperparameters"] != sage_manifest["pgexplainer_hyperparameters"]:
        raise RuntimeError("GCN and GraphSAGE frozen PG hyperparameters differ")
    keys = [
        "candidate_edge_scope",
        "edge_ranking_metric_scope",
        "budget_protocol",
        "selected_edge_ratio",
        "rounding_rule",
    ]
    for key in keys:
        if common_config["evaluation"][key] != sage_config["evaluation"][key]:
            raise RuntimeError(f"Common config changes frozen evaluation key {key}")


def train_graphsage_common(seed: int, device: str) -> dict[str, Any]:
    metrics_root = GRAPH_METRICS_ROOT / ("seed42" if seed == 42 else "stability")
    artifact_path = Path(
        f"artifacts/explainers/pgexplainer_graphsage_ba_house_common_seed{seed}.pt"
    )
    manifest_path = Path(
        f"artifacts/explainers/pgexplainer_graphsage_ba_house_common_seed{seed}_manifest.json"
    )
    result = train_pgexplainer_gcn(
        checkpoint_path=Path("artifacts/checkpoints/ba_house/ba500/backbones/graphsage/seed42.pt"),
        evaluation_config_path=COMMON_CONFIG_PATH,
        train_targets_path=COMMON_TRAIN_PATH,
        validation_targets_path=COMMON_VAL_PATH,
        final_test_targets_path=FINAL_TARGETS_PATH,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        training_log_path=metrics_root / f"seed_{seed}_training_log.csv",
        validation_health_path=metrics_root / f"seed_{seed}_validation_health.csv",
        validation_summary_path=metrics_root / f"seed_{seed}_validation_summary.json",
        device=device,
        seed_override=seed,
    )
    manifest = read_json(manifest_path)
    if int(result.epochs_completed) != 30:
        raise RuntimeError(f"Seed {seed} did not complete 30 epochs")
    if not result.model_parameters_unchanged:
        raise RuntimeError(f"GraphSAGE parameters changed during seed {seed} training")
    if not manifest["optimizer_excludes_gcn_parameters"]:
        raise RuntimeError(f"Optimizer includes GraphSAGE params for seed {seed}")
    if int(manifest["final_test_targets_used_for_training"]) != 0:
        raise RuntimeError(f"Final targets used during seed {seed} training")
    if not result.checkpoint_reload_ok:
        raise RuntimeError(f"PGExplainer checkpoint reload failed for seed {seed}")
    return {
        "seed": seed,
        "artifact_path": str(artifact_path),
        "manifest_path": str(manifest_path),
        "checkpoint_sha256": sha256_file(artifact_path),
        "epochs_completed": result.epochs_completed,
        "final_loss": result.final_loss,
        "training_runtime_seconds": result.training_runtime_seconds,
        "model_parameters_unchanged": result.model_parameters_unchanged,
        "optimizer_excludes_graphsage_parameters": manifest["optimizer_excludes_gcn_parameters"],
        "checkpoint_reload_ok": result.checkpoint_reload_ok,
        "validation_success_count": result.validation_success_count,
        "validation_failure_count": result.validation_failure_count,
    }


def evaluate_graphsage_common_seed(
    *,
    seed: int,
    device: str,
    metrics_dir: Path,
    explanations_dir: Path,
) -> dict[str, Any]:
    artifact_path = Path(
        f"artifacts/explainers/pgexplainer_graphsage_ba_house_common_seed{seed}.pt"
    )
    manifest_path = Path(
        f"artifacts/explainers/pgexplainer_graphsage_ba_house_common_seed{seed}_manifest.json"
    )
    started_at = perf_counter()
    result = evaluate_pgexplainer_gcn(
        checkpoint_path=Path("artifacts/checkpoints/ba_house/ba500/backbones/graphsage/seed42.pt"),
        explainer_checkpoint_path=artifact_path,
        explainer_manifest_path=manifest_path,
        evaluation_config_path=COMMON_CONFIG_PATH,
        target_file=FINAL_TARGETS_PATH,
        phase4_reference_per_node_path=Path(
            "results/metrics/graphsage_gnnexplainer/seed42/per_node_metrics.csv"
        ),
        output_root=explanations_dir,
        metrics_output_dir=metrics_dir,
        device=device,
    )
    evaluation_runtime_seconds = perf_counter() - started_at
    manifest = read_json(result.manifest_path)
    checks = {
        "success_12_of_12": result.success_count == 12 and result.failure_count == 0,
        "candidate_parity_12_of_12": bool(manifest["candidate_counts_match_phase4"]),
        "gt_parity_12_of_12": bool(manifest["ground_truth_counts_match_phase4"]),
        "selected_count_parity_12_of_12": bool(manifest["selected_counts_match_phase4"]),
        "prediction_parity_12_of_12": bool(manifest["original_predictions_match_phase4"]),
        "model_weights_unchanged": result.model_weights_unchanged,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"GraphSAGE common seed {seed} evaluation failed checks: {failed}")
    manifest["evaluation_runtime_seconds"] = evaluation_runtime_seconds
    write_json(result.manifest_path, manifest)
    return {
        "seed": seed,
        "per_node_metrics_path": str(result.per_node_metrics_path),
        "summary_metrics_path": str(result.summary_metrics_path),
        "manifest_path": str(result.manifest_path),
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "evaluation_runtime_seconds": evaluation_runtime_seconds,
        "checks": checks,
    }


def copy_quality_outputs() -> None:
    copies = {
        GRAPH_METRICS_ROOT / "seed42" / "per_node_metrics.csv": (
            RESULT_ROOT / "graphsage_common_seed42_per_node.csv"
        ),
        GRAPH_METRICS_ROOT / "seed42" / "summary_metrics.csv": (
            RESULT_ROOT / "graphsage_common_seed42_summary.csv"
        ),
    }
    for source, destination in copies.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def train_and_evaluate_stability(device: str) -> dict[str, Any]:
    training_rows = []
    evaluation_rows = []
    inference_rows = []
    per_seed_rows = []
    metrics_root = GRAPH_METRICS_ROOT / "stability"
    explanations_root = GRAPH_EXPLANATIONS_ROOT / "stability"
    for seed in STABILITY_SEEDS:
        training = train_graphsage_common(seed=seed, device=device)
        training_rows.append(training)
        evaluation = evaluate_graphsage_common_seed(
            seed=seed,
            device=device,
            metrics_dir=metrics_root / f"seed_{seed}",
            explanations_dir=explanations_root / f"seed_{seed}",
        )
        evaluation_rows.append(evaluation)
        rows = read_csv(Path(evaluation["per_node_metrics_path"]))
        for row in rows:
            row["seed"] = str(seed)
            per_seed_rows.append(row)
            inference_rows.append(
                {
                    "seed": seed,
                    "node_id": row["node_id"],
                    "inference_runtime_seconds": row["inference_runtime_seconds"],
                }
            )
    write_csv(metrics_root / "training_runtime.csv", training_rows)
    write_csv(metrics_root / "evaluation_runtime.csv", evaluation_rows)
    write_csv(metrics_root / "inference_runtime.csv", inference_rows)
    write_csv(metrics_root / "per_seed_metrics.csv", per_seed_rows)
    pairwise_rows, per_node_rows, summary = compute_jaccard(explanations_root, STABILITY_SEEDS)
    write_csv(metrics_root / "pairwise_jaccard.csv", pairwise_rows)
    write_csv(metrics_root / "per_node_jaccard.csv", per_node_rows)
    write_json(metrics_root / "stability_summary.json", summary)
    copy_stability_outputs(metrics_root)
    manifest = {
        "status": "success",
        "seeds": STABILITY_SEEDS,
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        "training_success_count": len(training_rows),
        "evaluation_success_count": sum(row["success_count"] == 12 for row in evaluation_rows),
        "total_explained_targets": sum(row["success_count"] for row in evaluation_rows),
        "jaccard_summary": summary,
        "runtime": {
            "training_runtime_seconds": summarize(
                float(row["training_runtime_seconds"]) for row in training_rows
            ),
            "inference_runtime_seconds": summarize(
                float(row["inference_runtime_seconds"]) for row in inference_rows
            ),
            "evaluation_runtime_seconds": summarize(
                float(row["evaluation_runtime_seconds"]) for row in evaluation_rows
            ),
        },
        "outputs": {
            "pairwise_jaccard": str(metrics_root / "pairwise_jaccard.csv"),
            "per_node_jaccard": str(metrics_root / "per_node_jaccard.csv"),
            "stability_summary": str(metrics_root / "stability_summary.json"),
        },
    }
    write_json(metrics_root / "manifest.json", manifest)
    write_json(RESULT_ROOT / "graphsage_common_stability_summary.json", summary)
    return manifest


def copy_stability_outputs(metrics_root: Path) -> None:
    copies = {
        metrics_root / "pairwise_jaccard.csv": RESULT_ROOT / "graphsage_common_stability.csv",
        metrics_root / "stability_summary.json": (
            RESULT_ROOT / "graphsage_common_stability_summary.json"
        ),
    }
    for source, destination in copies.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def compute_jaccard(
    explanations_root: Path,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    node_ids = [
        int(path.name.removeprefix("node_"))
        for path in sorted((explanations_root / f"seed_{seeds[0]}").glob("node_*"))
    ]
    pairwise_rows = []
    per_node_rows = []
    for node_id in node_ids:
        seed_sets = {
            seed: selected_edges(
                explanations_root / f"seed_{seed}" / f"node_{node_id}" / "selected_edges.csv"
            )
            for seed in seeds
        }
        node_values = []
        for seed_a, seed_b in combinations(seeds, 2):
            edges_a = seed_sets[seed_a]
            edges_b = seed_sets[seed_b]
            union = edges_a | edges_b
            jaccard = len(edges_a & edges_b) / len(union) if union else 1.0
            node_values.append(jaccard)
            pairwise_rows.append(
                {
                    "node_id": node_id,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "selected_count_a": len(edges_a),
                    "selected_count_b": len(edges_b),
                    "intersection_count": len(edges_a & edges_b),
                    "union_count": len(union),
                    "jaccard": jaccard,
                }
            )
        per_node_rows.append(
            {
                "node_id": node_id,
                "pair_count": len(node_values),
                "mean_jaccard": mean(node_values),
                "std_jaccard": stdev(node_values) if len(node_values) > 1 else 0.0,
                "median_jaccard": median(node_values),
                "min_jaccard": min(node_values),
                "max_jaccard": max(node_values),
            }
        )
    node_means = [float(row["mean_jaccard"]) for row in per_node_rows]
    return (
        pairwise_rows,
        per_node_rows,
        {
            "node_count": len(per_node_rows),
            "seed_count": len(seeds),
            "pairwise_comparisons_per_node": 10,
            "pairwise_row_count": len(pairwise_rows),
            "mean": mean(node_means),
            "std": stdev(node_means) if len(node_means) > 1 else 0.0,
            "median": median(node_means),
            "min": min(node_means),
            "max": max(node_means),
        },
    )


def write_comparison_tables(stability_manifest: dict[str, Any]) -> dict[str, Any]:
    natural_summary = read_summary(
        Path("results/metrics/graphsage_pgexplainer/seed42/summary_metrics.csv")
    )
    common_summary = read_summary(GRAPH_METRICS_ROOT / "seed42" / "summary_metrics.csv")
    gcn_summary = read_summary(Path("results/metrics/pgexplainer_gcn_seed42/summary_metrics.csv"))
    natural_stability = read_json(
        Path("results/metrics/graphsage_pgexplainer/stability/stability_summary.json")
    )
    gcn_stability = read_json(
        Path("results/metrics/pgexplainer_gcn_stability/stability_summary.json")
    )
    common_stability = stability_manifest["jaccard_summary"]
    natural_vs_common_rows = []
    for metric in comparison_metrics():
        natural_value = comparison_value(natural_summary, natural_stability, metric)
        common_value = comparison_value(common_summary, common_stability, metric)
        natural_vs_common_rows.append(
            {
                "metric": metric,
                "natural_106_35": natural_value,
                "common_48_16": common_value,
                "delta_common_minus_natural": common_value - natural_value,
            }
        )
    cross_rows = []
    for metric in comparison_metrics():
        gcn_value = comparison_value(gcn_summary, gcn_stability, metric)
        sage_value = comparison_value(common_summary, common_stability, metric)
        cross_rows.append(
            {
                "metric": metric,
                "pg_gcn_common": gcn_value,
                "pg_graphsage_common": sage_value,
                "delta_graphsage_minus_gcn": sage_value - gcn_value,
            }
        )
    write_csv(RESULT_ROOT / "natural_vs_common_graphsage.csv", natural_vs_common_rows)
    write_csv(RESULT_ROOT / "controlled_cross_backbone_pg.csv", cross_rows)
    return {
        "natural_vs_common_graphsage": natural_vs_common_rows,
        "controlled_cross_backbone_pg": cross_rows,
    }


def build_task19b_manifest(
    *,
    base_commit: str,
    device: str,
    common_freeze: dict[str, Any],
    seed42_training: dict[str, Any],
    seed42_evaluation: dict[str, Any],
    stability_manifest: dict[str, Any],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    common_summary = read_summary(GRAPH_METRICS_ROOT / "seed42" / "summary_metrics.csv")
    gcn_manifest = read_json(
        Path("artifacts/explainers/pgexplainer_gcn_ba_house_seed42_manifest.json")
    )
    sage_common_manifest = read_json(
        Path("artifacts/explainers/pgexplainer_graphsage_ba_house_common_seed42_manifest.json")
    )
    result_files = {
        "graphsage_common_seed42_per_node": RESULT_ROOT / "graphsage_common_seed42_per_node.csv",
        "graphsage_common_seed42_summary": RESULT_ROOT / "graphsage_common_seed42_summary.csv",
        "graphsage_common_stability": RESULT_ROOT / "graphsage_common_stability.csv",
        "graphsage_common_stability_summary": (
            RESULT_ROOT / "graphsage_common_stability_summary.json"
        ),
        "natural_vs_common_graphsage": RESULT_ROOT / "natural_vs_common_graphsage.csv",
        "controlled_cross_backbone_pg": RESULT_ROOT / "controlled_cross_backbone_pg.csv",
    }
    return {
        "status": "success",
        "task": "Task 19B controlled common-pool PGExplainer sensitivity analysis",
        "methodological_status": (
            "post-hoc controlled sensitivity analysis conducted after primary "
            "final-test exposure"
        ),
        "base_commit_before_task19b": base_commit,
        "git_commit_at_manifest_build": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "device": device,
        "dataset": file_record(Path("data/ba_house/ba500/graph.pt")),
        "checkpoints": {
            "gcn": file_record(Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt")),
            "graphsage": file_record(Path("artifacts/checkpoints/ba_house/ba500/backbones/graphsage/seed42.pt")),
            "gcn_pg_seed42": file_record(
                Path("artifacts/checkpoints/ba_house/ba500/pgexplainer/gcn/primary_seed42.pt")
            ),
            "graphsage_pg_natural_seed42": file_record(
                Path("artifacts/checkpoints/ba_house/ba500/pgexplainer/graphsage/natural106_35/primary_seed42.pt")
            ),
            "graphsage_pg_common_seed42": file_record(
                Path("artifacts/checkpoints/ba_house/ba500/pgexplainer/graphsage/common48_16/primary_seed42.pt")
            ),
        },
        "source_hashes": {
            "pg_training": file_record(
                Path("src/gnnexplainer_benchmark/explainers/pgexplainer_training.py")
            ),
            "pg_evaluation": file_record(
                Path("src/gnnexplainer_benchmark/explainers/pgexplainer_evaluation.py")
            ),
            "task19b_script": file_record(
                Path("scripts/run_pgexplainer_common_pool_sensitivity.py")
            ),
            "explainer_config": file_record(Path("configs/shared/pgexplainer.yaml")),
            "common_evaluation_config": file_record(COMMON_CONFIG_PATH),
        },
        "pgexplainer_config_hash": config_hash(
            load_yaml(Path("configs/shared/pgexplainer.yaml"))
        ),
        "common_pool": common_freeze,
        "gcn_common_reuse": {
            "existing_gcn_train_identical_to_common": True,
            "existing_gcn_val_identical_to_common": True,
            "gcn_retrained": False,
            "gcn_pg_checkpoint_sha256": gcn_manifest["pgexplainer_checkpoint_sha256"],
        },
        "graphsage_common_seed42_training": seed42_training,
        "graphsage_common_seed42_evaluation": seed42_evaluation,
        "graphsage_common_summary": common_summary,
        "graphsage_common_stability": stability_manifest["jaccard_summary"],
        "graphsage_common_stability_manifest": stability_manifest,
        "comparisons": comparisons,
        "runtime": {
            "seed42_training_runtime_seconds": seed42_training["training_runtime_seconds"],
            "seed42_evaluation_runtime_seconds": seed42_evaluation[
                "evaluation_runtime_seconds"
            ],
            "stability_runtime": stability_manifest["runtime"],
        },
        "integrity": {
            "original_task19_artifacts_modified": False,
            "retraining_of_gcn_pg": False,
            "hyperparameter_tuning": False,
            "best_seed_selection": False,
            "target_modification": False,
            "graphsage_common_checkpoint_differs_from_natural": (
                sage_common_manifest["pgexplainer_checkpoint_sha256"]
                != sha256_file(
                    Path("artifacts/checkpoints/ba_house/ba500/pgexplainer/graphsage/natural106_35/primary_seed42.pt")
                )
            ),
            "final_targets_used_for_training": seed42_training.get(
                "final_test_targets_used_for_training",
                0,
            ),
        },
        "result_file_hashes": {name: file_record(path) for name, path in result_files.items()},
    }


def write_freeze_files(manifest: dict[str, Any]) -> None:
    write_json(RESULT_ROOT / "manifest.json", manifest)
    write_json(Path("docs/pgexplainer_common_pool_sensitivity_freeze.json"), manifest)
    Path("docs/provenance/historical_reports/pgexplainer_common_pool_sensitivity_report.md").write_text(
        build_sensitivity_report(manifest),
        encoding="utf-8",
    )


def build_common_pool_markdown(freeze: dict[str, Any]) -> str:
    counts = freeze["counts"]
    return f"""# BA-House PGExplainer Common Pool Freeze

## Methodological Status

This pool is for a post-hoc controlled sensitivity analysis. It does not
replace the original frozen Task 19 primary results.

## Construction

The common pool was defined by the symmetric intersection of the pre-existing
backbone-specific eligible pools using `node_id`.

The fact that the common pool equals the original GCN pool is an empirical
consequence of set inclusion and was not used as a manual selection rule.

## Counts

- GCN train: `{counts['gcn_train']}`
- GraphSAGE train: `{counts['graphsage_train']}`
- Common train: `{counts['common_train']}`
- GCN validation: `{counts['gcn_val']}`
- GraphSAGE validation: `{counts['graphsage_val']}`
- Common validation: `{counts['common_val']}`
- Final targets: `{counts['final']}`

## Files

- Common train: `{freeze['common_files']['train']['path']}`
- Common train SHA256: `{freeze['common_files']['train']['sha256']}`
- Common validation: `{freeze['common_files']['validation']['path']}`
- Common validation SHA256: `{freeze['common_files']['validation']['sha256']}`
- Final targets: `{freeze['source_files']['final_targets']['path']}`
- Final targets SHA256: `{freeze['source_files']['final_targets']['sha256']}`
"""


def build_sensitivity_report(manifest: dict[str, Any]) -> str:
    common = manifest["common_pool"]
    train = manifest["graphsage_common_seed42_training"]
    eval_result = manifest["graphsage_common_seed42_evaluation"]
    summary = manifest["graphsage_common_summary"]
    stability = manifest["graphsage_common_stability"]
    natural_table = markdown_table(
        manifest["comparisons"]["natural_vs_common_graphsage"],
        ["metric", "natural_106_35", "common_48_16", "delta_common_minus_natural"],
    )
    cross_table = markdown_table(
        manifest["comparisons"]["controlled_cross_backbone_pg"],
        ["metric", "pg_gcn_common", "pg_graphsage_common", "delta_graphsage_minus_gcn"],
    )
    return f"""# PGExplainer Common-Pool Sensitivity Report

## 1. Motivation

Task 19 found a very strong GraphSAGE PGExplainer result using the natural
GraphSAGE eligible pool. Task 19B asks whether that result persists when the
PGExplainer training and validation node identities are controlled across GCN
and GraphSAGE.

## 2. Why Task 19B Was Added

The Task 19 audit showed that the GCN PG pool is a subset of the GraphSAGE PG
pool. A symmetric common pool therefore exists and can reduce the
training-pool membership confound for a secondary analysis.

## 3. Post-Hoc Sensitivity Status

This is a post-hoc controlled sensitivity analysis conducted after primary
final-test exposure. It is not a pre-registered primary experiment and does
not replace the original Task 19 primary results.

Primary original results remain:

- GCN PGExplainer with backbone-specific pool `48/16`
- GraphSAGE PGExplainer with backbone-specific pool `106/35`

Controlled sensitivity results are:

- GCN PGExplainer common pool `48/16`, reusing the existing frozen result
- GraphSAGE PGExplainer common pool `48/16`, newly trained in Task 19B

## 4. Common-Pool Construction

The common pool was created by deterministic `node_id` intersection:

- Common train count: `{common['counts']['common_train']}`
- Common train SHA256: `{common['common_files']['train']['sha256']}`
- Common validation count: `{common['counts']['common_val']}`
- Common validation SHA256: `{common['common_files']['validation']['sha256']}`
- Final target count: `{common['counts']['final']}`
- Final target SHA256: `{common['source_files']['final_targets']['sha256']}`

Train, validation, and final-test pools are disjoint.

## 5. Frozen Inputs

- Dataset SHA256: `{manifest['dataset']['sha256']}`
- GCN checkpoint SHA256: `{manifest['checkpoints']['gcn']['sha256']}`
- GraphSAGE checkpoint SHA256: `{manifest['checkpoints']['graphsage']['sha256']}`
- GCN PG seed42 SHA256: `{manifest['checkpoints']['gcn_pg_seed42']['sha256']}`
- GraphSAGE natural PG seed42 SHA256:
  `{manifest['checkpoints']['graphsage_pg_natural_seed42']['sha256']}`
- GraphSAGE common PG seed42 SHA256:
  `{manifest['checkpoints']['graphsage_pg_common_seed42']['sha256']}`

## 6. GCN Common-Pool Reuse Verification

The existing frozen GCN PGExplainer train identities exactly equal the common
train identities, and the existing GCN validation identities exactly equal the
common validation identities. GCN PGExplainer was not retrained.

## 7. GraphSAGE Common-Pool Training

- Seed: `42`
- Epochs completed: `{train['epochs_completed']}`
- Final mean loss: `{train['final_loss']:.6f}`
- Training runtime seconds: `{train['training_runtime_seconds']:.6f}`
- GraphSAGE weights unchanged: `{train['model_parameters_unchanged']}`
- Optimizer excludes GraphSAGE parameters:
  `{train['optimizer_excludes_graphsage_parameters']}`
- Checkpoint reload: `{train['checkpoint_reload_ok']}`

## 8. Seed42 Quality Results

- Targets: `12`
- Success: `{eval_result['success_count']}`
- Failed: `{eval_result['failure_count']}`
- Edge AUROC: `{metric_mean(summary, 'edge_auroc'):.6f}`
- Edge AP: `{metric_mean(summary, 'edge_ap'):.6f}`
- Fid+: `{metric_mean(summary, 'fid_plus'):.6f}`
- Fid-: `{metric_mean(summary, 'fid_minus'):.6f}`
- 1-Fid-: `{metric_mean(summary, 'one_minus_fid_minus'):.6f}`
- Sparsity: `{metric_mean(summary, 'sparsity'):.6f}`

## 9. Stability Results

- Seeds: `{STABILITY_SEEDS}`
- Pairwise rows: `{stability['pairwise_row_count']}`
- Jaccard mean: `{stability['mean']:.6f}`
- Jaccard std: `{stability['std']:.6f}`
- Jaccard median: `{stability['median']:.6f}`
- Jaccard min: `{stability['min']:.6f}`
- Jaccard max: `{stability['max']:.6f}`

## 10. Runtime

Runtime is decomposed into training per checkpoint, inference per target, and
evaluation per final-target batch. These values are descriptive efficiency
references and are not used for model selection.

## 11. GraphSAGE Natural vs Common Pool

| Metric | Natural 106/35 | Common 48/16 | Delta |
|---|---:|---:|---:|
{natural_table}

## 12. Controlled GCN vs GraphSAGE PG Comparison

| Metric | PG + GCN common | PG + GraphSAGE common | Delta |
|---|---:|---:|---:|
{cross_table}

## 13. Interpretation

Holding PGExplainer training-node and validation-node identities fixed across
backbones reduces the training-pool membership confound and provides a more
controlled cross-backbone sensitivity comparison. This does not prove a causal
backbone effect because the backbones still differ in architecture,
representations, predictive competence, and optimization history.

The strong GraphSAGE PGExplainer result persisted under the reduced common
training pool, suggesting that the original result was not solely attributable
to the larger backbone-specific PGExplainer training pool. The small decreases
in AP and stability are descriptive sensitivity signals, not failure cases and
not grounds for tuning.

## 14. Limitations

The final 12 BA-House targets had already been exposed before this task. The
analysis is therefore secondary and descriptive.

## 15. Implications for Task 20

Task 20 should keep three layers separate: primary within-backbone explainer
comparison, controlled PG cross-backbone sensitivity, and GraphSAGE PG
training-pool sensitivity.

## 16. Implications for ShapeGGen

For ShapeGGen, common cross-backbone PG train/validation pools should be
frozen before explainer training and before final explanation evaluation. This
prevents repeating the BA-House training-pool confound.

## 17. Freeze Statement

The original Task 19 artifacts remain unchanged. No GCN PG retraining,
hyperparameter tuning, best-seed selection, target modification, or Task 20
analysis was performed in this task.
"""


def comparison_metrics() -> list[str]:
    return [
        "edge_auroc",
        "edge_ap",
        "fid_plus",
        "fid_minus",
        "one_minus_fid_minus",
        "stability",
        "sparsity",
        "inference_runtime_seconds",
    ]


def comparison_value(
    summary: dict[str, dict[str, Any]],
    stability: dict[str, Any],
    metric: str,
) -> float:
    if metric == "stability":
        return float(stability["mean"])
    return float(summary[metric]["mean"])


def metric_mean(summary: dict[str, dict[str, Any]], metric: str) -> float:
    return float(summary[metric]["mean"])


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = []
    for row in rows:
        values = [row[key] for key in keys]
        lines.append(
            "| "
            + " | ".join(
                str(value) if isinstance(value, str) else f"{float(value):.6f}"
                for value in values
            )
            + " |"
        )
    return "\n".join(lines)


def selected_edges(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["global_edge_index"]) for row in csv.DictReader(handle)}


def rows_for_ids(rows: list[dict[str, str]], ids: list[int]) -> list[dict[str, str]]:
    by_id = {int(row["node_id"]): row for row in rows}
    return [by_id[node_id] for node_id in ids]


def node_ids(rows: list[dict[str, str]]) -> set[int]:
    return {int(row["node_id"]) for row in rows}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path) -> dict[str, dict[str, Any]]:
    summary = {}
    for row in read_csv(path):
        summary[row["metric"]] = {
            "n": int(row["n"]) if row["n"] else 0,
            "mean": float(row["mean"]) if row["mean"] else None,
            "std": float(row["std"]) if row["std"] else None,
            "median": float(row["median"]) if row["median"] else None,
            "min": float(row["min"]) if row["min"] else None,
            "max": float(row["max"]) if row["max"] else None,
        }
    return summary


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "count": len(read_csv(path)) if path.suffix.lower() == ".csv" else None,
    }


def summarize(values: Any) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "n": len(numbers),
        "mean": mean(numbers),
        "std": stdev(numbers) if len(numbers) > 1 else 0.0,
        "median": median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
        encoding="utf-8",
    ).strip()


if __name__ == "__main__":
    main()
