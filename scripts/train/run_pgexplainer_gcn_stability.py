# ruff: noqa: E402
from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

from gnnexplainer_benchmark.explainers.pgexplainer_evaluation import evaluate_pgexplainer_gcn
from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    config_hash,
    load_yaml,
    train_pgexplainer_gcn,
)

STABILITY_SEEDS = [0, 1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PGExplainer seeds 0..4 and freeze stability/runtime diagnostics."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pgexplainer_gcn_stability(device=args.device)


def run_pgexplainer_gcn_stability(device: str = "cpu") -> dict[str, Any]:
    metrics_root = Path("results/metrics/pgexplainer_gcn_stability")
    explanations_root = Path("results/explanations/pgexplainer_gcn_stability")
    metrics_root.mkdir(parents=True, exist_ok=True)
    explanations_root.mkdir(parents=True, exist_ok=True)

    seed_results = []
    training_rows = []
    evaluation_runtime_rows = []
    inference_rows = []
    for seed in STABILITY_SEEDS:
        artifact_path = Path(f"artifacts/explainers/pgexplainer_gcn_ba_house_seed{seed}.pt")
        manifest_path = Path(
            f"artifacts/explainers/pgexplainer_gcn_ba_house_seed{seed}_manifest.json"
        )
        training_result = train_pgexplainer_gcn(
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            training_log_path=metrics_root / f"seed_{seed}_training_log.csv",
            validation_health_path=metrics_root / f"seed_{seed}_validation_health.csv",
            validation_summary_path=metrics_root / f"seed_{seed}_validation_summary.json",
            device=device,
            seed_override=seed,
        )
        training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training_rows.append(
            {
                "seed": seed,
                "checkpoint": str(artifact_path),
                "checkpoint_sha256": sha256_file(artifact_path),
                "epochs_completed": training_result.epochs_completed,
                "final_loss": training_result.final_loss,
                "training_runtime_seconds": training_result.training_runtime_seconds,
                "gcn_parameters_unchanged": training_result.model_parameters_unchanged,
                "optimizer_excludes_gcn": training_manifest["optimizer_excludes_gcn_parameters"],
                "checkpoint_reload_ok": training_result.checkpoint_reload_ok,
                "final_test_targets_used_for_training": training_manifest[
                    "final_test_targets_used_for_training"
                ],
            }
        )

        eval_start = perf_counter()
        evaluation_result = evaluate_pgexplainer_gcn(
            explainer_checkpoint_path=artifact_path,
            explainer_manifest_path=manifest_path,
            output_root=explanations_root / f"seed_{seed}",
            metrics_output_dir=metrics_root / f"seed_{seed}",
            device=device,
        )
        evaluation_runtime_seconds = perf_counter() - eval_start
        evaluation_runtime_rows.append(
            {
                "seed": seed,
                "evaluation_runtime_seconds": evaluation_runtime_seconds,
                "success_count": evaluation_result.success_count,
                "failure_count": evaluation_result.failure_count,
                "model_weights_unchanged": evaluation_result.model_weights_unchanged,
            }
        )
        per_seed_rows = read_csv(evaluation_result.per_node_metrics_path)
        for row in per_seed_rows:
            row["seed"] = str(seed)
            seed_results.append(row)
            inference_rows.append(
                {
                    "seed": seed,
                    "node_id": row["node_id"],
                    "inference_runtime_seconds": row["inference_runtime_seconds"],
                }
            )

    write_csv(metrics_root / "training_runtime.csv", training_rows)
    write_csv(metrics_root / "evaluation_runtime.csv", evaluation_runtime_rows)
    write_csv(metrics_root / "inference_runtime.csv", inference_rows)
    write_csv(metrics_root / "per_seed_metrics.csv", seed_results)

    pairwise_rows, per_node_rows, stability_summary = compute_stability_rows(
        explanations_root,
        STABILITY_SEEDS,
    )
    write_csv(metrics_root / "pairwise_jaccard.csv", pairwise_rows)
    write_csv(metrics_root / "per_node_jaccard.csv", per_node_rows)
    (metrics_root / "stability_summary.json").write_text(
        json.dumps(stability_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = build_stability_manifest(
        metrics_root=metrics_root,
        training_rows=training_rows,
        evaluation_runtime_rows=evaluation_runtime_rows,
        inference_rows=inference_rows,
        stability_summary=stability_summary,
        per_node_rows=per_node_rows,
        pairwise_rows=pairwise_rows,
    )
    (metrics_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_freeze_docs(manifest)
    print(
        "PGExplainer GCN stability complete | "
        f"seeds={STABILITY_SEEDS} | "
        f"jaccard_mean={stability_summary['mean']:.6f} | "
        f"pairwise_rows={len(pairwise_rows)}"
    )
    return manifest


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


def selected_edges(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["global_edge_index"]) for row in csv.DictReader(handle)}


def compute_stability_rows(
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


def build_stability_manifest(
    *,
    metrics_root: Path,
    training_rows: list[dict[str, Any]],
    evaluation_runtime_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    stability_summary: dict[str, Any],
    per_node_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    explainer_config = load_yaml(Path("configs/shared/pgexplainer.yaml"))
    checkpoint_hashes = {
        str(row["seed"]): {
            "path": row["checkpoint"],
            "sha256": row["checkpoint_sha256"],
        }
        for row in training_rows
    }
    return {
        "status": "success",
        "primary_quality_seed": 42,
        "primary_quality_seed_checkpoint_sha256": sha256_file(
            Path("artifacts/checkpoints/ba_house/ba500/pgexplainer/gcn/primary_seed42.pt")
        ),
        "stability_seeds": STABILITY_SEEDS,
        "dataset_sha256": sha256_file(Path("data/ba_house/ba500/graph.pt")),
        "gcn_checkpoint_sha256": sha256_file(
            Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt")
        ),
        "train_targets": {
            "count": 48,
            "sha256": sha256_file(
                Path("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv")
            ),
        },
        "validation_targets": {
            "count": 16,
            "sha256": sha256_file(
                Path("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv")
            ),
        },
        "final_targets": {
            "count": 12,
            "sha256": sha256_file(Path("results/metrics/explainer_test_targets.csv")),
        },
        "pgexplainer_config_hash": config_hash(explainer_config),
        "stability_code_base_commit": current_git_commit(),
        "stability_script_sha256": sha256_file(Path("scripts/train/run_pgexplainer_gcn_stability.py")),
        "training_module_sha256": sha256_file(
            Path("src/gnnexplainer_benchmark/explainers/pgexplainer_training.py")
        ),
        "evaluation_module_sha256": sha256_file(
            Path("src/gnnexplainer_benchmark/explainers/pgexplainer_evaluation.py")
        ),
        "checkpoint_hashes": checkpoint_hashes,
        "training_success_count": sum(
            bool(row["gcn_parameters_unchanged"])
            and bool(row["optimizer_excludes_gcn"])
            and bool(row["checkpoint_reload_ok"])
            and int(row["epochs_completed"]) == 30
            and int(row["final_test_targets_used_for_training"]) == 0
            for row in training_rows
        ),
        "evaluation_success_count": sum(
            int(row["success_count"]) == 12 for row in evaluation_runtime_rows
        ),
        "total_explained_targets": sum(
            int(row["success_count"]) for row in evaluation_runtime_rows
        ),
        "protocol_parity": {
            "candidate_edge_count": "12/12 per seed",
            "gt_edge_counts": "12/12 per seed",
            "selected_edge_count": "12/12 per seed",
            "original_prediction": "12/12 per seed",
        },
        "jaccard_summary": stability_summary,
        "runtime": {
            "training_runtime_seconds": summarize_values(
                float(row["training_runtime_seconds"]) for row in training_rows
            ),
            "inference_runtime_seconds": summarize_values(
                float(row["inference_runtime_seconds"]) for row in inference_rows
            ),
            "evaluation_runtime_seconds": summarize_values(
                float(row["evaluation_runtime_seconds"]) for row in evaluation_runtime_rows
            ),
            "semantics": {
                "training_runtime_seconds": "upfront PGExplainer checkpoint training",
                "inference_runtime_seconds": (
                    "trained PGExplainer explanation-score generation per target"
                ),
                "evaluation_runtime_seconds": (
                    "per-seed final-target batch scoring/evaluation wall time"
                ),
            },
        },
        "outputs": {
            "per_seed_metrics": str(metrics_root / "per_seed_metrics.csv"),
            "pairwise_jaccard": str(metrics_root / "pairwise_jaccard.csv"),
            "per_node_jaccard": str(metrics_root / "per_node_jaccard.csv"),
            "stability_summary": str(metrics_root / "stability_summary.json"),
            "training_runtime": str(metrics_root / "training_runtime.csv"),
            "inference_runtime": str(metrics_root / "inference_runtime.csv"),
            "evaluation_runtime": str(metrics_root / "evaluation_runtime.csv"),
        },
        "result_file_hashes": {
            name: sha256_file(Path(path))
            for name, path in {
                "per_seed_metrics": metrics_root / "per_seed_metrics.csv",
                "pairwise_jaccard": metrics_root / "pairwise_jaccard.csv",
                "per_node_jaccard": metrics_root / "per_node_jaccard.csv",
                "stability_summary": metrics_root / "stability_summary.json",
                "training_runtime": metrics_root / "training_runtime.csv",
                "inference_runtime": metrics_root / "inference_runtime.csv",
                "evaluation_runtime": metrics_root / "evaluation_runtime.csv",
            }.items()
        },
        "per_node_jaccard_rows": per_node_rows,
        "pairwise_row_count": len(pairwise_rows),
        "post_test_tuning": False,
        "best_seed_selection": False,
    }


def summarize_values(values: Any) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "n": len(numbers),
        "mean": mean(numbers),
        "std": stdev(numbers) if len(numbers) > 1 else 0.0,
        "median": median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def write_freeze_docs(manifest: dict[str, Any]) -> None:
    Path("docs/pgexplainer_gcn_stability_freeze.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    jaccard = manifest["jaccard_summary"]
    runtime = manifest["runtime"]
    training_runtime_row = markdown_runtime_row(
        "Training per checkpoint",
        runtime["training_runtime_seconds"],
    )
    inference_runtime_row = markdown_runtime_row(
        "Inference per target",
        runtime["inference_runtime_seconds"],
    )
    evaluation_runtime_row = markdown_runtime_row(
        "Evaluation per seed batch",
        runtime["evaluation_runtime_seconds"],
    )
    report = f"""# PGExplainer GCN Stability And Runtime Freeze

## Scope

Task 16B trains independent PGExplainer checkpoints for seeds `0,1,2,3,4`
and evaluates the same 12 frozen final targets only for pre-specified stability
and sensitivity diagnostics. The primary quality result remains seed `42` and
is not replaced by any stability seed.

## Frozen Inputs

- Dataset SHA256: `{manifest["dataset_sha256"]}`
- Frozen GCN checkpoint SHA256: `{manifest["gcn_checkpoint_sha256"]}`
- PG train targets: `48 / {manifest["train_targets"]["sha256"]}`
- PG validation targets: `16 / {manifest["validation_targets"]["sha256"]}`
- Final targets: `12 / {manifest["final_targets"]["sha256"]}`
- Primary seed42 checkpoint SHA256: `{manifest["primary_quality_seed_checkpoint_sha256"]}`

## Stability Protocol

- Stability seeds: `{manifest["stability_seeds"]}`
- Epochs per checkpoint: `30`
- Selected-edge budget: fixed candidate-edge ratio `0.2`
- Pairwise Jaccard per node: `C(5,2) = 10` seed pairs
- Node aggregation: equal-weight mean of per-node Jaccard means

## Acceptance Checks

- Independent checkpoints trained: `{manifest["training_success_count"]}/5`
- Final-target evaluation batches successful: `{manifest["evaluation_success_count"]}/5`
- Total explained final targets: `{manifest["total_explained_targets"]}/60`
- Protocol parity: candidate counts, GT counts, selected counts, and original
  predictions match Phase 4 for `12/12` targets per seed.
- Post-test tuning: `false`
- Best-seed selection: `false`

## Code Provenance

- Stability code base commit: `{manifest["stability_code_base_commit"]}`
- Stability script SHA256: `{manifest["stability_script_sha256"]}`
- Training module SHA256: `{manifest["training_module_sha256"]}`
- Evaluation module SHA256: `{manifest["evaluation_module_sha256"]}`

## Jaccard Stability

| Metric | Value |
|---|---:|
| Node count | {jaccard["node_count"]} |
| Pairwise rows | {jaccard["pairwise_row_count"]} |
| Mean | {jaccard["mean"]:.6f} |
| Std | {jaccard["std"]:.6f} |
| Median | {jaccard["median"]:.6f} |
| Min | {jaccard["min"]:.6f} |
| Max | {jaccard["max"]:.6f} |

## Runtime Semantics

PGExplainer runtime is separated into upfront training, trained-explainer
inference, and per-seed evaluation batch wall time. These stages are not
collapsed into one ambiguous runtime.

| Runtime | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
{training_runtime_row}
{inference_runtime_row}
{evaluation_runtime_row}

## Freeze Statement

The PGExplainer GCN stability/runtime run is frozen for seeds `0..4`. Seed42
remains the primary quality checkpoint. The stability results are diagnostics
for sensitivity and must not be used to tune hyperparameters, alter target
membership, choose a better seed, or replace the primary seed42 result.
"""
    Path("docs/provenance/historical_reports/pgexplainer_gcn_stability_report.md").write_text(
        report,
        encoding="utf-8",
    )


def markdown_runtime_row(label: str, values: dict[str, float]) -> str:
    return (
        f"| {label} | {values['mean']:.6f} | {values['std']:.6f} | "
        f"{values['median']:.6f} | {values['min']:.6f} | {values['max']:.6f} |"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()


if __name__ == "__main__":
    main()
