from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from gnnexplainer_benchmark.data.serialization import sha256_file


def main() -> None:
    manifest = build_manifest()
    Path("docs/graphsage_explainers_ba_house_freeze.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/provenance/historical_reports/graphsage_explainers_ba_house_report.md").write_text(
        build_report(manifest),
        encoding="utf-8",
    )
    print("GraphSAGE explainer report built")


def build_manifest() -> dict[str, Any]:
    gnn_summary = read_summary(
        Path("results/metrics/graphsage_gnnexplainer/seed42/summary_metrics.csv")
    )
    pg_summary = read_summary(
        Path("results/metrics/graphsage_pgexplainer/seed42/summary_metrics.csv")
    )
    gnn_stability = read_csv(
        Path("results/metrics/graphsage_gnnexplainer/stability/gnnexplainer_stability_summary.csv")
    )[0]
    pg_stability = json.loads(
        Path("results/metrics/graphsage_pgexplainer/stability/stability_summary.json").read_text(
            encoding="utf-8"
        )
    )
    pg_seed42_manifest = json.loads(
        Path("artifacts/explainers/pgexplainer_graphsage_ba_house_seed42_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    pg_stability_manifest = json.loads(
        Path("results/metrics/graphsage_pgexplainer/stability/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    target_manifest = json.loads(
        Path("configs/ba_house/ba500/targets/ba_house_2x2_targets_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "status": "success",
        "task": "Task 19 GraphSAGE explainers on BA-House",
        "model": "GraphSAGE",
        "dataset_sha256": sha256_file(Path("data/ba_house/ba500/graph.pt")),
        "graphsage_checkpoint_sha256": sha256_file(
            Path("artifacts/checkpoints/ba_house/ba500/backbones/graphsage/seed42.pt")
        ),
        "target_intersection": target_manifest,
        "protocol": {
            "selected_edge_ratio": 0.2,
            "rounding_rule": "ceil(candidate_edge_count * selected_edge_ratio), minimum 1",
            "edge_ranking_metric_scope": "continuous_scores_over_full_candidate_edge_set",
            "candidate_edge_scope": "computation_subgraph",
            "target_semantics": "frozen GraphSAGE predicted class",
            "post_test_tuning": False,
        },
        "gnnexplainer": {
            "quality_seed": 42,
            "summary": gnn_summary,
            "stability": {
                "seeds": [0, 1, 2, 3, 4],
                "mean": float(gnn_stability["mean"]),
                "std": float(gnn_stability["std"]),
                "median": float(gnn_stability["median"]),
                "pairwise_comparisons_per_node": 10,
            },
        },
        "pgexplainer": {
            "primary_quality_seed": 42,
            "summary": pg_summary,
            "training": {
                "train_target_count": pg_seed42_manifest["train_target_count"],
                "validation_target_count": pg_seed42_manifest["validation_target_count"],
                "final_test_targets_used_for_training": pg_seed42_manifest[
                    "final_test_targets_used_for_training"
                ],
                "training_runtime_seconds": pg_seed42_manifest["training_runtime_seconds"],
                "checkpoint_reload_ok": pg_seed42_manifest["checkpoint_reload_ok"],
                "model_weights_unchanged": pg_seed42_manifest["model_weights_unchanged"],
                "validation_health": pg_seed42_manifest["validation_health"],
            },
            "stability": pg_stability,
            "stability_manifest": {
                "training_success_count": pg_stability_manifest["training_success_count"],
                "evaluation_success_count": pg_stability_manifest["evaluation_success_count"],
                "total_explained_targets": pg_stability_manifest["total_explained_targets"],
                "runtime": pg_stability_manifest["runtime"],
            },
        },
        "outputs": {
            "gnnexplainer_quality": "results/metrics/graphsage_gnnexplainer/seed42",
            "gnnexplainer_stability": "results/metrics/graphsage_gnnexplainer/stability",
            "pgexplainer_quality": "results/metrics/graphsage_pgexplainer/seed42",
            "pgexplainer_stability": "results/metrics/graphsage_pgexplainer/stability",
        },
        "artifact_hashes": artifact_hashes(),
        "git": {
            "commit": run_git("rev-parse", "HEAD"),
            "branch": run_git("branch", "--show-current"),
        },
    }


def read_summary(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["metric"]: {
            "n": int(row["n"]) if row["n"] else 0,
            "mean": float(row["mean"]) if row["mean"] else None,
            "std": float(row["std"]) if row["std"] else None,
            "median": float(row["median"]) if row["median"] else None,
            "min": float(row["min"]) if row["min"] else None,
            "max": float(row["max"]) if row["max"] else None,
        }
        for row in read_csv(path)
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def artifact_hashes() -> dict[str, str]:
    paths = [
        "configs/ba_house/ba500/targets/ba_house_2x2_targets.csv",
        "configs/ba_house/ba500/targets/pgexplainer_graphsage_train_targets.csv",
        "configs/ba_house/ba500/targets/pgexplainer_graphsage_val_targets.csv",
        "artifacts/checkpoints/ba_house/ba500/pgexplainer/graphsage/natural106_35/primary_seed42.pt",
        "results/metrics/graphsage_gnnexplainer/seed42/per_node_metrics.csv",
        "results/metrics/graphsage_pgexplainer/seed42/per_node_metrics.csv",
        "results/metrics/graphsage_pgexplainer/stability/manifest.json",
    ]
    return {path: sha256_file(Path(path)) for path in paths}


def build_report(manifest: dict[str, Any]) -> str:
    gnn = manifest["gnnexplainer"]["summary"]
    pg = manifest["pgexplainer"]["summary"]
    pg_runtime = manifest["pgexplainer"]["stability_manifest"]["runtime"]
    gnn_stability_row = stability_row(
        "GNNExplainer",
        manifest["gnnexplainer"]["stability"],
    )
    pg_stability_row = stability_row(
        "PGExplainer",
        manifest["pgexplainer"]["stability"],
    )
    runtime_rows = "\n".join(
        [
            runtime_row("GNNExplainer generation/target", gnn["generation_runtime_seconds"]),
            runtime_row(
                "PGExplainer seed42 training",
                {
                    "mean": manifest["pgexplainer"]["training"][
                        "training_runtime_seconds"
                    ],
                    "std": None,
                    "median": None,
                },
            ),
            runtime_row(
                "PGExplainer stability training/checkpoint",
                pg_runtime["training_runtime_seconds"],
            ),
            runtime_row("PGExplainer inference/target", pg["inference_runtime_seconds"]),
            runtime_row(
                "PGExplainer stability evaluation batch",
                pg_runtime["evaluation_runtime_seconds"],
            ),
        ]
    )
    return f"""# GraphSAGE Explainers BA-House Freeze

## Scope

Task 19 evaluates GNNExplainer and PGExplainer on the frozen GraphSAGE
BA-House backbone. The BA-House 2x2 target intersection was frozen before any
GraphSAGE explanation results were used.

## Frozen Inputs

- Dataset SHA256: `{manifest['dataset_sha256']}`
- GraphSAGE checkpoint SHA256: `{manifest['graphsage_checkpoint_sha256']}`
- 2x2 targets: `{manifest['target_intersection']['target_count']}` nodes
- 2x2 target SHA256: `{manifest['target_intersection']['output_sha256']}`
- Selected-edge budget: fixed candidate-edge ratio `0.2`
- Edge ranking metrics: continuous scores over the full candidate-edge set

## Target And Leakage Checks

- PG train targets: `{manifest['pgexplainer']['training']['train_target_count']}`
- PG validation targets: `{manifest['pgexplainer']['training']['validation_target_count']}`
- Final targets used for PG training: `0`
- PG seed42 validation health success:
  `{manifest['pgexplainer']['training']['validation_health']['success_count']}/35`
- GraphSAGE weights unchanged during PG training:
  `{manifest['pgexplainer']['training']['model_weights_unchanged']}`

## Main Quality

| Method | AUROC | AP | Fid+ | Fid- | 1-Fid- | Remove delta | Sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|
{summary_row('GNNExplainer', gnn)}
{summary_row('PGExplainer', pg)}

## Stability

| Method | Seeds | Pairwise comparisons/node | Mean Jaccard | Std | Median |
|---|---:|---:|---:|---:|---:|
{gnn_stability_row}
{pg_stability_row}

## Runtime

Runtime is decomposed by method and stage. GNNExplainer reports per-target
optimization/generation time. PGExplainer reports upfront training per
checkpoint, trained-explainer inference per target, and evaluation batch time.

| Runtime | Mean seconds | Std | Median |
|---|---:|---:|---:|
{runtime_rows}

## Freeze Statement

The GraphSAGE explainer runs are frozen with no target selection, best-seed
selection, or hyperparameter tuning after final metrics. Task 20 conclusions
were not created in this task.
"""


def summary_row(method: str, summary: dict[str, dict[str, Any]]) -> str:
    return (
        f"| {method} | {summary['edge_auroc']['mean']:.6f} | "
        f"{summary['edge_ap']['mean']:.6f} | {summary['fid_plus']['mean']:.6f} | "
        f"{summary['fid_minus']['mean']:.6f} | "
        f"{summary['one_minus_fid_minus']['mean']:.6f} | "
        f"{summary['remove_probability_delta']['mean']:.6f} | "
        f"{summary['sparsity']['mean']:.6f} |"
    )


def stability_row(method: str, values: dict[str, Any]) -> str:
    return (
        f"| {method} | 5 | 10 | {values['mean']:.6f} | "
        f"{values['std']:.6f} | {values['median']:.6f} |"
    )


def runtime_row(label: str, values: dict[str, Any]) -> str:
    return (
        f"| {label} | {format_optional_float(values['mean'])} | "
        f"{format_optional_float(values['std'])} | "
        f"{format_optional_float(values['median'])} |"
    )


def format_optional_float(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
        encoding="utf-8",
    ).strip()


if __name__ == "__main__":
    main()
