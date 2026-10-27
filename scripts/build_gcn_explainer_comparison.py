from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

METHODS = ["Random", "GNNExplainer", "PGExplainer"]
QUALITY_METRICS = [
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "remove_probability_delta",
    "retain_probability_delta",
    "retain_probability_ratio",
    "sparsity",
]


def main() -> None:
    build_gcn_explainer_comparison()


def build_gcn_explainer_comparison() -> dict[str, Any]:
    output_dir = Path("results/metrics/gcn_explainer_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    gnn_rows = read_csv(Path("results/metrics/frozen_phase4c_v2/per_node_metrics.csv"))
    random_rows = read_csv(
        Path("results/metrics/phase4d_random_baseline/random_baseline_per_node.csv")
    )
    pg_rows = read_csv(Path("results/metrics/pgexplainer_gcn_seed42/per_node_metrics.csv"))
    phase4_rows = {int(row["node_id"]): row for row in gnn_rows}

    validate_method_parity("GNNExplainer", gnn_rows, phase4_rows)
    validate_method_parity("Random", random_rows, phase4_rows)
    validate_method_parity("PGExplainer", pg_rows, phase4_rows)

    random_by_node = aggregate_rows_by_node(random_rows)
    gnn_by_node = {int(row["node_id"]): row for row in gnn_rows}
    pg_by_node = {int(row["node_id"]): row for row in pg_rows}
    node_ids = sorted(gnn_by_node)
    if node_ids != sorted(random_by_node) or node_ids != sorted(pg_by_node):
        raise ValueError("Method target node ids do not match")

    summary_rows = build_summary_rows(random_by_node)
    wide_per_node_rows = build_wide_per_node_rows(
        node_ids,
        gnn_by_node,
        pg_by_node,
        random_by_node,
    )
    per_node_rows = build_long_per_node_rows(wide_per_node_rows)
    paired_delta_rows = build_paired_delta_rows(wide_per_node_rows)
    runtime_rows = build_runtime_rows()
    random_aggregation_audit_rows = build_random_aggregation_audit_rows(
        random_rows,
        random_by_node,
    )

    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_node.csv", per_node_rows)
    write_csv(output_dir / "paired_deltas.csv", paired_delta_rows)
    write_csv(output_dir / "runtime.csv", runtime_rows)
    write_csv(output_dir / "random_aggregation_audit.csv", random_aggregation_audit_rows)

    manifest = build_manifest(
        output_dir,
        summary_rows,
        per_node_rows,
        paired_delta_rows,
        random_aggregation_audit_rows,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_freeze_docs(
        manifest,
        summary_rows,
        runtime_rows,
        random_aggregation_audit_rows,
        comparison_manifest_sha256=sha256_file(output_dir / "manifest.json"),
    )
    print(
        "GCN explainer comparison frozen | "
        f"targets={manifest['target_count']} | "
        f"methods={','.join(METHODS)}"
    )
    return manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_method_parity(
    method: str,
    rows: list[dict[str, str]],
    reference_by_node: dict[int, dict[str, str]],
) -> None:
    required_fields = [
        "candidate_edge_count",
        "positive_gt_edges",
        "negative_gt_edges",
        "selected_edge_count",
        "original_prediction",
    ]
    for row in rows:
        node_id = int(row["node_id"])
        if node_id not in reference_by_node:
            raise ValueError(f"{method} contains non-reference node {node_id}")
        reference = reference_by_node[node_id]
        for field in required_fields:
            if int(float(row[field])) != int(float(reference[field])):
                raise ValueError(
                    f"{method} parity mismatch node={node_id} field={field}: "
                    f"{row[field]} != {reference[field]}"
                )
        if abs(float(row["selected_edge_ratio"]) - 0.2) > 1e-12:
            raise ValueError(f"{method} selected_edge_ratio is not 0.2")


def aggregate_rows_by_node(rows: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["node_id"])].append(row)
    aggregated = {}
    for node_id, node_rows in grouped.items():
        first = node_rows[0]
        row: dict[str, Any] = {
            "node_id": node_id,
            "label": first["label"],
            "prediction": first["prediction"],
            "confidence": first["confidence"],
            "candidate_edge_count": first["candidate_edge_count"],
            "positive_gt_edges": first["positive_gt_edges"],
            "negative_gt_edges": first["negative_gt_edges"],
            "selected_edge_ratio": first["selected_edge_ratio"],
            "selected_edge_count": first["selected_edge_count"],
            "original_prediction": first["original_prediction"],
        }
        for metric in QUALITY_METRICS:
            row[metric] = mean(float(item[metric]) for item in node_rows)
        aggregated[node_id] = row
    return aggregated


def build_summary_rows(random_by_node: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = {
        "Random": summarize_quality_rows(list(random_by_node.values())),
        "GNNExplainer": read_summary(Path("results/metrics/frozen_phase4c_v2/summary_metrics.csv")),
        "PGExplainer": read_summary(
            Path("results/metrics/pgexplainer_gcn_seed42/summary_metrics.csv")
        ),
    }
    stability = {
        "Random": read_random_stability(),
        "GNNExplainer": read_gnn_stability(),
        "PGExplainer": read_pg_stability(),
    }
    rows = []
    for method in METHODS:
        summary = summaries[method]
        rows.append(
            {
                "method": method,
                "edge_auroc_mean": summary["edge_auroc"]["mean"],
                "edge_auroc_std": summary["edge_auroc"]["std"],
                "edge_auroc_median": summary["edge_auroc"]["median"],
                "edge_ap_mean": summary["edge_ap"]["mean"],
                "edge_ap_std": summary["edge_ap"]["std"],
                "edge_ap_median": summary["edge_ap"]["median"],
                "fid_plus_mean": summary["fid_plus"]["mean"],
                "fid_plus_std": summary["fid_plus"]["std"],
                "fid_plus_median": summary["fid_plus"]["median"],
                "fid_minus_mean": summary["fid_minus"]["mean"],
                "fid_minus_std": summary["fid_minus"]["std"],
                "fid_minus_median": summary["fid_minus"]["median"],
                "one_minus_fid_minus_mean": summary["one_minus_fid_minus"]["mean"],
                "one_minus_fid_minus_std": summary["one_minus_fid_minus"]["std"],
                "one_minus_fid_minus_median": summary[
                    "one_minus_fid_minus"
                ]["median"],
                "remove_probability_delta_mean": summary[
                    "remove_probability_delta"
                ]["mean"],
                "remove_probability_delta_std": summary[
                    "remove_probability_delta"
                ]["std"],
                "remove_probability_delta_median": summary[
                    "remove_probability_delta"
                ]["median"],
                "retain_probability_delta_mean": summary[
                    "retain_probability_delta"
                ]["mean"],
                "retain_probability_delta_std": summary[
                    "retain_probability_delta"
                ]["std"],
                "retain_probability_delta_median": summary[
                    "retain_probability_delta"
                ]["median"],
                "sparsity_mean": summary["sparsity"]["mean"],
                "sparsity_std": summary["sparsity"]["std"],
                "sparsity_median": summary["sparsity"]["median"],
                "stability_mean": stability[method]["mean"],
                "stability_std": stability[method]["std"],
                "stability_median": stability[method]["median"],
                "runtime_semantics": runtime_semantics(method),
            }
        )
    return rows


def summarize_quality_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        metric: summarize_values(float(row[metric]) for row in rows)
        for metric in QUALITY_METRICS
    }


def summarize_values(values: Any) -> dict[str, float]:
    items = list(values)
    if not items:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "n": float(len(items)),
        "mean": mean(items),
        "std": stdev(items) if len(items) > 1 else 0.0,
        "median": median(items),
        "min": min(items),
        "max": max(items),
    }


def read_summary(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    return {
        row["metric"]: {
            "n": float(row["n"]) if row["n"] else 0.0,
            "mean": float(row["mean"]) if row["mean"] else float("nan"),
            "std": float(row["std"]) if row["std"] else float("nan"),
            "median": float(row["median"]) if row["median"] else float("nan"),
            "min": float(row["min"]) if row["min"] else float("nan"),
            "max": float(row["max"]) if row["max"] else float("nan"),
        }
        for row in rows
    }


def read_gnn_stability() -> dict[str, float]:
    row = read_csv(Path("results/metrics/phase4d_stability/gnnexplainer_stability_summary.csv"))[0]
    return {
        "mean": float(row["mean"]),
        "std": float(row["std"]),
        "median": float(row["median"]),
    }


def read_random_stability() -> dict[str, float]:
    row = read_csv(
        Path("results/metrics/phase4d_random_baseline/random_stability_summary.csv")
    )[0]
    return {
        "mean": float(row["mean"]),
        "std": float(row["std"]),
        "median": float(row["median"]),
    }


def read_pg_stability() -> dict[str, float]:
    data = json.loads(Path("docs/pgexplainer_gcn_stability_freeze.json").read_text())
    summary = data["jaccard_summary"]
    return {
        "mean": float(summary["mean"]),
        "std": float(summary["std"]),
        "median": float(summary["median"]),
    }


def runtime_semantics(method: str) -> str:
    if method == "GNNExplainer":
        return "per-target optimization/generation runtime"
    if method == "PGExplainer":
        return "upfront training plus trained-explainer inference per target"
    return "random edge selection plus metric evaluation runtime"


def build_wide_per_node_rows(
    node_ids: list[int],
    gnn_by_node: dict[int, dict[str, str]],
    pg_by_node: dict[int, dict[str, str]],
    random_by_node: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for node_id in node_ids:
        row: dict[str, Any] = {
            "node_id": node_id,
            "candidate_edge_count": gnn_by_node[node_id]["candidate_edge_count"],
            "positive_gt_edges": gnn_by_node[node_id]["positive_gt_edges"],
            "negative_gt_edges": gnn_by_node[node_id]["negative_gt_edges"],
            "selected_edge_count": gnn_by_node[node_id]["selected_edge_count"],
            "original_prediction": gnn_by_node[node_id]["original_prediction"],
        }
        for prefix, source in [
            ("random", random_by_node[node_id]),
            ("gnn", gnn_by_node[node_id]),
            ("pg", pg_by_node[node_id]),
        ]:
            for metric in QUALITY_METRICS:
                row[f"{prefix}_{metric}"] = float(source[metric])
        rows.append(row)
    return rows


def build_long_per_node_rows(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for wide_row in wide_rows:
        base = {
            "node_id": wide_row["node_id"],
            "candidate_edge_count": wide_row["candidate_edge_count"],
            "positive_gt_edges": wide_row["positive_gt_edges"],
            "negative_gt_edges": wide_row["negative_gt_edges"],
            "selected_edge_count": wide_row["selected_edge_count"],
            "original_prediction": wide_row["original_prediction"],
        }
        for method, prefix in [
            ("Random", "random"),
            ("GNNExplainer", "gnn"),
            ("PGExplainer", "pg"),
        ]:
            row = {**base, "method": method}
            for metric in QUALITY_METRICS:
                row[metric] = wide_row[f"{prefix}_{metric}"]
            rows.append(row)
    return rows


def build_paired_delta_rows(per_node_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in per_node_rows:
        out = {"node_id": row["node_id"]}
        for metric in QUALITY_METRICS:
            out[f"pg_minus_gnn_{metric}"] = row[f"pg_{metric}"] - row[f"gnn_{metric}"]
            out[f"gnn_minus_random_{metric}"] = row[f"gnn_{metric}"] - row[f"random_{metric}"]
            out[f"pg_minus_random_{metric}"] = row[f"pg_{metric}"] - row[f"random_{metric}"]
        rows.append(out)
    return rows


def build_random_aggregation_audit_rows(
    random_rows: list[dict[str, str]],
    random_by_node: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    node_rows = list(random_by_node.values())
    rows = []
    for metric in QUALITY_METRICS:
        raw = summarize_values(float(row[metric]) for row in random_rows)
        node = summarize_values(float(row[metric]) for row in node_rows)
        rows.append(
            {
                "metric": metric,
                "raw_seed_level_n": int(raw["n"]),
                "raw_seed_level_mean": raw["mean"],
                "raw_seed_level_std": raw["std"],
                "raw_seed_level_median": raw["median"],
                "node_level_n": int(node["n"]),
                "node_level_mean": node["mean"],
                "node_level_std": node["std"],
                "node_level_median": node["median"],
                "mean_changed": abs(raw["mean"] - node["mean"]) > 1e-12,
                "std_changed": abs(raw["std"] - node["std"]) > 1e-12,
                "median_changed": abs(raw["median"] - node["median"]) > 1e-12,
            }
        )
    return rows


def build_runtime_rows() -> list[dict[str, Any]]:
    gnn_summary = read_summary(Path("results/metrics/frozen_phase4c_v2/summary_metrics.csv"))
    random_summary = read_summary(
        Path("results/metrics/phase4d_random_baseline/random_baseline_summary.csv")
    )
    pg_summary = read_summary(Path("results/metrics/pgexplainer_gcn_seed42/summary_metrics.csv"))
    pg_stability = json.loads(Path("docs/pgexplainer_gcn_stability_freeze.json").read_text())
    seed42_freeze = json.loads(
        Path("docs/pgexplainer_gcn_seed42_evaluation_freeze.json").read_text()
    )
    return [
        {
            "method": "Random",
            "upfront_training_runtime_mean": "",
            "per_target_generation_or_inference_mean": "",
            "metric_evaluation_runtime_mean": random_summary["evaluation_runtime_seconds"]["mean"],
            "runtime_semantics": runtime_semantics("Random"),
        },
        {
            "method": "GNNExplainer",
            "upfront_training_runtime_mean": "",
            "per_target_generation_or_inference_mean": gnn_summary[
                "generation_runtime_seconds"
            ]["mean"],
            "metric_evaluation_runtime_mean": "",
            "runtime_semantics": runtime_semantics("GNNExplainer"),
        },
        {
            "method": "PGExplainer",
            "upfront_training_runtime_mean": seed42_freeze["runtime_summary"][
                "pgexplainer_upfront_training_runtime_seconds"
            ],
            "per_target_generation_or_inference_mean": pg_summary[
                "inference_runtime_seconds"
            ]["mean"],
            "metric_evaluation_runtime_mean": pg_stability["runtime"][
                "evaluation_runtime_seconds"
            ]["mean"],
            "runtime_semantics": runtime_semantics("PGExplainer"),
        },
    ]


def build_manifest(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    per_node_rows: list[dict[str, Any]],
    paired_delta_rows: list[dict[str, Any]],
    random_aggregation_audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "success",
        "backbone": "frozen GCN",
        "target_count": 12,
        "methods": METHODS,
        "only_frozen_outputs_reused": True,
        "post_test_tuning": False,
        "primary_comparison_unit": "node_id",
        "primary_target_count": 12,
        "primary_pg_quality_seed": 42,
        "random_aggregation": {
            "raw_run_count": 120,
            "seeds_per_node": 10,
            "main_aggregation": (
                "mean within node_id across 10 frozen random seeds, "
                "then summarize across 12 nodes"
            ),
            "raw_seed_level_artifact": (
                "results/metrics/phase4d_random_baseline/"
                "random_baseline_per_node.csv"
            ),
            "audit_artifact": (
                "results/metrics/gcn_explainer_comparison/"
                "random_aggregation_audit.csv"
            ),
        },
        "primary_rows": {
            "Random": 12,
            "GNNExplainer": 12,
            "PGExplainer": 12,
        },
        "stability_protocol": {
            "metric": "pairwise Jaccard(A, B) = |A intersection B| / |A union B|",
            "selected_edge_sets": "top-20% selected candidate-edge sets",
            "stochastic_source": {
                "Random": "repeated random edge selections",
                "GNNExplainer": (
                    "independent explanation optimization runs / "
                    "frozen GNNExplainer stability runs"
                ),
                "PGExplainer": "independently trained PGExplainer checkpoints",
            },
            "pair_counts_per_node": {
                "Random": 45,
                "GNNExplainer": 10,
                "PGExplainer": 10,
            },
            "seed_counts": {
                "Random": 10,
                "GNNExplainer": 5,
                "PGExplainer": 5,
            },
        },
        "fairness_checks": {
            "join_key": "node_id",
            "candidate_edge_count": "12/12 all methods",
            "positive_negative_gt_counts": "12/12 all methods",
            "selected_edge_count": "12/12 all methods",
            "original_prediction": "12/12 all methods",
            "selected_edge_ratio": 0.2,
        },
        "authoritative_sources": authoritative_sources(),
        "summary_rows": summary_rows,
        "per_node_row_count": len(per_node_rows),
        "paired_delta_row_count": len(paired_delta_rows),
        "random_aggregation_audit_rows": random_aggregation_audit_rows,
        "runtime_semantics": {
            method: runtime_semantics(method) for method in METHODS
        },
        "result_file_hashes": {
            "summary": sha256_file(output_dir / "summary.csv"),
            "per_node": sha256_file(output_dir / "per_node.csv"),
            "paired_deltas": sha256_file(output_dir / "paired_deltas.csv"),
            "runtime": sha256_file(output_dir / "runtime.csv"),
            "random_aggregation_audit": sha256_file(
                output_dir / "random_aggregation_audit.csv"
            ),
        },
        "comparison_code": {
            "script": "scripts/build_gcn_explainer_comparison.py",
            "script_sha256": sha256_file(Path("scripts/build_gcn_explainer_comparison.py")),
            "base_commit": current_git_commit(),
        },
    }


def authoritative_sources() -> dict[str, Any]:
    return {
        "GNNExplainer": {
            "quality": "results/metrics/frozen_phase4c_v2/per_node_metrics.csv",
            "stability": "results/metrics/phase4d_stability/gnnexplainer_stability_summary.csv",
            "seeds": [0, 1, 2, 3, 4],
        },
        "Random": {
            "quality": "results/metrics/phase4d_random_baseline/random_baseline_per_node.csv",
            "stability": "results/metrics/phase4d_random_baseline/random_stability_summary.csv",
            "seeds": list(range(10)),
        },
        "PGExplainer": {
            "quality": "results/metrics/pgexplainer_gcn_seed42/per_node_metrics.csv",
            "stability": "docs/pgexplainer_gcn_stability_freeze.json",
            "quality_seed": 42,
            "stability_seeds": [0, 1, 2, 3, 4],
        },
    }


def write_freeze_docs(
    manifest: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    random_aggregation_audit_rows: list[dict[str, Any]],
    comparison_manifest_sha256: str,
) -> None:
    freeze_manifest = {
        **manifest,
        "comparison_result_manifest_sha256": comparison_manifest_sha256,
    }
    Path("docs/gcn_explainer_comparison_freeze.json").write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    row_by_method = {row["method"]: row for row in summary_rows}
    random_audit = {row["metric"]: row for row in random_aggregation_audit_rows}
    report = f"""# GCN Explainer Three-Way Comparison Freeze

## Scope

This report compares Random Explanation, GNNExplainer, and PGExplainer on the
same frozen GCN backbone and the same 12 frozen BA-House final targets. It
reuses frozen outputs only and does not rerun, tune, or replace any experiment.

## Inputs And Protocol

- Backbone: frozen GCN.
- Target join key: `node_id`.
- Candidate-edge scope: computation subgraph.
- Edge AUROC/AP: continuous scores over the full candidate-edge set.
- Explanation budget: fixed candidate-edge ratio `0.2`.
- Sparsity is primarily a protocol confirmation because the selected ratio is fixed.
- PGExplainer primary quality uses seed `42`, not a best seed from stability.
- Primary comparison unit: one node-level observation per `node_id`.

## Fairness Checks

All methods match on candidate-edge count, positive/negative GT counts,
selected-edge count, original prediction, and selected ratio for `12/12`
targets. Random uses 10 frozen random seeds. GNNExplainer and PGExplainer
stability use 5 frozen seeds.

## Summary

Values in the main table are means over the frozen evaluation rows. The full
`summary.csv` also records standard deviation and median columns. Random is
first averaged within each `node_id` across its 10 frozen random seeds, then
summarized across the same 12 node-level observations used by GNNExplainer and
PGExplainer. The original 120 Random rows remain preserved in the frozen Random
artifact and the aggregation audit is saved in
`random_aggregation_audit.csv`.

| Method | AUROC | AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability | Runtime semantics |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{summary_markdown_row(row_by_method['Random'])}
{summary_markdown_row(row_by_method['GNNExplainer'])}
{summary_markdown_row(row_by_method['PGExplainer'])}

## Random Aggregation Audit

Random has 120 frozen seed-level rows: 12 nodes times 10 seeds. For the main
comparison it is reduced to 12 node-level values by taking the mean over the 10
seeds within each node. Because every node has exactly 10 Random seeds, AUROC
and AP means are unchanged, while standard deviation and median reflect the
node-level sampling unit.

| Metric | Raw n | Raw mean | Raw std | Raw median | Node n | Node mean | Node std | Node median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{random_audit_markdown_row(random_audit['edge_auroc'])}
{random_audit_markdown_row(random_audit['edge_ap'])}

## Stability Protocol

All methods use the same pairwise Jaccard metric over top-20% selected
candidate-edge sets: `Jaccard(A, B) = |A intersection B| / |A union B|`. The
stochastic source follows each method's frozen protocol: Random uses repeated
random edge selections, GNNExplainer uses independent explanation optimization
runs, and PGExplainer uses independently trained checkpoints.

| Method | Seeds/checkpoints | Pairs per node | Stochastic source |
|---|---:|---:|---|
| Random | 10 | 45 | repeated random edge selections |
| GNNExplainer | 5 | 10 | independent explanation optimization runs |
| PGExplainer | 5 | 10 | independently trained PGExplainer checkpoints |

## Runtime

Runtime is decomposed rather than collapsed into a single winner column.
GNNExplainer runtime is per-target optimization/generation. PGExplainer has an
upfront training cost and then cheap trained-explainer inference. Random runtime
is random edge selection plus metric evaluation. Runtime boundaries differ by
method and therefore are interpreted as computational-behavior descriptors
rather than perfectly identical wall-clock workloads.

| Method | Upfront training | Per-target generation/inference | Other eval cost | Semantics |
|---|---:|---:|---:|---|
{runtime_markdown_row(runtime_rows[0])}
{runtime_markdown_row(runtime_rows[1])}
{runtime_markdown_row(runtime_rows[2])}

The PGExplainer primary seed42 training runtime is kept separate from
Task16B independent stability-checkpoint training runtime. Stability-checkpoint
timings are not averaged into the primary seed42 quality runtime.

## Sparsity

Sparsity is largely determined by the fixed 20% candidate-edge budget and is
therefore a protocol confirmation rather than an independent method-quality
discriminator.

## Interpretation

On frozen GCN/BA-House, GNNExplainer achieves the strongest observed
ground-truth agreement and faithfulness, while PGExplainer exhibits the highest
observed top-20% Jaccard stability and low post-training per-target inference
cost. These descriptive results show a trade-off between explanation quality,
reproducibility, and computational behavior, not one method dominating every
evaluation dimension.

PGExplainer final-test results have already been exposed. They must not be used
to tune epochs, learning rate, regularization, temperature, bias, target
membership, checkpoint selection, or seed choice.

## Artifacts

- `results/metrics/gcn_explainer_comparison/summary.csv`
- `results/metrics/gcn_explainer_comparison/per_node.csv`
- `results/metrics/gcn_explainer_comparison/paired_deltas.csv`
- `results/metrics/gcn_explainer_comparison/runtime.csv`
- `results/metrics/gcn_explainer_comparison/random_aggregation_audit.csv`
- `results/metrics/gcn_explainer_comparison/manifest.json`
- `docs/gcn_explainer_comparison_freeze.json`
"""
    Path("docs/provenance/historical_reports/gcn_explainer_comparison_report.md").write_text(report, encoding="utf-8")


def summary_markdown_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['method']} | {float(row['edge_auroc_mean']):.6f} | "
        f"{float(row['edge_ap_mean']):.6f} | {float(row['fid_plus_mean']):.6f} | "
        f"{float(row['fid_minus_mean']):.6f} | "
        f"{float(row['one_minus_fid_minus_mean']):.6f} | "
        f"{float(row['sparsity_mean']):.6f} | "
        f"{float(row['stability_mean']):.6f} | {row['runtime_semantics']} |"
    )


def random_audit_markdown_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['metric']} | {int(row['raw_seed_level_n'])} | "
        f"{float(row['raw_seed_level_mean']):.6f} | "
        f"{float(row['raw_seed_level_std']):.6f} | "
        f"{float(row['raw_seed_level_median']):.6f} | "
        f"{int(row['node_level_n'])} | {float(row['node_level_mean']):.6f} | "
        f"{float(row['node_level_std']):.6f} | "
        f"{float(row['node_level_median']):.6f} |"
    )


def runtime_markdown_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['method']} | {format_runtime(row['upfront_training_runtime_mean'])} | "
        f"{format_runtime(row['per_target_generation_or_inference_mean'])} | "
        f"{format_runtime(row['metric_evaluation_runtime_mean'])} | "
        f"{row['runtime_semantics']} |"
    )


def format_runtime(value: Any) -> str:
    if value == "":
        return "N/A"
    return f"{float(value):.6f} s"


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
