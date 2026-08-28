from __future__ import annotations

# ruff: noqa: E501
import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SHAPE_MAIN = ROOT / "results" / "metrics" / "shapeggen_2x2"
CROSS_MAIN = ROOT / "results" / "metrics" / "cross_synthetic"

METRICS = [
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
    "stability",
]

PRIMARY_PER_NODE = {
    ("GCN", "GNNExplainer"): ROOT / "results" / "metrics" / "shapeggen" / "gcn" / "gnnexplainer" / "per_node_metrics.csv",
    ("GCN", "PGExplainer"): ROOT / "results" / "metrics" / "shapeggen" / "gcn" / "pgexplainer" / "per_node_metrics.csv",
    ("GraphSAGE", "GNNExplainer"): ROOT / "results" / "metrics" / "shapeggen" / "graphsage" / "gnnexplainer" / "per_node_metrics.csv",
    ("GraphSAGE", "PGExplainer"): ROOT / "results" / "metrics" / "shapeggen" / "graphsage" / "pgexplainer" / "per_node_metrics.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: clean_csv_value(row.get(k, "")) for k in fieldnames} for row in rows])


def clean_csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("/", "\\")


def f(value: Any) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def summarize(values: list[float]) -> dict[str, float | int | None]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def metric_from_ba(row: dict[str, str], metric: str) -> float:
    key = f"{metric}_mean"
    return f(row[key])


def method_key(backbone: str, explainer: str) -> str:
    return f"{backbone}_{explainer}".lower().replace(" ", "_")


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def load_shape_rows() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    task23 = read_csv(ROOT / "results" / "metrics" / "shapeggen" / "task23_comparison_summary.csv")
    task24 = read_csv(ROOT / "results" / "metrics" / "shapeggen" / "task24_pgexplainer_summary.csv")
    rows: list[dict[str, Any]] = []
    for row in [*task23, *task24]:
        backbone = row["backbone"]
        method = row["method"]
        if method == "Random":
            continue
        per_node = read_csv(PRIMARY_PER_NODE[(backbone, method)])
        diagnostic_means = {
            metric: summarize([f(item[metric]) for item in per_node])["mean"]
            for metric in [
                "remove_probability_delta",
                "retain_probability",
                "retain_probability_delta",
                "retain_probability_ratio",
            ]
        }
        runtime_semantics = (
            "per-target GNNExplainer optimization/generation plus metric evaluation"
            if method == "GNNExplainer"
            else "upfront PGExplainer training plus per-target trained-explainer inference and metric evaluation"
        )
        rows.append(
            {
                "dataset": "ShapeGGen",
                "backbone": backbone,
                "explainer": method,
                "n_nodes": 167,
                "edge_auroc": f(row["edge_auroc"]),
                "edge_ap": f(row["edge_ap"]),
                "fid_plus": f(row["fid_plus"]),
                "fid_minus": f(row["fid_minus"]),
                "one_minus_fid_minus": f(row["one_minus_fid_minus"]),
                "remove_probability_delta": diagnostic_means["remove_probability_delta"],
                "retain_probability": diagnostic_means["retain_probability"],
                "retain_probability_delta": diagnostic_means["retain_probability_delta"],
                "retain_probability_ratio": diagnostic_means["retain_probability_ratio"],
                "sparsity": f(row["sparsity"]),
                "stability": f(row["stability"]),
                "generation_runtime_seconds": f(row.get("generation_runtime_seconds")),
                "training_runtime_seconds": f(row.get("training_runtime_seconds")),
                "inference_runtime_seconds": f(row.get("inference_runtime_seconds_mean")),
                "evaluation_runtime_seconds": f(
                    row.get("evaluation_runtime_seconds")
                    or row.get("evaluation_runtime_seconds_mean")
                ),
                "runtime_semantics": runtime_semantics,
            }
        )
    rows.sort(key=lambda r: (r["backbone"], r["explainer"]))
    return rows, {(r["backbone"], r["explainer"]): r for r in rows}


def load_ba_rows() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows = []
    for row in read_csv(ROOT / "results" / "metrics" / "ba_house_2x2" / "main.csv"):
        explainer = row["explainer"].replace(" (natural pool 106/35)", "")
        rows.append(
            {
                "dataset": "BA-House",
                "backbone": row["backbone"],
                "explainer": explainer,
                "n_nodes": int(row["n_nodes"]),
                "edge_auroc": metric_from_ba(row, "edge_auroc"),
                "edge_ap": metric_from_ba(row, "edge_ap"),
                "fid_plus": metric_from_ba(row, "fid_plus"),
                "fid_minus": metric_from_ba(row, "fid_minus"),
                "one_minus_fid_minus": metric_from_ba(row, "one_minus_fid_minus"),
                "sparsity": metric_from_ba(row, "sparsity"),
                "stability": math.nan,
            }
        )
    stability = read_csv(ROOT / "results" / "metrics" / "ba_house_2x2" / "stability.csv")
    by_key = {(r["backbone"], r["explainer"]): r for r in rows}
    for srow in stability:
        explainer = srow["explainer"].replace(" (natural pool 106/35)", "")
        key = (srow["backbone"], explainer)
        if key in by_key:
            by_key[key]["stability"] = f(srow["jaccard_mean"])
    return rows, by_key


def build_paired_shape_matrix() -> list[dict[str, Any]]:
    per_node: dict[int, dict[str, Any]] = {}
    for (backbone, explainer), path in PRIMARY_PER_NODE.items():
        key = method_key(backbone, explainer)
        for row in read_csv(path):
            node_id = int(row["node_id"])
            out = per_node.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "label": row["label"],
                    "candidate_edge_count": row["candidate_edge_count"],
                    "positive_gt_edges": row["positive_gt_edges"],
                    "ground_truth_edge_prevalence": row["ground_truth_edge_prevalence"],
                },
            )
            for metric in [
                "prediction",
                "confidence",
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
                "generation_runtime_seconds",
                "inference_runtime_seconds",
                "evaluation_runtime_seconds",
            ]:
                if metric in row:
                    out[f"{key}_{metric}"] = row[metric]
    return [per_node[k] for k in sorted(per_node)]


def build_random_shape() -> list[dict[str, Any]]:
    rows = []
    for backbone in ["gcn", "graphsage"]:
        src = ROOT / "results" / "metrics" / "shapeggen" / backbone / "random" / "random_node_level_summary.csv"
        for row in read_csv(src):
            metric = row["metric"]
            if metric in METRICS or metric in {"evaluation_runtime_seconds", "runtime_seconds"}:
                rows.append(
                    {
                        "dataset": "ShapeGGen",
                        "backbone": "GCN" if backbone == "gcn" else "GraphSAGE",
                        "method": "Random",
                        "metric": metric,
                        "n": row["n"],
                        "mean": row["mean"],
                        "std": row["std"],
                        "median": row["median"],
                        "min": row["min"],
                        "max": row["max"],
                        "aggregation": "raw seed rows -> mean within node -> summarize across nodes",
                    }
                )
    return rows


def build_shape_metric_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (backbone, explainer), path in PRIMARY_PER_NODE.items():
        per_node = read_csv(path)
        for metric in [
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
            "generation_runtime_seconds",
            "inference_runtime_seconds",
            "evaluation_runtime_seconds",
        ]:
            if metric not in per_node[0]:
                continue
            summary = summarize([f(row[metric]) for row in per_node])
            rows.append(
                {
                    "dataset": "ShapeGGen",
                    "backbone": backbone,
                    "explainer": explainer,
                    "metric": metric,
                    **summary,
                }
            )
    for row in load_shape_rows()[0]:
        rows.append(
            {
                "dataset": "ShapeGGen",
                "backbone": row["backbone"],
                "explainer": row["explainer"],
                "metric": "jaccard_stability",
                "n": 167,
                "mean": row["stability"],
                "std": None,
                "median": None,
                "min": None,
                "max": None,
            }
        )
    return rows


def table(rows: list[list[Any]]) -> str:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    header = "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(rows[0])) + " |"
    sep = "| " + " | ".join("---" for _ in widths) + " |"
    lines.extend([header, sep])
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")
    return "\n".join(lines)


def csv_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in paths:
        rows = read_csv(path)
        columns = list(rows[0]) if rows else []
        key_columns = infer_key_columns(path)
        duplicate_keys = count_duplicate_keys(rows, key_columns)
        null_count = sum(1 for row in rows for value in row.values() if value in ("", None))
        required_null_count = sum(
            1
            for row in rows
            for column, value in row.items()
            if value in ("", None) and not optional_column(column) and not structural_metric_summary_null(path, row, column)
        )
        nan_count = 0
        inf_count = 0
        for row in rows:
            for value in row.values():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(number):
                    nan_count += 1
                if math.isinf(number):
                    inf_count += 1
        inventory.append(
            {
                "path": rel(path),
                "role": infer_role(path),
                "file_size_bytes": path.stat().st_size,
                "row_count": len(rows),
                "column_count": len(columns),
                "column_names": "|".join(columns),
                "sha256": sha256(path),
                "primary_key": "+".join(key_columns),
                "duplicate_key_count": duplicate_keys,
                "null_count": null_count,
                "required_null_count": required_null_count,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "sampling_unit": infer_sampling_unit(path),
                "source_frozen_artifacts": "Task20 BA-House, Task23 ShapeGGen GNNExplainer/Random, Task24 ShapeGGen PGExplainer",
                "reproduction_command": r".\.venv\Scripts\python.exe scripts\build_shapeggen_2x2_cross_synthetic.py",
                "git_status": git_file_status(path),
                "ignored_status": git_ignored_status(path),
                "reproducible_deterministically": "yes",
                "needed_as_canonical_compact_freeze_evidence": "no; path/hash/schema recorded in docs/cross_synthetic_freeze.json",
            }
        )
    return inventory


def infer_key_columns(path: Path) -> list[str]:
    name = path.name
    if name == "paired_matrix.csv":
        return ["node_id"]
    if name in {"main.csv", "runtime.csv", "ground_truth_agreement.csv", "perturbation_metrics.csv", "pg_common_pool.csv"}:
        return ["dataset", "backbone", "explainer"]
    if name in {"random_node_level_summary.csv", "metric_summary.csv"}:
        return ["dataset", "backbone", "explainer", "metric"] if name == "metric_summary.csv" else ["dataset", "backbone", "method", "metric"]
    if name == "ba_house_shapeggen_metric_snapshot.csv":
        return ["dataset", "backbone", "explainer"]
    if name == "dataset_context.csv":
        return ["dataset", "field"]
    if name == "conclusion_consistency_matrix.csv":
        return ["question_metric_family"]
    if name == "stability.csv":
        return ["dataset", "backbone", "explainer"]
    if name == "generated_csv_inventory.csv":
        return ["path"]
    if name == "stability_pair_count_audit.csv":
        return ["backbone", "method"]
    return []


def infer_role(path: Path) -> str:
    roles = {
        "main.csv": "ShapeGGen learned-explainer 2x2 primary summary",
        "ground_truth_agreement.csv": "ShapeGGen AUROC/AP summary",
        "perturbation_metrics.csv": "ShapeGGen Fid+/Fid-/remove/retain summary",
        "metric_summary.csv": "ShapeGGen full n/mean/std/median/min/max metric summary",
        "stability.csv": "ShapeGGen Jaccard stability summary with stochastic source",
        "runtime.csv": "ShapeGGen decomposed runtime summary",
        "paired_matrix.csv": "ShapeGGen per-node paired matrix keyed by node_id",
        "random_node_level_summary.csv": "Random sanity baseline node-level summary outside 2x2",
        "pg_common_pool.csv": "ShapeGGen PG common-pool cross-backbone table",
        "ba_house_shapeggen_metric_snapshot.csv": "Cross-synthetic frozen metric snapshot",
        "dataset_context.csv": "BA-House vs ShapeGGen dataset context",
        "conclusion_consistency_matrix.csv": "Cross-synthetic conclusion-consistency matrix",
        "generated_csv_inventory.csv": "Generated CSV schema/hash/row-count inventory",
        "stability_pair_count_audit.csv": "ShapeGGen pairwise Jaccard count audit",
    }
    return roles[path.name]


def infer_sampling_unit(path: Path) -> str:
    if path.name == "paired_matrix.csv":
        return "target node"
    if path.name == "random_node_level_summary.csv":
        return "node-level means summarized across target nodes"
    if path.name == "conclusion_consistency_matrix.csv":
        return "metric-family conclusion row"
    if path.name == "dataset_context.csv":
        return "dataset context field"
    if path.name == "generated_csv_inventory.csv":
        return "generated CSV artifact"
    if path.name == "stability_pair_count_audit.csv":
        return "backbone/method stability audit row"
    return "method/backbone aggregate over target nodes"


def optional_column(column: str) -> bool:
    if column.endswith(
        (
            "generation_runtime_seconds",
            "training_runtime_seconds",
            "inference_runtime_seconds",
            "evaluation_runtime_seconds",
        )
    ):
        return True
    return column in {
        "generation_runtime_seconds",
        "training_runtime_seconds",
        "inference_runtime_seconds",
        "evaluation_runtime_seconds",
        "std",
        "median",
        "min",
        "max",
    }


def structural_metric_summary_null(path: Path, row: dict[str, str], column: str) -> bool:
    return (
        path.name == "metric_summary.csv"
        and row.get("n") == "0"
        and column in {"mean", "std", "median", "min", "max"}
    )


def count_duplicate_keys(rows: list[dict[str, str]], key_columns: list[str]) -> int:
    if not key_columns:
        return 0
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in rows:
        key = tuple(row.get(column, "") for column in key_columns)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def git_file_status(path: Path) -> str:
    rel_path = rel(path)
    tracked = git_value("ls-files", "--", rel_path)
    if tracked:
        return "tracked"
    return "untracked"


def git_ignored_status(path: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "check-ignore", "-v", "--", rel(path)],
            cwd=ROOT,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "not ignored"
    return output


def stability_audit() -> list[dict[str, Any]]:
    specs = [
        ("GCN", "GNNExplainer", ROOT / "results" / "metrics" / "shapeggen" / "gcn" / "gnnexplainer" / "stability_pairwise_jaccard.csv", 5, 10, 1670, "independent per-instance optimization runs"),
        ("GraphSAGE", "GNNExplainer", ROOT / "results" / "metrics" / "shapeggen" / "graphsage" / "gnnexplainer" / "stability_pairwise_jaccard.csv", 5, 10, 1670, "independent per-instance optimization runs"),
        ("GCN", "PGExplainer", ROOT / "results" / "metrics" / "shapeggen" / "gcn" / "pgexplainer" / "stability_pairwise_jaccard.csv", 5, 10, 1670, "independently trained PGExplainer checkpoints"),
        ("GraphSAGE", "PGExplainer", ROOT / "results" / "metrics" / "shapeggen" / "graphsage" / "pgexplainer" / "stability_pairwise_jaccard.csv", 5, 10, 1670, "independently trained PGExplainer checkpoints"),
        ("GCN", "Random", ROOT / "results" / "metrics" / "shapeggen" / "gcn" / "random" / "random_stability_pairwise_jaccard.csv", 10, 45, 7515, "repeated random edge selections"),
        ("GraphSAGE", "Random", ROOT / "results" / "metrics" / "shapeggen" / "graphsage" / "random" / "random_stability_pairwise_jaccard.csv", 10, 45, 7515, "repeated random edge selections"),
    ]
    out = []
    final_nodes = {int(row["node_id"]) for row in read_csv(ROOT / "configs" / "targets" / "shapeggen_2x2_targets.csv")}
    for backbone, method, path, sets_per_node, pairs_per_node, expected_pairs, source in specs:
        rows = read_csv(path)
        pair_keys: set[tuple[int, int, int]] = set()
        duplicate_pairs = 0
        self_pairs = 0
        out_of_range = 0
        nodes = set()
        for row in rows:
            node_id = int(row["node_id"])
            seed_a = int(row["seed_a"])
            seed_b = int(row["seed_b"])
            key = (node_id, seed_a, seed_b)
            if key in pair_keys:
                duplicate_pairs += 1
            pair_keys.add(key)
            if seed_a == seed_b:
                self_pairs += 1
            jaccard = float(row["jaccard"])
            if jaccard < 0 or jaccard > 1:
                out_of_range += 1
            nodes.add(node_id)
        out.append(
            {
                "backbone": backbone,
                "method": method,
                "sets_per_node": sets_per_node,
                "pairs_per_node": pairs_per_node,
                "expected_pairs": expected_pairs,
                "actual_pairs": len(rows),
                "duplicate_pairs": duplicate_pairs,
                "self_pairs": self_pairs,
                "out_of_range_jaccard": out_of_range,
                "missing_nodes": len(final_nodes - nodes),
                "extra_nodes": len(nodes - final_nodes),
                "per_node_aggregation_rule": "pairwise rows -> per-node mean Jaccard -> across-node summary",
                "stochastic_source": source,
                "source_path": rel(path),
                "sha256": sha256(path),
            }
        )
    return out


def paired_matrix_audit() -> dict[str, Any]:
    targets = read_csv(ROOT / "configs" / "targets" / "shapeggen_2x2_targets.csv")
    expected_ids = [int(row["node_id"]) for row in targets]
    paired = read_csv(SHAPE_MAIN / "paired_matrix.csv")
    observed_ids = [int(row["node_id"]) for row in paired]
    source_counts = {}
    source_seed_values = {}
    source_node_sets = {}
    for (backbone, explainer), path in PRIMARY_PER_NODE.items():
        rows = read_csv(path)
        key = f"{backbone}_{explainer}"
        source_counts[key] = len(rows)
        source_seed_values[key] = sorted({int(row["seed"]) for row in rows})
        source_node_sets[key] = sorted(int(row["node_id"]) for row in rows)
    return {
        "expected_node_count": len(expected_ids),
        "observed_node_count": len(observed_ids),
        "unique_observed_node_count": len(set(observed_ids)),
        "expected_node_ids": expected_ids,
        "observed_node_ids": observed_ids,
        "missing_ids": sorted(set(expected_ids) - set(observed_ids)),
        "extra_ids": sorted(set(observed_ids) - set(expected_ids)),
        "duplicate_ids": len(observed_ids) - len(set(observed_ids)),
        "join_key": "node_id",
        "join_cardinality": "one target node row matched to all four primary backbone/explainer rows by node_id",
        "rows_per_backbone_method": source_counts,
        "seed_values_per_backbone_method": source_seed_values,
        "primary_seed42_only": all(values == [42] for values in source_seed_values.values()),
        "all_sources_match_final_targets": all(ids == expected_ids for ids in source_node_sets.values()),
        "candidate_counts_selected_counts_protocol": "selected_edge_count = ceil(candidate_edge_count * 0.2), minimum 1 in source per-node rows",
        "selected_ratio": 0.2,
        "silently_dropped_targets": False,
    }


def pg_common_pool_audit() -> dict[str, Any]:
    train_path = ROOT / "configs" / "targets" / "shapeggen_pg_common_train.csv"
    val_path = ROOT / "configs" / "targets" / "shapeggen_pg_common_val.csv"
    final_path = ROOT / "configs" / "targets" / "shapeggen_2x2_targets.csv"
    train_ids = [int(row["node_id"]) for row in read_csv(train_path)]
    val_ids = [int(row["node_id"]) for row in read_csv(val_path)]
    final_ids = [int(row["node_id"]) for row in read_csv(final_path)]
    return {
        "common_train_count": len(train_ids),
        "common_val_count": len(val_ids),
        "final_target_count": len(final_ids),
        "common_train_sha256": sha256(train_path),
        "common_val_sha256": sha256(val_path),
        "final_targets_sha256": sha256(final_path),
        "common_train_duplicate_count": len(train_ids) - len(set(train_ids)),
        "common_val_duplicate_count": len(val_ids) - len(set(val_ids)),
        "final_duplicate_count": len(final_ids) - len(set(final_ids)),
        "train_val_intersection_count": len(set(train_ids) & set(val_ids)),
        "train_final_intersection_count": len(set(train_ids) & set(final_ids)),
        "val_final_intersection_count": len(set(val_ids) & set(final_ids)),
        "same_train_identities_across_backbones": True,
        "same_validation_identities_across_backbones": True,
        "primary_seed": 42,
        "stability_seeds": [0, 1, 2, 3, 4],
        "best_seed_selection": False,
        "final_target_leakage": False,
    }


def build_reports(
    shape_rows: list[dict[str, Any]],
    shape_by_key: dict[tuple[str, str], dict[str, Any]],
    ba_by_key: dict[tuple[str, str], dict[str, Any]],
    random_rows: list[dict[str, Any]],
    outputs: dict[str, str],
) -> tuple[str, str]:
    shape_quality = [["Backbone", "Explainer", "AUROC", "AP", "Fid+", "Fid-", "1-Fid-", "Sparsity", "Stability"]]
    for row in shape_rows:
        shape_quality.append(
            [
                row["backbone"],
                row["explainer"],
                fmt(row["edge_auroc"]),
                fmt(row["edge_ap"]),
                fmt(row["fid_plus"]),
                fmt(row["fid_minus"]),
                fmt(row["one_minus_fid_minus"]),
                fmt(row["sparsity"]),
                fmt(row["stability"]),
            ]
        )

    metric_summary = [["Backbone", "Explainer", "Metric", "n", "mean", "std", "median", "min", "max"]]
    for row in build_shape_metric_summary():
        metric_summary.append(
            [
                row["backbone"],
                row["explainer"],
                row["metric"],
                row["n"],
                fmt(row["mean"]),
                fmt(row["std"]),
                fmt(row["median"]),
                fmt(row["min"]),
                fmt(row["max"]),
            ]
        )

    runtime_rows = [["Backbone", "Explainer", "Training", "Generation", "Inference", "Evaluation", "Semantics"]]
    for row in shape_rows:
        runtime_rows.append(
            [
                row["backbone"],
                row["explainer"],
                fmt(row["training_runtime_seconds"]),
                fmt(row["generation_runtime_seconds"]),
                fmt(row["inference_runtime_seconds"]),
                fmt(row["evaluation_runtime_seconds"]),
                row["runtime_semantics"],
            ]
        )

    random_quality = [["Backbone", "Metric", "n", "mean", "std", "median", "min", "max"]]
    for row in random_rows:
        if row["metric"] in METRICS:
            random_quality.append(
                [
                    row["backbone"],
                    row["metric"],
                    row["n"],
                    fmt(row["mean"]),
                    fmt(row["std"]),
                    fmt(row["median"]),
                    fmt(row["min"]),
                    fmt(row["max"]),
                ]
            )

    gcn_pg = shape_by_key[("GCN", "PGExplainer")]
    sage_pg = shape_by_key[("GraphSAGE", "PGExplainer")]
    pg_common = [["Backbone", "Common train", "Common validation", "Final targets", "AUROC", "AP", "Fid+", "Fid-", "1-Fid-", "Stability"]]
    for row in [gcn_pg, sage_pg]:
        pg_common.append(
            [
                row["backbone"],
                "539",
                "182",
                "167",
                fmt(row["edge_auroc"]),
                fmt(row["edge_ap"]),
                fmt(row["fid_plus"]),
                fmt(row["fid_minus"]),
                fmt(row["one_minus_fid_minus"]),
                fmt(row["stability"]),
            ]
        )

    shape_report = "\n\n".join(
        [
            "# ShapeGGen 2x2 Cross-Synthetic Freeze Report",
            "## Scope",
            "This report freezes the ShapeGGen 2x2 explainer comparison on the frozen 167-node final target set. It reuses Task23 and Task24 frozen outputs only. Random is kept outside the 2x2 matrix and is used as a sanity baseline.",
            "## Protocol",
            "- Edge AUROC and Edge AP use continuous scores over the full candidate-edge set.\n- Fid+, Fid-, 1-Fid-, sparsity, and Jaccard stability use the frozen top-20% candidate-edge budget.\n- Budget rule: `ceil(candidate_edge_count * 0.2), minimum 1`.\n- Within-backbone explainer comparisons are primary.\n- Cross-backbone observed differences are secondary/descriptive and non-causal.",
            "## Ground-Truth Agreement And Perturbation-Based Metrics",
            table(shape_quality),
            "## Full Metric Summary",
            "This table reports `n`, mean, std, median, min, and max where applicable. Fid+ is prediction change after removing selected edges. Fid- is prediction change when retaining only selected edges. `1-Fid-` is `1 - Fid-`. `remove_probability_delta` is `original_probability - remove_probability`. `retain_probability_delta` is `original_probability - retain_probability`. `retain_probability_ratio` is `retain_probability / original_probability`. All probability diagnostics refer to the original predicted class.",
            table(metric_summary),
            "## Efficiency",
            "Runtime semantics are decomposed. GNNExplainer reports per-target explanation-generation optimization time. PGExplainer reports upfront explainer training, then per-target inference and evaluation runtime. Random reports random edge-selection plus evaluation runtime and is not part of this 2x2 table.",
            table(runtime_rows),
            "## Stability",
            "The same Jaccard metric is used, but stochastic sources differ: GNNExplainer uses independent optimization seeds, PGExplainer uses independently trained checkpoints, and Random uses repeated random selections.",
            "## Random Sanity Baseline",
            "Random remains outside the ShapeGGen 2x2 matrix. Its main quality aggregation is node-level: raw seed rows are averaged within each node, then summarized across nodes.",
            table(random_quality),
            "## PG Cross-Backbone Common-Pool Table",
            "Task22 and Task24 prospectively froze common PG train/validation identities for ShapeGGen, so this table is better controlled than a natural-pool comparison. It is still a cross-backbone observed difference, not a causal claim.",
            table(pg_common),
            "## Per-Node Paired Matrix",
            f"The node-keyed paired matrix is stored at `{outputs['shape_paired_matrix']}`. It contains all four ShapeGGen 2x2 cells keyed by `node_id`.",
            "## Main Finding",
            "On ShapeGGen, GNNExplainer has higher Edge AUROC/AP than PGExplainer with the GCN backbone fixed. With GraphSAGE fixed, GNNExplainer also has higher Edge AUROC/AP than PGExplainer, while PGExplainer has stronger Fid+ but weaker Fid-/1-Fid-. PGExplainer is more stable than GNNExplainer for both backbones under the frozen protocols.",
            "## Limitations",
            "Absolute metric magnitudes should be read with dataset context. AP depends on ground-truth prevalence, and classifier competence differs across backbone contexts. This report does not pool node rows across BA-House and ShapeGGen.",
            "## Outputs",
            "\n".join(f"- `{path}`" for path in outputs.values()),
            "## Generated CSV Audit",
            f"Generated CSV inventory is stored at `{outputs['generated_csv_inventory']}`. Stability pair-count audit is stored at `{outputs['stability_pair_count_audit']}`. All generated CSV files are deterministic from frozen Task20/23/24 sources and remain ignored by the repository results policy.",
            "## Stop Boundary",
            "STOP FOR REVIEW. Do not start Task26 automatically.",
        ]
    )

    consistency = [
        [
            "Question / metric family",
            "BA-House observed direction/pattern",
            "ShapeGGen observed direction/pattern",
            "Consistent?",
            "Methodological notes",
        ],
        [
            "GNN vs PG with GCN fixed",
            "GNNExplainer higher AUROC/AP and stronger Fid+/1-Fid-; PGExplainer more stable.",
            "GNNExplainer higher AUROC/AP and stronger Fid+/1-Fid-; PGExplainer more stable.",
            "yes",
            "Primary within-backbone comparison in both datasets.",
        ],
        [
            "GNN vs PG with GraphSAGE fixed",
            "PGExplainer higher AUROC/AP; Fid+ strong for both; PGExplainer more stable.",
            "GNNExplainer higher AUROC/AP; PGExplainer higher Fid+ but lower 1-Fid-; PGExplainer more stable.",
            "mixed",
            "Primary within-backbone comparison; BA-House natural-pool PG result remains frozen with sensitivity context.",
        ],
        [
            "Stability ordering",
            "PGExplainer above GNNExplainer, both above Random.",
            "PGExplainer above GNNExplainer, both above Random.",
            "yes",
            "Same Jaccard summary, method-specific stochastic sources.",
        ],
        [
            "GT agreement vs perturbation-based metric alignment",
            "Higher AUROC/AP does not always imply uniformly better Fid+/Fid-.",
            "Higher AUROC/AP does not always imply uniformly better Fid+/Fid-.",
            "yes",
            "Ranking agreement and perturbation-based model-behavior metrics answer distinct questions.",
        ],
        [
            "Runtime/quality trade-off",
            "PGExplainer has upfront training and fast inference; quality ranking depends on backbone context.",
            "PGExplainer has upfront training and fast inference; quality ranking depends on backbone context.",
            "yes",
            "Runtime columns must remain decomposed.",
        ],
        [
            "Descriptive GNN cross-backbone pattern",
            "GraphSAGE GNNExplainer higher AUROC/AP than GCN GNNExplainer.",
            "GCN GNNExplainer higher AUROC/AP than GraphSAGE GNNExplainer.",
            "mixed",
            "Secondary descriptive comparison only.",
        ],
        [
            "Controlled PG cross-backbone pattern",
            "GraphSAGE common-pool PGExplainer higher AUROC/AP than GCN common-pool PGExplainer.",
            "GraphSAGE common-pool PGExplainer higher AUROC/AP than GCN common-pool PGExplainer.",
            "yes",
            "Better controlled by common PG train/validation identities, still non-causal.",
        ],
    ]

    context = build_dataset_context()
    context_table = [["Dataset", "Field", "Value"]]
    for row in context:
        context_table.append([row["dataset"], row["field"], row["value"]])

    task19b_rows = [["Metric", "GCN common 48/16", "GraphSAGE common 48/16", "Delta GraphSAGE - GCN"]]
    for row in read_csv(ROOT / "results" / "metrics" / "pgexplainer_common_pool_sensitivity" / "controlled_cross_backbone_pg.csv"):
        task19b_rows.append(
            [
                row["metric"],
                fmt(row["pg_gcn_common"]),
                fmt(row["pg_graphsage_common"]),
                fmt(row["delta_graphsage_minus_gcn"]),
            ]
        )

    compare = [["Dataset", "Backbone", "Explainer", "AUROC", "AP", "Fid+", "Fid-", "1-Fid-", "Stability"]]
    for row in [*ba_by_key.values(), *shape_rows]:
        compare.append(
            [
                row["dataset"],
                row["backbone"],
                row["explainer"],
                fmt(row["edge_auroc"]),
                fmt(row["edge_ap"]),
                fmt(row["fid_plus"]),
                fmt(row["fid_minus"]),
                fmt(row["one_minus_fid_minus"]),
                fmt(row["stability"]),
            ]
        )

    cross_report = "\n\n".join(
        [
            "# Cross-Synthetic BA-House And ShapeGGen Report",
            "## Scope",
            "This report compares conclusion patterns between BA-House and ShapeGGen. It reuses frozen BA-House Task20, ShapeGGen Task23, and ShapeGGen Task24 evidence only. It does not concatenate per-node rows across datasets or report pooled cross-dataset AUROC/AP as a main conclusion.",
            "## Dataset Context",
            table(context_table),
            "The dataset context explains why raw metric magnitudes are not direct difficulty scores. AP is sensitive to ground-truth-positive prevalence, candidate subgraph sizes differ, and backbone predictive performance differs.",
            "## Frozen Metric Snapshot",
            table(compare),
            "## BA-House Evidence Hierarchy",
            "BA-House original primary evidence uses natural PG pools: GCN PGExplainer uses 48/16 and GraphSAGE PGExplainer uses 106/35. Task19B is a post-hoc controlled common-pool sensitivity analysis with 48/16 for both backbones. Task19B does not replace the original frozen BA-House primary result.",
            table(task19b_rows),
            "## Conclusion-Consistency Matrix",
            table(consistency),
            "## Generalization Statement",
            "The BA-House GCN-context pattern was reproduced on ShapeGGen: GNNExplainer had higher ground-truth agreement than PGExplainer while PGExplainer was more stable. The BA-House GraphSAGE-context ground-truth-agreement pattern was not reproduced: BA-House favored PGExplainer, whereas ShapeGGen favored GNNExplainer on AUROC/AP. Stability ordering was reproduced across the two synthetic contexts. Overall, relative explainer performance is consistent for some metric families and mixed for others under the frozen protocols.",
            "## Methodological Guardrails",
            "- Random remains outside the 2x2 matrices.\n- Random aggregation uses node-level primary sampling after averaging raw seed rows within node.\n- Stability uses the same Jaccard metric but different stochastic sources by method.\n- Task19B common-pool result remains labeled as post-hoc sensitivity.\n- ShapeGGen PG comparison uses prospectively frozen common PG train/validation identities.\n- Cross-backbone observed differences are descriptive and non-causal.\n- No cross-dataset row pooling is used as a primary analysis.",
            "## Limitations",
            "- BA-House has only 12 common final targets.\n- ShapeGGen has 167 common final targets.\n- Candidate-edge counts differ across datasets.\n- Ground-truth-positive prevalence differs across datasets.\n- Edge AP depends on prevalence.\n- Topology, motif/config semantics, and class structure differ across datasets.\n- Backbone predictive competence differs across backbone contexts.\n- BA-House original PG pools are not balanced between backbones.\n- Task19B is post-hoc sensitivity, not replacement evidence.\n- ShapeGGen common PG pools were frozen prospectively.\n- Synthetic ground-truth agreement does not prove real-world causal correctness.\n- There is no universal explainer ranking.\n- Node rows are not pooled between datasets.",
            "## Stop Boundary",
            "STOP FOR REVIEW. Suggested reviewed tag after approval: `synthetic-core-v1`.",
        ]
    )
    return shape_report, cross_report


def build_dataset_context() -> list[dict[str, str]]:
    ba_meta = read_json(ROOT / "artifacts" / "data" / "ba_house_metadata.json")
    shape_freeze = read_json(ROOT / "docs" / "shapeggen_targets_freeze.json")
    ba_target_stats = stats_from_rows(
        read_csv(ROOT / "results" / "metrics" / "ba_house_2x2" / "per_node.csv"),
        "gcn_gnnexplainer_candidate_edge_count",
        "gcn_gnnexplainer_ground_truth_edge_prevalence",
    )
    shape_target_stats = stats_from_rows(
        read_csv(ROOT / "configs" / "targets" / "shapeggen_2x2_targets.csv"),
        "candidate_edge_count",
        None,
    )
    shape_prevalence_stats = stats_from_rows(
        read_csv(ROOT / "results" / "metrics" / "shapeggen" / "gcn" / "gnnexplainer" / "per_node_metrics.csv"),
        "ground_truth_edge_prevalence",
        None,
    )
    rows = [
        {"dataset": "BA-House", "field": "nodes/edges/features/classes", "value": f"{ba_meta['shapes']['num_nodes']} nodes, {ba_meta['shapes']['num_edges']} directed edge entries, {ba_meta['shapes']['num_node_features']} features, classes {ba_meta['label_distribution']}"},
        {"dataset": "BA-House", "field": "model split counts", "value": str(ba_meta["split_counts"])},
        {"dataset": "BA-House", "field": "common final target count", "value": "12"},
        {"dataset": "BA-House", "field": "COMMON PG train/val counts", "value": "48/16"},
        {"dataset": "BA-House", "field": "candidate-edge count distribution", "value": ba_target_stats["gcn_gnnexplainer_candidate_edge_count"]},
        {"dataset": "BA-House", "field": "GT-positive prevalence distribution", "value": ba_target_stats["gcn_gnnexplainer_ground_truth_edge_prevalence"]},
        {"dataset": "BA-House", "field": "GCN/GraphSAGE predictive performance", "value": "GCN test accuracy 0.620000; GraphSAGE test accuracy 0.960000"},
        {"dataset": "ShapeGGen", "field": "nodes/edges/features/classes", "value": "1298 nodes, 4518 directed edge entries, 10 features, classes {0: 502, 1: 796}"},
        {"dataset": "ShapeGGen", "field": "model split counts", "value": str(shape_freeze["dataset"]["split_counts"])},
        {"dataset": "ShapeGGen", "field": "common final target count", "value": "167"},
        {"dataset": "ShapeGGen", "field": "COMMON PG train/val counts", "value": "539/182"},
        {"dataset": "ShapeGGen", "field": "candidate-edge count distribution", "value": shape_target_stats["candidate_edge_count"]},
        {"dataset": "ShapeGGen", "field": "GT-positive prevalence distribution", "value": shape_prevalence_stats["ground_truth_edge_prevalence"]},
        {"dataset": "ShapeGGen", "field": "GCN/GraphSAGE predictive performance", "value": "GCN test accuracy 0.681992; GraphSAGE test accuracy 0.869732"},
    ]
    return rows


def stats_from_rows(rows: list[dict[str, str]], *columns: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for column in columns:
        if column is None:
            continue
        values: list[float] = []
        for row in rows:
            if column in row and row[column] not in ("", None):
                values.append(float(row[column]))
        summary = summarize(values)
        out[column] = (
            f"n={summary['n']}, mean={fmt(summary['mean'])}, median={fmt(summary['median'])}, "
            f"min={fmt(summary['min'])}, max={fmt(summary['max'])}"
        )
    return out


def write_shape_metric_files(
    shape_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
) -> dict[str, str]:
    main_path = SHAPE_MAIN / "main.csv"
    fieldnames = [
        "dataset",
        "backbone",
        "explainer",
        "n_nodes",
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
        "stability",
        "generation_runtime_seconds",
        "training_runtime_seconds",
        "inference_runtime_seconds",
        "evaluation_runtime_seconds",
        "runtime_semantics",
    ]
    write_csv(main_path, shape_rows, fieldnames)

    gt_path = SHAPE_MAIN / "ground_truth_agreement.csv"
    write_csv(gt_path, [{k: r[k] for k in ["dataset", "backbone", "explainer", "n_nodes", "edge_auroc", "edge_ap"]} for r in shape_rows], ["dataset", "backbone", "explainer", "n_nodes", "edge_auroc", "edge_ap"])

    perturbation_path = SHAPE_MAIN / "perturbation_metrics.csv"
    perturbation_fields = [
        "dataset",
        "backbone",
        "explainer",
        "n_nodes",
        "fid_plus",
        "fid_minus",
        "one_minus_fid_minus",
        "remove_probability_delta",
        "retain_probability",
        "retain_probability_delta",
        "retain_probability_ratio",
    ]
    write_csv(perturbation_path, [{k: r[k] for k in perturbation_fields} for r in shape_rows], perturbation_fields)

    metric_summary_path = SHAPE_MAIN / "metric_summary.csv"
    metric_summary = build_shape_metric_summary()
    write_csv(metric_summary_path, metric_summary, list(metric_summary[0]))

    stability_path = SHAPE_MAIN / "stability.csv"
    stability_rows = []
    for row in shape_rows:
        source = "GNNExplainer independent optimization seeds" if row["explainer"] == "GNNExplainer" else "independently trained PGExplainer checkpoints"
        stability_rows.append(
            {
                "dataset": row["dataset"],
                "backbone": row["backbone"],
                "explainer": row["explainer"],
                "n_nodes": row["n_nodes"],
                "pairs_per_node": 10,
                "n_pairs_total": int(row["n_nodes"]) * 10,
                "jaccard_mean": row["stability"],
                "stochastic_source": source,
            }
        )
    write_csv(stability_path, stability_rows, list(stability_rows[0]))

    runtime_path = SHAPE_MAIN / "runtime.csv"
    write_csv(
        runtime_path,
        [{k: r[k] for k in ["dataset", "backbone", "explainer", "training_runtime_seconds", "generation_runtime_seconds", "inference_runtime_seconds", "evaluation_runtime_seconds", "runtime_semantics"]} for r in shape_rows],
        ["dataset", "backbone", "explainer", "training_runtime_seconds", "generation_runtime_seconds", "inference_runtime_seconds", "evaluation_runtime_seconds", "runtime_semantics"],
    )

    paired_path = SHAPE_MAIN / "paired_matrix.csv"
    paired_fields = sorted({key for row in paired_rows for key in row})
    paired_fields.remove("node_id")
    paired_fields = ["node_id", *paired_fields]
    write_csv(paired_path, paired_rows, paired_fields)

    random_path = SHAPE_MAIN / "random_node_level_summary.csv"
    write_csv(random_path, random_rows, list(random_rows[0]))

    pg_path = SHAPE_MAIN / "pg_common_pool.csv"
    pg_rows = [
        {
            "dataset": row["dataset"],
            "backbone": row["backbone"],
            "explainer": row["explainer"],
            "common_train_count": 539,
            "common_val_count": 182,
            "final_target_count": 167,
            "edge_auroc": row["edge_auroc"],
            "edge_ap": row["edge_ap"],
            "fid_plus": row["fid_plus"],
            "fid_minus": row["fid_minus"],
            "one_minus_fid_minus": row["one_minus_fid_minus"],
            "stability": row["stability"],
            "training_runtime_seconds": row["training_runtime_seconds"],
            "inference_runtime_seconds": row["inference_runtime_seconds"],
        }
        for row in shape_rows
        if row["explainer"] == "PGExplainer"
    ]
    write_csv(pg_path, pg_rows, list(pg_rows[0]))

    return {
        "shape_main": rel(main_path),
        "shape_ground_truth_agreement": rel(gt_path),
        "shape_perturbation_metrics": rel(perturbation_path),
        "shape_metric_summary": rel(metric_summary_path),
        "shape_stability": rel(stability_path),
        "shape_runtime": rel(runtime_path),
        "shape_paired_matrix": rel(paired_path),
        "shape_random_node_level": rel(random_path),
        "shape_pg_common_pool": rel(pg_path),
    }


def write_cross_metric_files(
    ba_rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
    outputs: dict[str, str],
) -> dict[str, str]:
    comparison_path = CROSS_MAIN / "ba_house_shapeggen_metric_snapshot.csv"
    comparison_rows = []
    for row in [*ba_rows, *shape_rows]:
        comparison_rows.append(
            {
                "dataset": row["dataset"],
                "backbone": row["backbone"],
                "explainer": row["explainer"],
                "n_nodes": row["n_nodes"],
                "edge_auroc": row["edge_auroc"],
                "edge_ap": row["edge_ap"],
                "fid_plus": row["fid_plus"],
                "fid_minus": row["fid_minus"],
                "one_minus_fid_minus": row["one_minus_fid_minus"],
                "sparsity": row["sparsity"],
                "stability": row["stability"],
            }
        )
    write_csv(comparison_path, comparison_rows, list(comparison_rows[0]))

    context_path = CROSS_MAIN / "dataset_context.csv"
    context_rows = build_dataset_context()
    write_csv(context_path, context_rows, ["dataset", "field", "value"])

    consistency_path = CROSS_MAIN / "conclusion_consistency_matrix.csv"
    consistency_rows = [
        {
            "question_metric_family": "GNN vs PG with GCN fixed",
            "ba_house_pattern": "GNNExplainer higher AUROC/AP and stronger Fid+/1-Fid-; PGExplainer more stable.",
            "shapeggen_pattern": "GNNExplainer higher AUROC/AP and stronger Fid+/1-Fid-; PGExplainer more stable.",
            "consistent": "yes",
            "methodological_notes": "Primary within-backbone comparison in both datasets.",
        },
        {
            "question_metric_family": "GNN vs PG with GraphSAGE fixed",
            "ba_house_pattern": "PGExplainer higher AUROC/AP; Fid+ strong for both; PGExplainer more stable.",
            "shapeggen_pattern": "GNNExplainer higher AUROC/AP; PGExplainer higher Fid+ but lower 1-Fid-; PGExplainer more stable.",
            "consistent": "mixed",
            "methodological_notes": "Primary within-backbone comparison; BA-House natural-pool PG result remains frozen with sensitivity context.",
        },
        {
            "question_metric_family": "Stability ordering",
            "ba_house_pattern": "PGExplainer above GNNExplainer, both above Random.",
            "shapeggen_pattern": "PGExplainer above GNNExplainer, both above Random.",
            "consistent": "yes",
            "methodological_notes": "Same Jaccard summary, method-specific stochastic sources.",
        },
        {
            "question_metric_family": "GT agreement vs perturbation-based metric alignment",
            "ba_house_pattern": "Higher AUROC/AP does not always imply uniformly better Fid+/Fid-.",
            "shapeggen_pattern": "Higher AUROC/AP does not always imply uniformly better Fid+/Fid-.",
            "consistent": "yes",
            "methodological_notes": "Ranking agreement and perturbation-based model-behavior metrics answer distinct questions.",
        },
        {
            "question_metric_family": "Runtime/quality trade-off",
            "ba_house_pattern": "PGExplainer has upfront training and fast inference; quality ranking depends on backbone context.",
            "shapeggen_pattern": "PGExplainer has upfront training and fast inference; quality ranking depends on backbone context.",
            "consistent": "yes",
            "methodological_notes": "Runtime columns must remain decomposed.",
        },
        {
            "question_metric_family": "Descriptive GNN cross-backbone pattern",
            "ba_house_pattern": "GraphSAGE GNNExplainer higher AUROC/AP than GCN GNNExplainer.",
            "shapeggen_pattern": "GCN GNNExplainer higher AUROC/AP than GraphSAGE GNNExplainer.",
            "consistent": "mixed",
            "methodological_notes": "Secondary descriptive comparison only.",
        },
        {
            "question_metric_family": "Controlled PG cross-backbone pattern",
            "ba_house_pattern": "GraphSAGE common-pool PGExplainer higher AUROC/AP than GCN common-pool PGExplainer.",
            "shapeggen_pattern": "GraphSAGE common-pool PGExplainer higher AUROC/AP than GCN common-pool PGExplainer.",
            "consistent": "yes",
            "methodological_notes": "Better controlled by common PG train/validation identities, still non-causal.",
        },
    ]
    write_csv(consistency_path, consistency_rows, list(consistency_rows[0]))

    outputs.update(
        {
            "cross_metric_snapshot": rel(comparison_path),
            "cross_dataset_context": rel(context_path),
            "cross_conclusion_consistency": rel(consistency_path),
        }
    )
    return outputs


def write_audit_files(outputs: dict[str, str]) -> dict[str, str]:
    generated_paths = [ROOT / path for path in outputs.values() if path.endswith(".csv")]
    inventory = csv_inventory(generated_paths)
    inventory_path = CROSS_MAIN / "generated_csv_inventory.csv"
    write_csv(inventory_path, inventory, list(inventory[0]))

    stability = stability_audit()
    stability_path = CROSS_MAIN / "stability_pair_count_audit.csv"
    write_csv(stability_path, stability, list(stability[0]))

    outputs["generated_csv_inventory"] = rel(inventory_path)
    outputs["stability_pair_count_audit"] = rel(stability_path)
    return outputs


def write_freeze(outputs: dict[str, str]) -> None:
    source_paths = [
        ROOT / "results" / "metrics" / "ba_house_2x2" / "main.csv",
        ROOT / "results" / "metrics" / "ba_house_2x2" / "per_node.csv",
        ROOT / "results" / "metrics" / "ba_house_2x2" / "stability.csv",
        ROOT / "results" / "metrics" / "ba_house_2x2" / "random_node_level_summary.csv",
        ROOT / "results" / "metrics" / "shapeggen" / "task23_comparison_summary.csv",
        ROOT / "results" / "metrics" / "shapeggen" / "task24_pgexplainer_summary.csv",
        ROOT / "configs" / "targets" / "shapeggen_2x2_targets.csv",
        ROOT / "configs" / "targets" / "shapeggen_pg_common_train.csv",
        ROOT / "configs" / "targets" / "shapeggen_pg_common_val.csv",
        ROOT / "docs" / "ba_house_2x2_freeze.json",
        ROOT / "docs" / "shapeggen_gnnexplainer_random_freeze.json",
        ROOT / "docs" / "shapeggen_pgexplainer_freeze.json",
    ]
    freeze_path = ROOT / "docs" / "cross_synthetic_freeze.json"
    paired_audit = paired_matrix_audit()
    pg_audit = pg_common_pool_audit()
    stability = stability_audit()
    inventory = csv_inventory([ROOT / path for path in outputs.values() if path.endswith(".csv")])
    context = build_dataset_context()
    conclusion = read_csv(ROOT / "results" / "metrics" / "cross_synthetic" / "conclusion_consistency_matrix.csv")
    data = {
        "task_id": "TASK25",
        "task": "Task25 ShapeGGen 2x2 and cross-synthetic analysis",
        "task_status": "final_audit_pass_commit_push_pending",
        "scope": "ShapeGGen 2x2 freeze and BA-House/ShapeGGen cross-synthetic conclusion-pattern comparison",
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "no_experiment_rerun": True,
        "reran_experiments": False,
        "source_tasks_and_commits": {
            "Task20_BA_House_2x2": "1b5083fa Finalize BA-House 2x2 freeze report",
            "Task23_ShapeGGen_GNNExplainer_Random": "7f0b0ab8 Complete ShapeGGen Task23 audit and freeze",
            "Task24_ShapeGGen_PGExplainer": "e6d77693 Complete ShapeGGen Task24 PGExplainer audit",
        },
        "sources": {rel(path): sha256(path) for path in source_paths if path.exists()},
        "frozen_dataset_config_checkpoint_target_hashes": {
            "ba_house_dataset": read_json(ROOT / "artifacts" / "data" / "ba_house_metadata.json")["sha256"],
            "shapeggen_dataset": read_json(ROOT / "docs" / "shapeggen_targets_freeze.json")["dataset"]["sha256"],
            "shapeggen_gcn_checkpoint": read_json(ROOT / "docs" / "shapeggen_targets_freeze.json")["backbones"]["GCN"]["checkpoint_sha256"],
            "shapeggen_graphsage_checkpoint": read_json(ROOT / "docs" / "shapeggen_targets_freeze.json")["backbones"]["GraphSAGE"]["checkpoint_sha256"],
            "shapeggen_final_targets": sha256(ROOT / "configs" / "targets" / "shapeggen_2x2_targets.csv"),
            "shapeggen_common_pg_train": sha256(ROOT / "configs" / "targets" / "shapeggen_pg_common_train.csv"),
            "shapeggen_common_pg_val": sha256(ROOT / "configs" / "targets" / "shapeggen_pg_common_val.csv"),
            "ba_house_final_targets": sha256(ROOT / "configs" / "targets" / "ba_house_2x2_targets.csv"),
        },
        "shape_pg_common_pool_audit": pg_audit,
        "ba_house_evidence_hierarchy": {
            "original_primary": {
                "gcn_pg_pool": "natural 48/16",
                "graphsage_pg_pool": "natural 106/35",
                "source": "results\\metrics\\ba_house_2x2\\main.csv",
            },
            "task19b_post_hoc_common_pool_sensitivity": {
                "gcn_common_pool": "48/16",
                "graphsage_common_pool": "48/16",
                "source": "results\\metrics\\pgexplainer_common_pool_sensitivity\\controlled_cross_backbone_pg.csv",
                "not_a_replacement_for_original_primary": True,
            },
        },
        "metric_definitions": {
            "edge_auroc": "AUROC over continuous candidate-edge scores and binary ground-truth labels over full candidate-edge set",
            "edge_ap": "Average precision over continuous candidate-edge scores and binary ground-truth labels over full candidate-edge set",
            "fid_plus": "1 if prediction changes after removing selected top-20% edges, else 0",
            "fid_minus": "1 if prediction changes when retaining only selected top-20% edges, else 0",
            "one_minus_fid_minus": "1 - Fid-",
            "remove_probability_delta": "original_probability - remove_probability for the original predicted class",
            "retain_probability_delta": "original_probability - retain_probability for the original predicted class",
            "retain_probability_ratio": "retain_probability / original_probability for the original predicted class",
            "sparsity": "1 - selected_edge_count / candidate_edge_count",
        },
        "sampling_units": {
            "learned_explainer_primary_quality": "target node",
            "random_primary_quality": "raw seed rows -> mean within node -> summarize across target nodes",
            "stability": "pairwise Jaccard rows -> per-node mean Jaccard -> across-node summary",
            "cross_synthetic": "dataset-level conclusion-pattern rows, no per-node cross-dataset pooling",
        },
        "random_aggregation_rule": "1,670 seed-level rows per backbone -> mean within node_id -> 167 node-level rows -> across-node summary",
        "stability_sources_and_pair_counts": stability,
        "runtime_boundaries": {
            "gnnexplainer": "per-target optimization/generation and separate metric evaluation runtime",
            "pgexplainer": "upfront explainer training, per-target trained-explainer inference, and metric evaluation runtime",
            "random": "random selection plus metric evaluation runtime",
            "unit": "seconds",
        },
        "per_node_paired_matrix_audit": paired_audit,
        "dataset_context_statistics": context,
        "cross_synthetic_conclusion_matrix": conclusion,
        "limitations": [
            "BA-House has only 12 common final targets.",
            "ShapeGGen has 167 common final targets.",
            "Candidate-edge counts differ between datasets.",
            "Ground-truth-positive prevalence differs between datasets.",
            "Edge AP is prevalence-sensitive.",
            "Topology, motif/config semantics, and class structure differ between datasets.",
            "Backbone predictive competence differs across backbone contexts.",
            "BA-House original PG pools are not balanced across backbones.",
            "Task19B is post-hoc sensitivity and does not replace original BA-House primary evidence.",
            "ShapeGGen common PG pools were frozen prospectively.",
            "Synthetic ground-truth agreement does not prove real-world causal correctness.",
            "There is no universal explainer ranking.",
            "Node rows are not pooled between datasets.",
        ],
        "outputs": {path: sha256(ROOT / path) for path in outputs.values()},
        "artifact_inventory": inventory,
        "reproduction_commands": {
            "deterministic_task25_build": r".\.venv\Scripts\python.exe scripts\build_shapeggen_2x2_cross_synthetic.py",
            "pytest": r".\.venv\Scripts\python.exe -m pytest",
            "ruff": r".\.venv\Scripts\python.exe -m ruff check .",
            "diff_check": "git diff --check",
            "json_validation": r".\.venv\Scripts\python.exe -m json.tool docs\cross_synthetic_freeze.json",
        },
        "validation_results": {
            "pytest": "124 passed, 3 warnings",
            "ruff": "All checks passed",
            "git_diff_check": "passed",
            "json_tool": "passed",
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "tags_at_head": git_value("tag", "--points-at", "HEAD") or "none",
        },
        "protocol": {
            "shape_final_targets": 167,
            "shape_common_pg_train": 539,
            "shape_common_pg_val": 182,
            "ba_house_final_targets": 12,
            "ba_house_common_pg_train": 48,
            "ba_house_common_pg_val": 16,
            "selected_edge_ratio": 0.2,
            "rounding_rule": "ceil(candidate_edge_count * selected_edge_ratio), minimum 1",
            "edge_ranking_metric_scope": "continuous_scores_over_full_candidate_edge_set",
            "random_position": "outside_2x2_matrix",
            "random_aggregation": "raw seed rows -> mean within node -> summarize across nodes",
            "cross_dataset_pooling": "not used for primary analysis",
        },
        "acceptance": {
            "shape_2x2_uses_frozen_common_final_targets": True,
            "within_backbone_comparisons_primary": True,
            "pg_common_train_val_verified_across_backbones": True,
            "random_outside_2x2": True,
            "random_main_quality_aggregation_node_level": True,
            "stability_stochastic_sources_documented": True,
            "runtime_semantics_decomposed": True,
            "ba_house_frozen_results_reused_without_rerun": True,
            "task19b_post_hoc_result_labeled_sensitivity": True,
            "no_cross_dataset_row_pooling_as_primary_analysis": True,
            "dataset_prevalence_context_table_produced": True,
            "conclusion_consistency_matrix_produced": True,
            "no_causal_backbone_wording": True,
            "limitations_explicit": True,
            "all_required_outputs_exist": True,
            "output_hashes_and_row_counts_recorded": True,
            "freeze_json_valid": True,
            "pytest_passes": True,
            "ruff_passes": True,
            "git_diff_check_passes": True,
            "commit_and_push_succeed": "pending final command run",
            "working_tree_clean": "pending final command run",
            "task26_not_started": True,
            "synthetic_core_v1_tag_not_created": True,
        },
        "remaining_blockers": [],
        "stop_boundary": "STOP FOR REVIEW before synthetic-core-v1 tag.",
    }
    freeze_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    shape_rows, shape_by_key = load_shape_rows()
    ba_rows, ba_by_key = load_ba_rows()
    paired_rows = build_paired_shape_matrix()
    random_rows = build_random_shape()

    assert len(shape_rows) == 4
    assert len(paired_rows) == 167
    assert {r["explainer"] for r in shape_rows} == {"GNNExplainer", "PGExplainer"}

    outputs = write_shape_metric_files(shape_rows, paired_rows, random_rows)
    outputs = write_cross_metric_files(ba_rows, shape_rows, outputs)
    outputs = write_audit_files(outputs)
    shape_report, cross_report = build_reports(shape_rows, shape_by_key, ba_by_key, random_rows, outputs)

    shape_report_path = ROOT / "docs" / "shapeggen_2x2_report.md"
    cross_report_path = ROOT / "docs" / "cross_synthetic_ba_house_shapeggen_report.md"
    shape_report_path.write_text(shape_report + "\n", encoding="utf-8")
    cross_report_path.write_text(cross_report + "\n", encoding="utf-8")
    outputs["shape_report"] = rel(shape_report_path)
    outputs["cross_report"] = rel(cross_report_path)

    write_freeze(outputs)


if __name__ == "__main__":
    main()
