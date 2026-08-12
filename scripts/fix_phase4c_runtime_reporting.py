from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import (
    PER_NODE_FIELDS,
    SUMMARY_METRICS,
    validate_fidelity_alias,
    validate_runtime_reporting,
)

DEFAULT_PAIRS = [
    (
        Path("results/metrics/phase4c_pilot"),
        Path("results/explanations/phase4c_pilot"),
    ),
    (
        Path("results/metrics/frozen_explainer_test_ratio20"),
        Path("results/explanations/frozen_explainer_test_ratio20"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix Phase 4C runtime reporting without reruns.")
    parser.add_argument("--metrics-dir", action="append", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = DEFAULT_PAIRS
    if args.metrics_dir:
        pairs = [(path, explanation_dir_for_metrics(path)) for path in args.metrics_dir]
    for metrics_dir, explanations_dir in pairs:
        rows = update_per_node(metrics_dir / "per_node_metrics.csv", explanations_dir)
        write_summary(metrics_dir / "summary_metrics.csv", rows)
        update_manifest(metrics_dir / "batch_gnnexplainer_manifest.json", rows)
    print(f"Fixed runtime reporting for {len(pairs)} Phase 4C metrics directories")


def explanation_dir_for_metrics(metrics_dir: Path) -> Path:
    return Path("results/explanations") / metrics_dir.name


def update_per_node(path: Path, explanations_dir: Path) -> list[dict[str, str]]:
    rows, fieldnames = read_csv(path)
    fieldnames = ensure_fieldnames(fieldnames)
    for row in rows:
        metadata = read_metadata(explanations_dir, row["node_id"])
        source = (
            "reused_existing_raw_mask"
            if metadata.get("raw_explanation_reused")
            else "generated_raw_mask"
        )
        generation_runtime = recovered_generation_runtime(metadata)
        evaluation_runtime = metadata.get("evaluation_runtime_seconds")
        row["explanation_source"] = source
        row["generation_runtime_seconds"] = stringify_nullable(generation_runtime)
        row["evaluation_runtime_seconds"] = stringify_nullable(evaluation_runtime)
        row["runtime_seconds"] = stringify_nullable(generation_runtime)
    validate_fidelity_alias(rows)
    validate_runtime_reporting(rows)
    write_csv(path, fieldnames, rows)
    update_node_metadata(explanations_dir, rows)
    return rows


def update_node_metadata(explanations_dir: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        metadata_path = explanations_dir / f"node_{row['node_id']}" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["explanation_source"] = row["explanation_source"]
        metadata["generation_runtime_seconds"] = nullable_float(row["generation_runtime_seconds"])
        metadata["evaluation_runtime_seconds"] = nullable_float(row["evaluation_runtime_seconds"])
        metadata["runtime_seconds"] = nullable_float(row["runtime_seconds"])
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summary = summarize(rows)
    write_csv(path, ["metric", "n", "mean", "std", "median", "min", "max"], summary)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    success_rows = [row for row in rows if row["status"] == "success"]
    summary = []
    for metric in SUMMARY_METRICS:
        values = [
            float(row[metric])
            for row in success_rows
            if row.get(metric) not in (None, "")
        ]
        if values:
            summary.append(
                {
                    "metric": metric,
                    "n": len(values),
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "median": median(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
        else:
            summary.append(
                {
                    "metric": metric,
                    "n": 0,
                    "mean": None,
                    "std": None,
                    "median": None,
                    "min": None,
                    "max": None,
                }
            )
    return summary


def update_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    generation_values = [
        float(row["generation_runtime_seconds"])
        for row in rows
        if row.get("generation_runtime_seconds") not in (None, "")
    ]
    evaluation_values = [
        float(row["evaluation_runtime_seconds"])
        for row in rows
        if row.get("evaluation_runtime_seconds") not in (None, "")
    ]
    manifest["runtime_reporting"] = "generation_runtime_seconds and evaluation_runtime_seconds"
    manifest["raw_generation_runtime_seconds"] = sum(generation_values)
    manifest["evaluation_runtime_seconds"] = sum(evaluation_values) if evaluation_values else None
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recovered_generation_runtime(metadata: dict[str, Any]) -> float | None:
    for key in ("generation_runtime_seconds", "source_runtime_seconds", "runtime_seconds"):
        value = metadata.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def read_metadata(explanations_dir: Path, node_id: str) -> dict[str, Any]:
    path = explanations_dir / f"node_{node_id}" / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_fieldnames(fieldnames: list[str]) -> list[str]:
    output = list(fieldnames)
    runtime_fields = {
        "generation_runtime_seconds",
        "evaluation_runtime_seconds",
        "explanation_source",
    }
    for field in PER_NODE_FIELDS:
        if field not in output:
            if field in runtime_fields:
                output.insert(output.index("runtime_seconds") + 1, field)
            else:
                output.append(field)
    return output


def stringify_nullable(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(float(value))


def nullable_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


if __name__ == "__main__":
    main()
