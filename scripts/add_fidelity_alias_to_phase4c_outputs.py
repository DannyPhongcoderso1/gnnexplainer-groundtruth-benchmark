from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import validate_fidelity_alias

DEFAULT_METRICS_DIRS = [
    Path("results/metrics/phase4c_pilot"),
    Path("results/metrics/frozen_explainer_test_ratio20"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add fidelity=remove_delta alias to Phase 4C outputs."
    )
    parser.add_argument("--metrics-dir", action="append", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dirs = args.metrics_dir or DEFAULT_METRICS_DIRS
    for metrics_dir in metrics_dirs:
        update_per_node(metrics_dir / "per_node_metrics.csv")
        update_summary(metrics_dir / "summary_metrics.csv")
    print(f"Updated fidelity alias for {len(metrics_dirs)} Phase 4C metrics directories")


def update_per_node(path: Path) -> None:
    rows, fieldnames = read_csv(path)
    if "fidelity" not in fieldnames:
        fieldnames = insert_after(fieldnames, "remove_delta", "fidelity")
    for row in rows:
        row["fidelity"] = row.get("remove_delta", "")
    validate_fidelity_alias(rows)
    write_csv(path, fieldnames, rows)


def update_summary(path: Path) -> None:
    rows, fieldnames = read_csv(path)
    metric_names = [row["metric"] for row in rows]
    if "fidelity" not in metric_names:
        remove_delta_row = next(row for row in rows if row["metric"] == "remove_delta")
        fidelity_row = dict(remove_delta_row)
        fidelity_row["metric"] = "fidelity"
        insert_at = metric_names.index("remove_delta") + 1
        rows.insert(insert_at, fidelity_row)
    else:
        remove_delta_row = next(row for row in rows if row["metric"] == "remove_delta")
        for row in rows:
            if row["metric"] == "fidelity":
                row.update({**remove_delta_row, "metric": "fidelity"})
    write_csv(path, fieldnames, rows)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def insert_after(items: list[str], after: str, item: str) -> list[str]:
    output = list(items)
    output.insert(output.index(after) + 1, item)
    return output


if __name__ == "__main__":
    main()
