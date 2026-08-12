from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnnexplainer_benchmark.evaluation.schema import validate_core_metric_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Phase 4B core-metrics gate.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/metrics/node_465_metrics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    validate_core_metric_schema(metrics)
    edge_ground_truth = metrics["edge_ground_truth"]
    if edge_ground_truth["candidate_positive_edge_count"] <= 0:
        raise SystemExit("Phase 4B gate failed: no positive ground-truth edges")
    if edge_ground_truth["candidate_negative_edge_count"] <= 0:
        raise SystemExit("Phase 4B gate failed: no negative candidate edges")
    if not edge_ground_truth["edge_mapping_validated"]:
        raise SystemExit("Phase 4B gate failed: edge mapping was not validated")
    print(
        "Phase 4B validation gate passed | "
        f"node={metrics['node_id']} | "
        f"schema={metrics['schema_version']} | "
        f"top_k={metrics['sparsity']['top_k']} | "
        f"prevalence={edge_ground_truth['ground_truth_edge_prevalence']:.6f}"
    )


if __name__ == "__main__":
    main()
