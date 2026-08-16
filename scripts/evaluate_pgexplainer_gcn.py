from __future__ import annotations

import argparse
from pathlib import Path

from gnnexplainer_benchmark.explainers.pgexplainer_evaluation import (
    evaluate_pgexplainer_gcn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained PGExplainer on frozen GCN final test targets."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/explanations/pgexplainer_gcn_seed42"),
    )
    parser.add_argument(
        "--metrics-output-dir",
        type=Path,
        default=Path("results/metrics/pgexplainer_gcn_seed42"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_pgexplainer_gcn(
        device=args.device,
        output_root=args.output_root,
        metrics_output_dir=args.metrics_output_dir,
    )
    print(
        "PGExplainer GCN evaluation complete | "
        f"success={result.success_count} | "
        f"failed={result.failure_count} | "
        f"model_weights_unchanged={result.model_weights_unchanged}"
    )
    print(f"per_node_metrics={result.per_node_metrics_path}")
    print(f"summary_metrics={result.summary_metrics_path}")
    print(f"manifest={result.manifest_path}")


if __name__ == "__main__":
    main()
