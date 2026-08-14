from __future__ import annotations

import argparse
from pathlib import Path

from gnnexplainer_benchmark.explainers.pgexplainer_smoke import run_pgexplainer_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal PGExplainer smoke test.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pgexplainer_smoke_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pgexplainer_smoke(device=args.device, output_path=args.output)
    print(
        "PGExplainer smoke complete | "
        f"train_nodes={result.train_nodes} | "
        f"validation_node={result.validation_node} | "
        f"losses={result.losses} | "
        f"edge_mask_shape={result.edge_mask_shape} | "
        f"candidate_edges={result.candidate_edge_count} | "
        f"selected_edges={result.selected_edge_count} | "
        f"explainer_parameters_changed={result.explainer_parameters_changed} | "
        f"explainer_gradients_present={result.explainer_gradients_present} | "
        f"explainer_gradients_finite={result.explainer_gradients_finite} | "
        f"model_parameters_unchanged={result.model_parameters_unchanged} | "
        f"sentinel_logits_unchanged={result.sentinel_logits_unchanged} | "
        f"target_requires_grad={result.target_requires_grad} | "
        f"target_has_grad_fn={result.target_has_grad_fn}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
