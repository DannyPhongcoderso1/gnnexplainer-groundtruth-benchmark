"""Compatibility imports for the original GNNExplainer entry point."""

from gnnexplainer_benchmark.explainers.gnnexplainer import (
    load_explainer_config,
    model_weights_unchanged,
    run_gnnexplainer_for_node,
)

__all__ = ["load_explainer_config", "model_weights_unchanged", "run_gnnexplainer_for_node"]
