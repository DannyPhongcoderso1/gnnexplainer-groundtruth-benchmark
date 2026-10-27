"""Accepted PyG PGExplainer API; training and evaluation remain separate modules.

Historical seed42 and FRESH5 checkpoints/configs are distinct. Experimental
variants are not exported here.
"""

from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    build_pgexplainer_from_config,
    load_yaml,
)

__all__ = ["build_pgexplainer_from_config", "load_yaml"]
