from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from gnnexplainer_benchmark.explain.node_selection import TargetNode
from gnnexplainer_benchmark.utils.reproducibility import seed_everything


def load_explainer_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "explainer" not in config:
        raise ValueError("Config must contain a top-level 'explainer' mapping")
    return config


def run_gnnexplainer_for_node(
    model: torch.nn.Module,
    data: Data,
    target: TargetNode,
    config: dict[str, Any],
) -> Any:
    explainer_config = config["explainer"]
    seed_everything(int(explainer_config.get("seed", 42)))
    model.eval()
    before_state = _state_snapshot(model)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(
            epochs=int(explainer_config["epochs"]),
            lr=float(explainer_config["learning_rate"]),
        ),
        explanation_type=str(explainer_config["explanation_type"]),
        node_mask_type=str(explainer_config["node_mask_type"]),
        edge_mask_type=str(explainer_config["edge_mask_type"]),
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level=str(explainer_config["task_level"]),
            return_type=str(explainer_config["return_type"]),
        ),
    )
    explanation = explainer(data.x, data.edge_index, index=target.node_id)
    if not model_weights_unchanged(model, before_state):
        raise RuntimeError("Model weights changed during explanation")
    return explanation


def model_weights_unchanged(
    model: torch.nn.Module,
    before_state: dict[str, torch.Tensor],
) -> bool:
    current = model.state_dict()
    return all(
        torch.equal(current[name].detach().cpu(), tensor) for name, tensor in before_state.items()
    )


def _state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: deepcopy(value.detach().cpu()) for name, value in model.state_dict().items()}
