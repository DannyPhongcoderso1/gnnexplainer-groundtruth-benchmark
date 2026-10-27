from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config
from gnnexplainer_benchmark.explainers.gnnexplainer import run_gnnexplainer_for_node
from gnnexplainer_benchmark.explain.node_selection import TargetNode
from gnnexplainer_benchmark.training.trainer import build_gcn_model, load_training_config


def test_model_weights_are_frozen_during_explanation() -> None:
    data = generate_ba_house_data(load_ba_house_config(Path("configs/ba_house/ba500/ba_house.yaml")))
    model = build_gcn_model(data, load_training_config(Path("configs/ba_house/ba500/gcn.yaml")))
    before = deepcopy(model.state_dict())
    target_id = int(torch.nonzero(data.test_mask & (data.y > 0), as_tuple=False).view(-1)[0].item())
    target = TargetNode(
        node_id=target_id,
        label=int(data.y[target_id].item()),
        prediction=1,
        confidence=0.5,
    )

    run_gnnexplainer_for_node(
        model,
        data,
        target,
        {
            "explainer": {
                "epochs": 1,
                "learning_rate": 0.01,
                "explanation_type": "model",
                "task_level": "node",
                "return_type": "raw",
                "edge_mask_type": "object",
                "node_mask_type": "common_attributes",
                "seed": 42,
            }
        },
    )

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
