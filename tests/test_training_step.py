from __future__ import annotations

from pathlib import Path

import torch

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config
from gnnexplainer_benchmark.training.trainer import build_gcn_model, load_training_config
from gnnexplainer_benchmark.utils.reproducibility import seed_everything


def test_single_training_step_updates_with_finite_loss() -> None:
    seed_everything(42)
    data = generate_ba_house_data(load_ba_house_config(Path("configs/dataset/ba_house.yaml")))
    config = load_training_config(Path("configs/model/gcn.yaml"))
    model = build_gcn_model(data, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = torch.nn.functional.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
