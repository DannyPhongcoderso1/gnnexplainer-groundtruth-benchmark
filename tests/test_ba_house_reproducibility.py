from __future__ import annotations

from pathlib import Path

import torch

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config


def test_same_seed_generates_same_graph_and_masks() -> None:
    config = load_ba_house_config(Path("configs/ba_house/ba500/ba_house.yaml"))

    first = generate_ba_house_data(config)
    second = generate_ba_house_data(config)

    assert torch.equal(first.x, second.x)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.y, second.y)
    assert torch.equal(first.node_mask, second.node_mask)
    assert torch.equal(first.edge_mask, second.edge_mask)
    assert torch.equal(first.train_mask, second.train_mask)
    assert torch.equal(first.val_mask, second.val_mask)
    assert torch.equal(first.test_mask, second.test_mask)
