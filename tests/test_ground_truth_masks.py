from __future__ import annotations

from pathlib import Path

import torch

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config


def test_ground_truth_masks_exist_and_are_not_empty() -> None:
    config = load_ba_house_config(Path("configs/ba_house/ba500/ba_house.yaml"))

    data = generate_ba_house_data(config)

    assert hasattr(data, "node_mask")
    assert hasattr(data, "edge_mask")
    assert data.node_mask.numel() == data.num_nodes
    assert data.edge_mask.numel() == data.edge_index.size(1)
    assert int(data.node_mask.sum().item()) > 0
    assert int(data.edge_mask.sum().item()) > 0
    assert torch.isfinite(data.node_mask).all()
    assert torch.isfinite(data.edge_mask).all()
