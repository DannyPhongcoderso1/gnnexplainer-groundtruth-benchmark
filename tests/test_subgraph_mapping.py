from __future__ import annotations

from pathlib import Path

import torch

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask


def test_subgraph_mapping_returns_full_edge_boolean_mask() -> None:
    data = generate_ba_house_data(load_ba_house_config(Path("configs/dataset/ba_house.yaml")))
    node_id = int(torch.nonzero(data.test_mask & (data.y > 0), as_tuple=False).view(-1)[0].item())

    edge_mask = computation_subgraph_edge_mask(data, node_id=node_id, num_hops=3)

    assert edge_mask.dtype == torch.bool
    assert edge_mask.numel() == data.edge_index.size(1)
    assert int(edge_mask.sum().item()) > 0
