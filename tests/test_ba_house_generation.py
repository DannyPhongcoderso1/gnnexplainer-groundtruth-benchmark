from __future__ import annotations

from pathlib import Path

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config
from gnnexplainer_benchmark.data.validation import validate_ba_house_data


def test_ba_house_generation_shapes_are_valid() -> None:
    config = load_ba_house_config(Path("configs/ba_house/ba500/ba_house.yaml"))

    data = generate_ba_house_data(config)

    validate_ba_house_data(data)
    assert data.x.size(0) == data.y.numel()
    assert data.edge_mask.numel() == data.edge_index.size(1)
    assert data.node_mask.numel() == data.num_nodes
    assert data.x.size(1) == config["dataset"]["feature_dim"]
