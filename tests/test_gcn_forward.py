from __future__ import annotations

from pathlib import Path

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config
from gnnexplainer_benchmark.training.trainer import build_gcn_model, load_training_config


def test_gcn_forward_returns_node_logits() -> None:
    data = generate_ba_house_data(load_ba_house_config(Path("configs/ba_house/ba500/ba_house.yaml")))
    config = load_training_config(Path("configs/ba_house/ba500/gcn.yaml"))
    model = build_gcn_model(data, config)

    logits = model(data.x, data.edge_index)

    assert logits.shape == (data.num_nodes, int(data.y.max().item()) + 1)
