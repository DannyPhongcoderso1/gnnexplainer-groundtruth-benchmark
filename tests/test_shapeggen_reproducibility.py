from __future__ import annotations

from pathlib import Path

import torch

from gnnexplainer_benchmark.data.shapeggen import generate_shapeggen_data, load_shapeggen_config


def test_shapeggen_generation_is_deterministic_for_same_config() -> None:
    config = load_shapeggen_config(Path("configs/dataset/shapeggen.yaml"))
    config["dataset"]["upstream_generator"]["num_subgraphs"] = 12
    config["dataset"]["upstream_generator"]["seed"] = 456
    config["dataset"]["upstream_generator"]["prob_connection"] = 0.1
    config["dataset"]["local_adapter"]["split"]["seed"] = 456

    first = generate_shapeggen_data(config)
    second = generate_shapeggen_data(config)

    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.y, second.y)
    assert torch.equal(first.edge_mask, second.edge_mask)
    assert torch.equal(first.candidate_edge_mask, second.candidate_edge_mask)
