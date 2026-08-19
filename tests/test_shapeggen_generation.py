from __future__ import annotations

from pathlib import Path

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.data.shapeggen import generate_shapeggen_data, load_shapeggen_config
from gnnexplainer_benchmark.data.validation import validate_shapeggen_data


def test_frozen_shapeggen_artifact_shapes_are_valid() -> None:
    data = load_data_artifact(Path("artifacts/data/shapeggen.pt"))

    validate_shapeggen_data(data)
    assert data.num_nodes == 1279
    assert data.edge_index.size(1) == 4390
    assert data.x.size(1) == 10
    assert data.y.bincount().tolist() == [508, 771]
    assert int(data.edge_mask.sum().item()) == 27700


def test_frozen_shapeggen_artifact_checksum_is_locked() -> None:
    assert (
        sha256_file(Path("artifacts/data/shapeggen.pt"))
        == "ecf0975a1a38ed89225b91992331165b14879b943dc96360ca4406dbbaacc6ac"
    )


def test_small_shapeggen_generation_is_valid() -> None:
    config = load_shapeggen_config(Path("configs/dataset/shapeggen.yaml"))
    config["dataset"]["num_subgraphs"] = 12
    config["dataset"]["seed"] = 123
    config["dataset"]["prob_connection"] = 0.12

    data = generate_shapeggen_data(config)

    validate_shapeggen_data(data)
    assert data.edge_index.size(1) % 2 == 0
    assert int(data.edge_mask.sum().item()) > 0
