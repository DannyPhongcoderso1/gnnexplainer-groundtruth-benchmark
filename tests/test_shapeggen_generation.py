from __future__ import annotations

import inspect
from pathlib import Path

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.data.shapeggen import (
    GRAPHXAI_COMMIT,
    generate_shapeggen_data,
    load_shapeggen_config,
    validate_shapeggen_config,
)
from gnnexplainer_benchmark.data.validation import validate_shapeggen_data
from graphxai.datasets.shape_graph import ShapeGGen


def test_frozen_shapeggen_artifact_shapes_are_valid() -> None:
    data = load_data_artifact(Path("data/shapeggen/v1/graph.pt"))

    validate_shapeggen_data(data)
    assert data.num_nodes == 1298
    assert data.edge_index.size(1) == 4518
    assert data.x.size(1) == 10
    assert data.y.bincount().tolist() == [502, 796]
    assert int(data.edge_mask.sum().item()) == 28316
    assert data.graphxai_upstream_commit == GRAPHXAI_COMMIT


def test_frozen_shapeggen_artifact_checksum_is_locked() -> None:
    assert (
        sha256_file(Path("data/shapeggen/v1/graph.pt"))
        == "9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e"
    )


def test_graphxai_import_and_upstream_identity() -> None:
    assert ShapeGGen.__module__ == "graphxai.datasets.shape_graph"


def test_active_adapter_does_not_contain_local_generator_algorithm() -> None:
    import gnnexplainer_benchmark.data.shapeggen as shapeggen

    source = inspect.getsource(shapeggen)
    forbidden = [
        "def _build_bound_graph",
        "def _ba_around_shape",
        "def _make_structured_feature",
        "def _generate_hypercube",
    ]
    assert not any(token in source for token in forbidden)


def test_small_shapeggen_generation_is_valid() -> None:
    config = load_shapeggen_config(Path("configs/shapeggen/v1/shapeggen.yaml"))
    config["dataset"]["upstream_generator"]["num_subgraphs"] = 12
    config["dataset"]["upstream_generator"]["seed"] = 123
    config["dataset"]["upstream_generator"]["prob_connection"] = 0.12
    config["dataset"]["local_adapter"]["split"]["seed"] = 123

    data = generate_shapeggen_data(config)

    validate_shapeggen_data(data)
    assert data.edge_index.size(1) % 2 == 0
    assert int(data.edge_mask.sum().item()) > 0


def test_unsupported_shapeggen_config_fields_fail_fast() -> None:
    config = load_shapeggen_config(Path("configs/shapeggen/v1/shapeggen.yaml"))
    config["dataset"]["upstream_generator"]["unsupported_field"] = "boom"

    try:
        validate_shapeggen_config(config)
    except ValueError as exc:
        assert "Unsupported ShapeGGen configuration field" in str(exc)
    else:
        raise AssertionError("unsupported ShapeGGen field was silently accepted")
