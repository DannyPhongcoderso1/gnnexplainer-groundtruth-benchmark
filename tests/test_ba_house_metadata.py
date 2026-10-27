from __future__ import annotations

from pathlib import Path

from gnnexplainer_benchmark.data.ba_house import generate_and_save_ba_house
from gnnexplainer_benchmark.data.serialization import load_metadata, sha256_file


def test_metadata_matches_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "ba_house.pt"
    metadata_path = tmp_path / "ba_house_metadata.json"

    data = generate_and_save_ba_house(
        Path("configs/ba_house/ba500/ba_house.yaml"),
        artifact,
        metadata_path,
    )
    metadata = load_metadata(metadata_path)

    assert metadata["sha256"] == sha256_file(artifact)
    assert metadata["seed"] == 42
    assert metadata["shapes"]["num_nodes"] == data.num_nodes
    assert metadata["shapes"]["num_edges"] == data.edge_index.size(1)
    assert metadata["ground_truth_counts"]["nodes"] == int(data.node_mask.sum().item())
    assert metadata["ground_truth_counts"]["edges"] == int(data.edge_mask.sum().item())
    assert sum(metadata["split_counts"].values()) == data.num_nodes
