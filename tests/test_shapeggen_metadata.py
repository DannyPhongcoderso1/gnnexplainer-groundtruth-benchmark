from __future__ import annotations

from pathlib import Path

from gnnexplainer_benchmark.data.serialization import load_metadata, sha256_file


def test_shapeggen_manifest_matches_artifact() -> None:
    metadata = load_metadata(Path("artifacts/data/shapeggen_manifest.json"))

    assert metadata["sha256"] == sha256_file(Path("artifacts/data/shapeggen.pt"))
    assert metadata["seed"] == 912
    assert metadata["source"]["commit"] == "a11e65ffbc4df737f35522a8accf2283a8aeaa37"
    assert metadata["source"]["local_package_installed"] is False
    assert metadata["protocol"]["name"] == "THESIS_SHAPEGGEN_V1"
    assert metadata["protocol"]["feature_explanations_in_core_metrics"] is False
    assert metadata["split_counts"] == {"train": 766, "validation": 255, "test": 258}


def test_shapeggen_manifest_records_ground_truth_statistics() -> None:
    metadata = load_metadata(Path("artifacts/data/shapeggen_manifest.json"))
    gt_counts = metadata["ground_truth_counts"]

    assert gt_counts["eligible_targets"] == metadata["shapes"]["num_nodes"]
    assert gt_counts["positive_edge_entries"] == 27700
    assert gt_counts["candidate_edge_entries"] == 148852
    assert gt_counts["min_positive_edges_per_target"] >= 1
    assert gt_counts["min_candidate_edges_per_target"] > gt_counts["min_positive_edges_per_target"]
