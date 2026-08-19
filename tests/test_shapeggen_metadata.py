from __future__ import annotations

from pathlib import Path

from gnnexplainer_benchmark.data.serialization import load_metadata, sha256_file


def test_shapeggen_manifest_matches_artifact() -> None:
    metadata = load_metadata(Path("artifacts/data/shapeggen_manifest.json"))

    assert metadata["sha256"] == sha256_file(Path("artifacts/data/shapeggen.pt"))
    assert metadata["generator_seed"] == 912
    assert metadata["upstream_commit_or_version"] == "a11e65ffbc4df737f35522a8accf2283a8aeaa37"
    assert metadata["dependency_strategy"] == "vendored unmodified official GraphXAI source"
    assert metadata["protocol_name"] == "THESIS_SHAPEGGEN_OFFICIAL_V1"
    assert metadata["split"]["counts"] == {"train": 778, "validation": 259, "test": 261}


def test_shapeggen_manifest_records_ground_truth_statistics() -> None:
    metadata = load_metadata(Path("artifacts/data/shapeggen_manifest.json"))
    gt_counts = metadata["ground_truth"]

    assert gt_counts["eligible_targets"] == metadata["shapes"]["num_nodes"]
    assert gt_counts["positive_edge_entries"] == 28316
    assert gt_counts["candidate_edge_entries"] == 156266
    assert gt_counts["min_positive_edges_per_target"] >= 1
    assert gt_counts["min_candidate_edges_per_target"] > gt_counts["min_positive_edges_per_target"]
