from __future__ import annotations

from pathlib import Path

from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.data.shapeggen import (
    get_target_candidate_edge_mask,
    get_target_edge_ground_truth,
)


def test_shapeggen_positive_edges_are_candidate_edges() -> None:
    data = load_data_artifact(Path("artifacts/data/shapeggen.pt"))

    assert bool((data.edge_mask & ~data.candidate_edge_mask).any()) is False
    assert bool((data.edge_mask.sum(dim=1) > 0).all())
    assert bool((data.candidate_edge_mask.sum(dim=1) > data.edge_mask.sum(dim=1)).all())


def test_shapeggen_reverse_edge_masks_are_aligned() -> None:
    data = load_data_artifact(Path("artifacts/data/shapeggen.pt"))
    edge_lookup = {
        (int(data.edge_index[0, edge_idx]), int(data.edge_index[1, edge_idx])): edge_idx
        for edge_idx in range(data.edge_index.size(1))
    }

    for (src, dst), edge_idx in edge_lookup.items():
        reverse_idx = edge_lookup[(dst, src)]
        candidate_forward = data.candidate_edge_mask[:, edge_idx]
        candidate_reverse = data.candidate_edge_mask[:, reverse_idx]
        positive_forward = data.edge_mask[:, edge_idx]
        positive_reverse = data.edge_mask[:, reverse_idx]
        assert bool(
            (candidate_forward == candidate_reverse).all()
        )
        assert bool((positive_forward == positive_reverse).all())


def test_shapeggen_feature_explanations_are_stored_but_not_core() -> None:
    data = load_data_artifact(Path("artifacts/data/shapeggen.pt"))

    assert data.feature_mask.numel() == data.x.size(1)
    assert int(data.feature_mask.sum().item()) == 6
    assert data.feature_mask.nonzero(as_tuple=True)[0].tolist() == [0, 1, 3, 6, 8, 9]
    assert "Feature explanations are excluded from core metrics" not in data.ground_truth_semantics


def test_target_specific_gt_adapter_returns_edge_index_aligned_masks() -> None:
    data = load_data_artifact(Path("artifacts/data/shapeggen.pt"))

    for node_id in [0, 17, data.num_nodes - 1]:
        gt = get_target_edge_ground_truth(data, node_id)
        candidate = get_target_candidate_edge_mask(data, node_id)
        assert gt.shape == candidate.shape == (data.edge_index.size(1),)
        assert int(gt[candidate].sum().item()) > 0
        assert int((candidate & ~gt).sum().item()) > 0
