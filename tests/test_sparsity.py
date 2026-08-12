from __future__ import annotations

import pytest
import torch

from gnnexplainer_benchmark.evaluation.sparsity import compute_sparsity, select_top_k_edges


def test_select_top_k_edges_only_inside_candidate_scope() -> None:
    scores = torch.tensor([0.9, 0.8, 0.7, 0.1])
    candidate = torch.tensor([False, True, True, True])

    selected = select_top_k_edges(scores, candidate, top_k=2)

    assert selected.tolist() == [False, True, True, False]


def test_sparsity_uses_top_k_candidate_edges_protocol() -> None:
    candidate = torch.tensor([False, True, True, True])
    selected = torch.tensor([False, True, True, False])

    result = compute_sparsity(candidate, selected, top_k=2)

    assert result.protocol == "top_k_candidate_edges"
    assert result.candidate_edge_count == 3
    assert result.selected_edge_count == 2
    assert result.sparsity == pytest.approx(1.0 / 3.0)


def test_sparsity_rejects_selected_edge_outside_candidate_scope() -> None:
    candidate = torch.tensor([False, True, True])
    selected = torch.tensor([True, True, False])

    with pytest.raises(ValueError, match="inside candidate"):
        compute_sparsity(candidate, selected, top_k=2)
