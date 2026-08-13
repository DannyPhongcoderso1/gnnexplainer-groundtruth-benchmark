from __future__ import annotations

import torch

from gnnexplainer_benchmark.evaluation.budget import (
    select_top_ratio_edges,
    selected_edge_count_for_ratio,
)
from gnnexplainer_benchmark.explainers.random_baseline import random_candidate_edge_scores


def test_random_baseline_scores_only_candidate_edges() -> None:
    candidate_edge_mask = torch.tensor([True, False, True, False, True])

    scores = random_candidate_edge_scores(
        edge_count=5,
        candidate_edge_mask=candidate_edge_mask,
        seed=7,
    )

    assert torch.equal(scores[~candidate_edge_mask], torch.zeros(2))
    assert torch.all((scores[candidate_edge_mask] >= 0.0) & (scores[candidate_edge_mask] <= 1.0))


def test_random_baseline_is_seeded_and_reproducible() -> None:
    candidate_edge_mask = torch.tensor([True, True, True])

    first = random_candidate_edge_scores(
        edge_count=3,
        candidate_edge_mask=candidate_edge_mask,
        seed=3,
    )
    second = random_candidate_edge_scores(
        edge_count=3,
        candidate_edge_mask=candidate_edge_mask,
        seed=3,
    )

    assert torch.equal(first, second)


def test_random_baseline_different_seeds_can_select_different_edges() -> None:
    candidate_edge_mask = torch.ones(20, dtype=torch.bool)
    first_scores = random_candidate_edge_scores(
        edge_count=20,
        candidate_edge_mask=candidate_edge_mask,
        seed=1,
    )
    second_scores = random_candidate_edge_scores(
        edge_count=20,
        candidate_edge_mask=candidate_edge_mask,
        seed=2,
    )
    first_selected, first_budget = select_top_ratio_edges(first_scores, candidate_edge_mask, 0.2)
    second_selected, second_budget = select_top_ratio_edges(second_scores, candidate_edge_mask, 0.2)

    assert first_budget.selected_edge_count == selected_edge_count_for_ratio(20, 0.2)
    assert second_budget.selected_edge_count == selected_edge_count_for_ratio(20, 0.2)
    assert not torch.equal(first_selected, second_selected)
    assert len(torch.nonzero(first_selected, as_tuple=False).view(-1).tolist()) == len(
        set(torch.nonzero(first_selected, as_tuple=False).view(-1).tolist())
    )
