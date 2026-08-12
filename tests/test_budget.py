from __future__ import annotations

import pytest
import torch

from gnnexplainer_benchmark.evaluation.budget import (
    choose_ratio_by_validation_remove_delta,
    select_top_ratio_edges,
    selected_edge_count_for_ratio,
)


def test_selected_edge_count_for_ratio_uses_candidate_relative_budget() -> None:
    assert selected_edge_count_for_ratio(24, 0.05) == 2
    assert selected_edge_count_for_ratio(434, 0.05) == 22
    assert selected_edge_count_for_ratio(866, 0.20) == 174


def test_select_top_ratio_edges_stays_inside_candidate_scope() -> None:
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    candidate = torch.tensor([False, True, True, True])

    selected, budget = select_top_ratio_edges(scores, candidate, 0.50)

    assert selected.tolist() == [False, True, True, False]
    assert budget.selected_edge_count == 2
    assert budget.sparsity == pytest.approx(1.0 / 3.0)


def test_choose_ratio_by_validation_remove_delta_uses_validation_only_summary() -> None:
    rows = [
        {"ratio": 0.05, "remove_delta": 0.1},
        {"ratio": 0.05, "remove_delta": 0.0},
        {"ratio": 0.10, "remove_delta": 0.2},
        {"ratio": 0.10, "remove_delta": 0.0},
        {"ratio": 0.20, "remove_delta": 0.05},
        {"ratio": 0.20, "remove_delta": 0.05},
    ]

    assert choose_ratio_by_validation_remove_delta(rows, [0.05, 0.10, 0.20]) == 0.10
