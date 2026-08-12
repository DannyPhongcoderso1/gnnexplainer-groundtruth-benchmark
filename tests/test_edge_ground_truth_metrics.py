from __future__ import annotations

import pytest
import torch

from gnnexplainer_benchmark.evaluation.edge_ground_truth import (
    candidate_edge_scores_and_labels,
    edge_auroc,
    edge_average_precision,
)


def test_edge_metrics_perfect_ranking() -> None:
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    ground_truth = torch.tensor([True, True, False, False])

    assert edge_auroc(scores, ground_truth).value == 1.0
    assert edge_average_precision(scores, ground_truth).value == 1.0


def test_edge_auroc_reversed_ranking() -> None:
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    ground_truth = torch.tensor([True, True, False, False])

    assert edge_auroc(scores, ground_truth).value == 0.0


def test_all_equal_scores_are_valid_when_labels_have_both_classes() -> None:
    scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
    ground_truth = torch.tensor([True, True, False, False])

    assert edge_auroc(scores, ground_truth).valid is True
    assert edge_auroc(scores, ground_truth).value == 0.5
    assert edge_average_precision(scores, ground_truth).valid is True


def test_no_positive_ground_truth_is_invalid_not_fabricated() -> None:
    scores = torch.tensor([0.9, 0.1])
    ground_truth = torch.tensor([False, False])

    result = edge_auroc(scores, ground_truth)

    assert result.valid is False
    assert result.value is None
    assert result.reason == "candidate set has no positive ground-truth edges"


def test_candidate_alignment_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="must align"):
        candidate_edge_scores_and_labels(
            edge_scores=torch.tensor([0.1, 0.2]),
            ground_truth_edge_mask=torch.tensor([True]),
            candidate_edge_mask=torch.tensor([True, True]),
        )


def test_edge_mapping_requires_full_graph_aligned_candidate_mask() -> None:
    with pytest.raises(ValueError, match="must align"):
        candidate_edge_scores_and_labels(
            edge_scores=torch.tensor([0.1, 0.2, 0.3]),
            ground_truth_edge_mask=torch.tensor([True, False, False]),
            candidate_edge_mask=torch.tensor([True, False]),
        )


def test_candidate_scope_slices_scores_and_ground_truth() -> None:
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    ground_truth = torch.tensor([False, True, True, False])
    candidate = torch.tensor([False, True, True, False])

    candidate_scores, candidate_ground_truth = candidate_edge_scores_and_labels(
        scores,
        ground_truth,
        candidate,
    )

    assert candidate_scores.tolist() == pytest.approx([0.9, 0.2])
    assert candidate_ground_truth.tolist() == [True, True]
