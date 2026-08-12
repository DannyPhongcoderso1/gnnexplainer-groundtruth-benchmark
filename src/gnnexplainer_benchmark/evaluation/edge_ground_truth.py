from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | None
    valid: bool
    reason: str | None = None


def candidate_edge_scores_and_labels(
    edge_scores: torch.Tensor,
    ground_truth_edge_mask: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_aligned_edge_tensors(edge_scores, ground_truth_edge_mask, candidate_edge_mask)
    candidate_scores = edge_scores[candidate_edge_mask].float()
    candidate_ground_truth = ground_truth_edge_mask[candidate_edge_mask].bool()
    return candidate_scores, candidate_ground_truth


def validate_aligned_edge_tensors(
    edge_scores: torch.Tensor,
    ground_truth_edge_mask: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
) -> None:
    lengths = {edge_scores.numel(), ground_truth_edge_mask.numel(), candidate_edge_mask.numel()}
    if len(lengths) != 1:
        raise ValueError("edge_scores, ground_truth_edge_mask, and candidate_edge_mask must align")
    if candidate_edge_mask.dtype != torch.bool:
        raise ValueError("candidate_edge_mask must be boolean")
    if int(candidate_edge_mask.sum().item()) == 0:
        raise ValueError("candidate_edge_mask must not be empty")
    if not torch.isfinite(edge_scores.float()).all():
        raise ValueError("edge_scores contains NaN or Inf")


def edge_auroc(edge_scores: torch.Tensor, ground_truth_edge_mask: torch.Tensor) -> MetricResult:
    valid, reason = _binary_metric_validity(ground_truth_edge_mask)
    if not valid:
        return MetricResult(name="edge_auroc", value=None, valid=False, reason=reason)
    value = roc_auc_score(
        ground_truth_edge_mask.bool().numpy().astype(int),
        edge_scores.float().numpy(),
    )
    return MetricResult(name="edge_auroc", value=float(value), valid=True)


def edge_average_precision(
    edge_scores: torch.Tensor,
    ground_truth_edge_mask: torch.Tensor,
) -> MetricResult:
    valid, reason = _binary_metric_validity(ground_truth_edge_mask)
    if not valid:
        return MetricResult(name="edge_ap", value=None, valid=False, reason=reason)
    value = average_precision_score(
        ground_truth_edge_mask.bool().numpy().astype(int),
        edge_scores.float().numpy(),
    )
    return MetricResult(name="edge_ap", value=float(value), valid=True)


def compute_edge_ground_truth_metrics(
    edge_scores: torch.Tensor,
    ground_truth_edge_mask: torch.Tensor,
) -> dict[str, MetricResult]:
    return {
        "edge_auroc": edge_auroc(edge_scores, ground_truth_edge_mask),
        "edge_ap": edge_average_precision(edge_scores, ground_truth_edge_mask),
    }


def _binary_metric_validity(ground_truth_edge_mask: torch.Tensor) -> tuple[bool, str | None]:
    positives = int(ground_truth_edge_mask.bool().sum().item())
    total = int(ground_truth_edge_mask.numel())
    negatives = total - positives
    if positives == 0:
        return False, "candidate set has no positive ground-truth edges"
    if negatives == 0:
        return False, "candidate set has no negative ground-truth edges"
    return True, None
