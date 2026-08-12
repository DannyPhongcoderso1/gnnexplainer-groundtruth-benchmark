from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import mean

import torch


@dataclass(frozen=True)
class RatioBudget:
    ratio: float
    candidate_edge_count: int
    selected_edge_count: int
    sparsity: float


def selected_edge_count_for_ratio(candidate_edge_count: int, ratio: float) -> int:
    if candidate_edge_count <= 0:
        raise ValueError("candidate_edge_count must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    return min(candidate_edge_count, max(1, ceil(candidate_edge_count * ratio)))


def select_top_ratio_edges(
    edge_scores: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, RatioBudget]:
    if edge_scores.numel() != candidate_edge_mask.numel():
        raise ValueError("edge_scores and candidate_edge_mask must align")
    if candidate_edge_mask.dtype != torch.bool:
        raise ValueError("candidate_edge_mask must be boolean")
    candidate_indices = torch.nonzero(candidate_edge_mask, as_tuple=False).view(-1)
    candidate_count = int(candidate_indices.numel())
    selected_count = selected_edge_count_for_ratio(candidate_count, ratio)
    candidate_scores = edge_scores[candidate_indices].float()
    _, local_indices = torch.topk(candidate_scores, k=selected_count)
    selected_indices = candidate_indices[local_indices]
    selected_mask = torch.zeros_like(candidate_edge_mask, dtype=torch.bool)
    selected_mask[selected_indices] = True
    return selected_mask, RatioBudget(
        ratio=ratio,
        candidate_edge_count=candidate_count,
        selected_edge_count=selected_count,
        sparsity=1.0 - (selected_count / candidate_count),
    )


def choose_ratio_by_validation_remove_delta(rows: list[dict], ratios: list[float]) -> float:
    if not rows:
        raise ValueError("rows must not be empty")
    ratio_to_mean = {}
    for ratio in ratios:
        values = [float(row["remove_delta"]) for row in rows if float(row["ratio"]) == ratio]
        if not values:
            raise ValueError(f"No validation rows for ratio={ratio}")
        ratio_to_mean[ratio] = mean(values)
    best_mean = max(ratio_to_mean.values())
    best_ratios = [ratio for ratio, value in ratio_to_mean.items() if value == best_mean]
    return min(best_ratios)
