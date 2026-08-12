from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparsityResult:
    protocol: str
    candidate_edge_count: int
    selected_edge_count: int
    sparsity: float
    top_k: int


def select_top_k_edges(
    edge_scores: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if edge_scores.numel() != candidate_edge_mask.numel():
        raise ValueError("edge_scores and candidate_edge_mask must align")
    if candidate_edge_mask.dtype != torch.bool:
        raise ValueError("candidate_edge_mask must be boolean")
    candidate_indices = torch.nonzero(candidate_edge_mask, as_tuple=False).view(-1)
    if candidate_indices.numel() == 0:
        raise ValueError("candidate_edge_mask must not be empty")

    selected_count = min(top_k, int(candidate_indices.numel()))
    candidate_scores = edge_scores[candidate_indices].float()
    _, local_indices = torch.topk(candidate_scores, k=selected_count)
    selected_indices = candidate_indices[local_indices]
    selected_mask = torch.zeros_like(candidate_edge_mask, dtype=torch.bool)
    selected_mask[selected_indices] = True
    return selected_mask


def compute_sparsity(
    candidate_edge_mask: torch.Tensor,
    selected_edge_mask: torch.Tensor,
    top_k: int,
) -> SparsityResult:
    if candidate_edge_mask.numel() != selected_edge_mask.numel():
        raise ValueError("candidate_edge_mask and selected_edge_mask must align")
    candidate_count = int(candidate_edge_mask.sum().item())
    selected_count = int(selected_edge_mask.sum().item())
    if candidate_count == 0:
        raise ValueError("candidate_edge_count must be positive")
    if bool((selected_edge_mask & ~candidate_edge_mask).any().item()):
        raise ValueError("selected edges must be inside candidate edges")
    return SparsityResult(
        protocol="top_k_candidate_edges",
        candidate_edge_count=candidate_count,
        selected_edge_count=selected_count,
        sparsity=1.0 - (selected_count / candidate_count),
        top_k=top_k,
    )
