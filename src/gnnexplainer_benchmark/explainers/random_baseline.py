from __future__ import annotations

import torch


def random_candidate_edge_scores(
    *,
    edge_count: int,
    candidate_edge_mask: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    if candidate_edge_mask.dtype != torch.bool:
        raise ValueError("candidate_edge_mask must be boolean")
    if candidate_edge_mask.numel() != edge_count:
        raise ValueError("candidate_edge_mask must align with edge_count")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scores = torch.zeros(edge_count, dtype=torch.float32)
    candidate_count = int(candidate_edge_mask.sum().item())
    scores[candidate_edge_mask.cpu()] = torch.rand(candidate_count, generator=generator)
    return scores
