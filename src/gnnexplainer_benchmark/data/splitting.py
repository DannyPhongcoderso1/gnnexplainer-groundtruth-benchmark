from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


def make_stratified_node_splits(
    labels: torch.Tensor,
    seed: int,
    config: SplitConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_config = config or SplitConfig()
    total = split_config.train_ratio + split_config.val_ratio + split_config.test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    if labels.dim() != 1:
        raise ValueError("labels must be a 1D tensor")

    num_nodes = int(labels.numel())
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed)

    for label in torch.unique(labels, sorted=True):
        node_idx = torch.nonzero(labels == label, as_tuple=False).view(-1)
        node_idx = node_idx[torch.randperm(node_idx.numel(), generator=generator)]
        train_count = int(node_idx.numel() * split_config.train_ratio)
        val_count = int(node_idx.numel() * split_config.val_ratio)
        train_nodes = node_idx[:train_count]
        val_nodes = node_idx[train_count : train_count + val_count]
        test_nodes = node_idx[train_count + val_count :]
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

    return train_mask, val_mask, test_mask


def split_counts(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict[str, int]:
    return {
        "train": int(train_mask.sum().item()),
        "validation": int(val_mask.sum().item()),
        "test": int(test_mask.sum().item()),
    }
