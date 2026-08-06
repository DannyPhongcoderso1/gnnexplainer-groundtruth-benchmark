from __future__ import annotations

import torch
from torch_geometric.data import Data


def validate_ba_house_data(data: Data) -> None:
    required_attrs = ("x", "edge_index", "y", "node_mask", "edge_mask")
    for attr in required_attrs:
        if not hasattr(data, attr) or getattr(data, attr) is None:
            raise ValueError(f"Missing required attribute: {attr}")

    if data.x.dim() != 2:
        raise ValueError("x must be a 2D node feature tensor")
    if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if data.y.dim() != 1:
        raise ValueError("y must be a 1D node label tensor")
    if data.node_mask.dim() != 1:
        raise ValueError("node_mask must be a 1D tensor")
    if data.edge_mask.dim() != 1:
        raise ValueError("edge_mask must be a 1D tensor")

    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    if data.y.numel() != num_nodes:
        raise ValueError("y must contain one label per node")
    if data.node_mask.numel() != num_nodes:
        raise ValueError("node_mask must contain one value per node")
    if data.edge_mask.numel() != num_edges:
        raise ValueError("edge_mask.numel() must equal edge_index.size(1)")
    if num_nodes == 0 or num_edges == 0:
        raise ValueError("graph must not be empty")
    if int(data.node_mask.sum().item()) == 0:
        raise ValueError("node_mask must not be empty")
    if int(data.edge_mask.sum().item()) == 0:
        raise ValueError("edge_mask must not be empty")

    for name in ("x", "edge_index", "y", "node_mask", "edge_mask"):
        tensor = getattr(data, name)
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")

    if hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask"):
        validate_splits(data.train_mask, data.val_mask, data.test_mask, num_nodes)


def validate_splits(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_nodes: int,
) -> None:
    masks = (train_mask, val_mask, test_mask)
    if any(mask.dtype != torch.bool for mask in masks):
        raise ValueError("split masks must be boolean tensors")
    if any(mask.dim() != 1 or mask.numel() != num_nodes for mask in masks):
        raise ValueError("split masks must be 1D tensors with one value per node")
    covered = train_mask | val_mask | test_mask
    if int(covered.sum().item()) != num_nodes:
        raise ValueError("split masks must cover every node")
    has_overlap = (
        (train_mask & val_mask).any()
        or (train_mask & test_mask).any()
        or (val_mask & test_mask).any()
    )
    if bool(has_overlap):
        raise ValueError("split masks must not overlap")
