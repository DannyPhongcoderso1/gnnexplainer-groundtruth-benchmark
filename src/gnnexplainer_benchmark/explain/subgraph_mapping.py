from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def computation_subgraph_edge_mask(data: Data, node_id: int, num_hops: int) -> torch.Tensor:
    _, local_edge_index, _, edge_mask = k_hop_subgraph(
        node_idx=node_id,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    validate_subgraph_edge_mapping(data, local_edge_index, edge_mask)
    return edge_mask


def validate_subgraph_edge_mapping(
    data: Data,
    local_edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
) -> None:
    global_edge_index = data.edge_index[:, edge_mask]
    if global_edge_index.size(1) != local_edge_index.size(1):
        raise ValueError("Subgraph local/global edge counts do not match")
    if edge_mask.dtype != torch.bool or edge_mask.numel() != data.edge_index.size(1):
        raise ValueError("Subgraph edge mask must be boolean and full-edge aligned")
