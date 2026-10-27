"""Frozen Elliptic architecture, derived from the Dang/RMIT MIT baseline.

Attribution and full MIT notice: external/elliptic_baseline/LICENSE.
GCN cache semantics match the original explanation-compatible implementation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class EllipticSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embedding_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        aggregator: str,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, embedding_dim, aggr=aggregator))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_dim, aggr=aggregator))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            self.convs.append(SAGEConv(hidden_dim, embedding_dim, aggr=aggregator))
        self.dropout = float(dropout)
        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx < len(self.convs) - 1:
                x = x.relu()
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)
