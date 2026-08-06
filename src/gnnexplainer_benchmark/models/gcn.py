from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        cached: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("GCN requires at least 2 layers")

        self.dropout = float(dropout)
        channels = [in_channels, *([hidden_channels] * (num_layers - 1)), out_channels]
        self.convs = nn.ModuleList(
            GCNConv(channels[idx], channels[idx + 1], cached=cached)
            for idx in range(num_layers)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
