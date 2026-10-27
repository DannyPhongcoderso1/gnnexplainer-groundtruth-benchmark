"""Backbones shared by BA-House and ShapeGGen; dataset conventions remain separate."""

from .gcn import GCN
from .graphsage import GraphSAGE

__all__ = ["GCN", "GraphSAGE"]
