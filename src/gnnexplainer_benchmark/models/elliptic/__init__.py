"""Elliptic classifiers; separate from the synthetic backbone family."""

from .gcn import EllipticGCN
from .graphsage import EllipticSAGE

__all__ = ["EllipticGCN", "EllipticSAGE"]
