from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnnexplainer_benchmark.explain.mask_export import validate_explanation_masks


class DummyExplanation:
    def __init__(self) -> None:
        self.edge_mask = torch.tensor([0.1, 0.2])
        self.node_mask = torch.tensor([[0.3, 0.4]])


def test_explanation_masks_must_be_finite_and_edge_aligned() -> None:
    data = Data(
        x=torch.ones((3, 2)),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, 1]),
        edge_mask=torch.ones(2),
        node_mask=torch.ones(3),
    )

    validate_explanation_masks(DummyExplanation(), data)
