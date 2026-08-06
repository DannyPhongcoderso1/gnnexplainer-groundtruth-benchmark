from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnnexplainer_benchmark.explain.node_selection import (
    correctly_predicted_motif_test_nodes,
    select_target_node,
)


def test_selects_only_correctly_predicted_motif_test_nodes() -> None:
    data = Data(
        y=torch.tensor([0, 1, 2, 1]),
        test_mask=torch.tensor([True, True, True, False]),
    )
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    targets = correctly_predicted_motif_test_nodes(logits, data)

    assert [target.node_id for target in targets] == [1]
    assert select_target_node(logits, data).node_id == 1
