from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnnexplainer_benchmark.data.target_selection import (
    ExplainerTarget,
    select_correct_motif_targets,
    split_explainer_targets,
)


def test_selects_only_correct_test_split_ground_truth_motif_nodes() -> None:
    data = Data(
        y=torch.tensor([0, 1, 2, 1, 2]),
        node_mask=torch.tensor([False, True, True, True, False]),
        test_mask=torch.tensor([True, True, True, False, True]),
    )
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )

    targets = select_correct_motif_targets(logits, data)

    assert [target.node_id for target in targets] == [1]
    assert targets[0].split == "test"
    assert targets[0].is_ground_truth_motif is True
    assert targets[0].label == targets[0].prediction


def test_split_explainer_targets_is_non_overlapping_and_deterministic() -> None:
    targets = [
        ExplainerTarget(
            node_id=node_id,
            label=1,
            prediction=1,
            confidence=0.9,
            split="test",
            is_ground_truth_motif=True,
        )
        for node_id in range(10)
    ]

    first_validation, first_test = split_explainer_targets(targets, seed=42)
    second_validation, second_test = split_explainer_targets(targets, seed=42)

    assert [target.node_id for target in first_validation] == [
        target.node_id for target in second_validation
    ]
    assert [target.node_id for target in first_test] == [target.node_id for target in second_test]
    assert {target.node_id for target in first_validation}.isdisjoint(
        {target.node_id for target in first_test}
    )
    assert len(first_validation) == 2
    assert len(first_test) == 8
