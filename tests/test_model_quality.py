from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnnexplainer_benchmark.evaluation.model_quality import build_model_quality_report


def test_model_quality_report_contains_split_metrics_and_motif_counts() -> None:
    data = Data(
        x=torch.ones((6, 2)),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, 2, 0, 1, 2]),
        node_mask=torch.tensor([False, True, True, False, True, False]),
        train_mask=torch.tensor([True, True, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
    )
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )

    report = build_model_quality_report(logits, data)

    assert report["splits"]["validation"]["overall_accuracy"]["accuracy"] == 0.5
    assert report["splits"]["test"]["overall_accuracy"]["accuracy"] == 0.5
    assert report["splits"]["test"]["motif_nodes"] == 1
    assert report["splits"]["test"]["correctly_predicted_motif_nodes"] == 1
    assert report["splits"]["test"]["confusion_matrix"] == [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
