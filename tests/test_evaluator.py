from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from gnnexplainer_benchmark.evaluation.evaluator import evaluate_node_explanation
from gnnexplainer_benchmark.evaluation.schema import validate_core_metric_schema


class EdgeCountModel(nn.Module):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        class_one_logit = torch.full((x.size(0),), float(edge_index.size(1)))
        class_zero_logit = torch.zeros(x.size(0))
        return torch.stack([class_zero_logit, class_one_logit], dim=-1)


def test_evaluator_returns_core_metric_schema() -> None:
    data = Data(
        x=torch.ones((3, 2)),
        edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]),
        edge_mask=torch.tensor([True, True, False, False]),
    )
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    candidate = torch.tensor([True, True, True, False])

    result = evaluate_node_explanation(
        model=EdgeCountModel(),
        data=data,
        node_id=0,
        edge_scores=scores,
        candidate_edge_mask=candidate,
        top_k=25,
        runtime_seconds=1.25,
        explainer="unit-test-explainer",
        target={"node_id": 0, "label": 1, "prediction": 1, "confidence": 0.8},
    )

    validate_core_metric_schema(result)
    assert result["schema_version"] == "core_metrics_v1"
    assert result["node_id"] == 0
    assert result["edge_ground_truth"]["edge_auroc"] == 1.0
    assert result["edge_ground_truth"]["edge_ap"] == 1.0
    assert result["edge_ground_truth"]["candidate_edge_count"] == 3
    assert result["edge_ground_truth"]["ground_truth_edge_prevalence"] == 2 / 3
    assert result["edge_ground_truth"]["edge_mapping_validated"] is True
    assert result["sparsity"]["selected_edge_count"] == 3
    assert result["sparsity"]["top_k"] == 25
    assert result["fidelity"]["metric_name"] == "remove_delta"
    assert result["fidelity"]["original"]["probability"] >= 0.0
    assert "remove_delta" in result["fidelity"]["remove_result"]
    assert "probability_ratio" in result["fidelity"]["retain_result"]
    assert result["runtime"]["runtime_seconds"] == 1.25
