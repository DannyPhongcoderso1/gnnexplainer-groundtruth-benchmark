from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.explain.metric import fidelity as pyg_fidelity

from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity, fid_minus, fid_plus


class EdgeCountModel(nn.Module):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        class_one_logit = torch.full((x.size(0),), float(edge_index.size(1)))
        class_zero_logit = torch.zeros(x.size(0))
        return torch.stack([class_zero_logit, class_one_logit], dim=-1)


def test_fidelity_remove_and_retain_top_k_edges_protocol() -> None:
    data = Data(
        x=torch.ones((3, 2)),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
    )
    selected = torch.tensor([True, False, False])

    result = compute_fidelity(EdgeCountModel(), data, node_id=0, selected_edge_mask=selected)

    assert result.protocol == "remove_top_k_and_retain_top_k_edges"
    assert result.metric_name == "remove_delta"
    assert result.original_prediction == 1
    assert result.remove_probability < result.original_probability
    assert result.retain_probability < result.original_probability
    assert result.remove_delta > 0
    assert result.remove_probability_delta == result.remove_delta
    assert result.retain_probability_delta == result.retain_delta
    assert result.remove_probability_drop == result.remove_delta


def test_fid_plus_minus_manual_cases() -> None:
    assert fid_plus(original_prediction=1, remove_prediction=0) == 1
    assert fid_minus(original_prediction=1, retain_prediction=1) == 0

    assert fid_plus(original_prediction=1, remove_prediction=1) == 0
    assert fid_minus(original_prediction=1, retain_prediction=0) == 1

    assert fid_plus(original_prediction=1, remove_prediction=1) == 0
    assert fid_minus(original_prediction=1, retain_prediction=1) == 0


def test_pyg_fidelity_metric_available_and_manual_model_formula_matches() -> None:
    assert callable(pyg_fidelity)

    cases = [
        (1, 0, 1, 1, 0),
        (1, 1, 0, 0, 1),
        (1, 1, 1, 0, 0),
    ]
    for original, remove, retain, expected_plus, expected_minus in cases:
        assert fid_plus(original, remove) == expected_plus
        assert fid_minus(original, retain) == expected_minus
