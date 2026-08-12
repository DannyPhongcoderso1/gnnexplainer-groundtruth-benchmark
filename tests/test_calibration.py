from __future__ import annotations

import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from gnnexplainer_benchmark.evaluation.calibration import (
    CalibrationRun,
    assert_gcnconv_cached_false,
    exact_ground_truth_candidate_edge_mask,
    summarize_runs,
)


def test_exact_ground_truth_candidate_edge_mask_uses_intersection() -> None:
    data = Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        edge_mask=torch.tensor([True, False, True, True]),
    )
    candidate = torch.tensor([True, True, False, True])

    selected = exact_ground_truth_candidate_edge_mask(data, candidate)

    assert selected.tolist() == [True, False, False, True]


def test_assert_gcnconv_cached_false_rejects_cached_model() -> None:
    model = nn.Sequential(GCNConv(2, 2, cached=True))

    with pytest.raises(ValueError, match="cached=False"):
        assert_gcnconv_cached_false(model)


def test_random_calibration_summary_reports_mean_and_std() -> None:
    runs = [
        CalibrationRun(edge_auroc=0.4, edge_ap=0.1, remove_delta=-0.1, retain_probability=0.2),
        CalibrationRun(edge_auroc=0.6, edge_ap=0.3, remove_delta=0.1, retain_probability=0.4),
    ]

    summary = summarize_runs(runs)

    assert summary["edge_auroc"]["mean"] == 0.5
    assert summary["edge_ap"]["mean"] == 0.2
    assert summary["remove_delta"]["mean"] == 0.0
    assert summary["retain_probability"]["mean"] == pytest.approx(0.3)
    assert summary["edge_auroc"]["std"] == pytest.approx(0.1414213562)
