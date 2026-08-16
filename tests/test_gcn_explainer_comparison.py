from __future__ import annotations

import pytest

from scripts.build_gcn_explainer_comparison import (
    QUALITY_METRICS,
    build_paired_delta_rows,
    runtime_semantics,
    validate_method_parity,
)


def test_comparison_parity_rejects_candidate_mismatch() -> None:
    reference = {
        1: {
            "candidate_edge_count": "10",
            "positive_gt_edges": "2",
            "negative_gt_edges": "8",
            "selected_edge_count": "2",
            "original_prediction": "1",
        }
    }
    rows = [
        {
            "node_id": "1",
            "candidate_edge_count": "11",
            "positive_gt_edges": "2",
            "negative_gt_edges": "8",
            "selected_edge_count": "2",
            "original_prediction": "1",
            "selected_edge_ratio": "0.2",
        }
    ]

    with pytest.raises(ValueError, match="candidate_edge_count"):
        validate_method_parity("test", rows, reference)


def test_comparison_paired_deltas_keep_direction_labels() -> None:
    row = {"node_id": 1}
    for metric in QUALITY_METRICS:
        row[f"pg_{metric}"] = 3.0
        row[f"gnn_{metric}"] = 2.0
        row[f"random_{metric}"] = 1.0

    delta = build_paired_delta_rows([row])[0]

    assert delta["pg_minus_gnn_edge_auroc"] == pytest.approx(1.0)
    assert delta["gnn_minus_random_edge_auroc"] == pytest.approx(1.0)
    assert delta["pg_minus_random_edge_auroc"] == pytest.approx(2.0)


def test_comparison_runtime_semantics_are_decomposed() -> None:
    assert "upfront training" in runtime_semantics("PGExplainer")
    assert "optimization" in runtime_semantics("GNNExplainer")
    assert "random edge selection" in runtime_semantics("Random")
