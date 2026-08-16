from __future__ import annotations

from pathlib import Path

import pytest

from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import read_explainer_targets
from gnnexplainer_benchmark.explainers.pgexplainer_evaluation import (
    PGEXPLAINER_PER_NODE_FIELDS,
    PGEXPLAINER_SUMMARY_METRICS,
    read_phase4_reference_rows,
    validate_target_order,
    write_pgexplainer_summary_metrics,
)


def test_pgexplainer_eval_schema_has_pg_specific_runtime_fields() -> None:
    assert "explainer_checkpoint" in PGEXPLAINER_PER_NODE_FIELDS
    assert "explainer_training_seed" in PGEXPLAINER_PER_NODE_FIELDS
    assert "inference_runtime_seconds" in PGEXPLAINER_PER_NODE_FIELDS
    assert "generation_runtime_seconds" not in PGEXPLAINER_PER_NODE_FIELDS
    assert "inference_runtime_seconds" in PGEXPLAINER_SUMMARY_METRICS


def test_pgexplainer_eval_target_order_matches_phase4_reference() -> None:
    targets = read_explainer_targets(Path("results/metrics/explainer_test_targets.csv"))
    phase4_rows = read_phase4_reference_rows(
        Path("results/metrics/frozen_phase4c_v2/per_node_metrics.csv")
    )

    validate_target_order(targets, phase4_rows)


def test_pgexplainer_eval_target_order_rejects_mismatch() -> None:
    targets = read_explainer_targets(Path("results/metrics/explainer_test_targets.csv"))
    phase4_rows = read_phase4_reference_rows(
        Path("results/metrics/frozen_phase4c_v2/per_node_metrics.csv")
    )
    reversed_rows = dict(reversed(list(phase4_rows.items())))

    with pytest.raises(ValueError, match="target order"):
        validate_target_order(targets, reversed_rows)


def test_pgexplainer_summary_writes_inference_runtime(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    summary = write_pgexplainer_summary_metrics(
        path,
        [
            {
                "status": "success",
                "edge_auroc": 0.5,
                "edge_ap": 0.4,
                "fid_plus": 1,
                "fid_minus": 0,
                "one_minus_fid_minus": 1,
                "remove_probability_delta": 0.1,
                "retain_probability": 0.2,
                "retain_probability_delta": -0.1,
                "retain_probability_ratio": 0.8,
                "fidelity_legacy": 0.1,
                "sparsity": 0.8,
                "inference_runtime_seconds": 0.03,
            }
        ],
    )

    by_metric = {row["metric"]: row for row in summary}
    assert by_metric["inference_runtime_seconds"]["mean"] == pytest.approx(0.03)
