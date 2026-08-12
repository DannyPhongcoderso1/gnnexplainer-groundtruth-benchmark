from __future__ import annotations

import pytest

from gnnexplainer_benchmark.evaluation.schema import (
    CORE_METRICS_SCHEMA_VERSION,
    validate_core_metric_schema,
)


def valid_payload() -> dict:
    return {
        "schema_version": CORE_METRICS_SCHEMA_VERSION,
        "explainer": "shared-schema-test",
        "node_id": 465,
        "target": {"node_id": 465, "label": 1, "prediction": 1, "confidence": 0.27},
        "edge_ground_truth": {
            "candidate_edge_scope": "computation_subgraph",
            "candidate_edge_count": 434,
            "candidate_positive_edge_count": 12,
            "candidate_negative_edge_count": 422,
            "ground_truth_edge_prevalence": 12 / 434,
            "edge_mapping_validated": True,
            "edge_auroc": 0.93,
            "edge_auroc_valid": True,
            "edge_auroc_reason": None,
            "edge_ap": 0.21,
            "edge_ap_valid": True,
            "edge_ap_reason": None,
        },
        "sparsity": {
            "protocol": "top_k_candidate_edges",
            "top_k": 25,
            "selected_edge_count": 25,
            "sparsity": 1.0 - (25 / 434),
        },
        "fidelity": {
            "protocol": "remove_top_k_and_retain_top_k_edges",
            "metric_name": "remove_delta",
            "original": {"prediction": 1, "probability": 0.27},
            "remove_result": {
                "prediction": 1,
                "probability": 0.28,
                "remove_delta": -0.01,
                "changed_prediction": False,
            },
            "retain_result": {
                "prediction": 1,
                "probability": 0.29,
                "retain_delta": -0.02,
                "probability_ratio": 1.07,
                "matches_original_prediction": True,
            },
        },
        "runtime": {"runtime_seconds": 0.4},
    }


def test_core_metric_schema_accepts_locked_shared_output() -> None:
    validate_core_metric_schema(valid_payload())


def test_core_metric_schema_rejects_non_locked_top_k() -> None:
    payload = valid_payload()
    payload["sparsity"]["top_k"] = 10

    with pytest.raises(ValueError, match="top_k"):
        validate_core_metric_schema(payload)
