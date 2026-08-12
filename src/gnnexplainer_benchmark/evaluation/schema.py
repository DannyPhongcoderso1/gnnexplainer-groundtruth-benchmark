from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CORE_METRICS_SCHEMA_VERSION = "core_metrics_v1"

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "explainer",
    "node_id",
    "target",
    "edge_ground_truth",
    "sparsity",
    "fidelity",
    "runtime",
}

REQUIRED_EDGE_GROUND_TRUTH_KEYS = {
    "candidate_edge_scope",
    "candidate_edge_count",
    "candidate_positive_edge_count",
    "candidate_negative_edge_count",
    "ground_truth_edge_prevalence",
    "edge_mapping_validated",
    "edge_auroc",
    "edge_auroc_valid",
    "edge_auroc_reason",
    "edge_ap",
    "edge_ap_valid",
    "edge_ap_reason",
}

REQUIRED_SPARSITY_KEYS = {
    "protocol",
    "top_k",
    "selected_edge_count",
    "sparsity",
}

REQUIRED_FIDELITY_KEYS = {
    "protocol",
    "metric_name",
    "original",
    "remove_result",
    "retain_result",
}

REQUIRED_RUNTIME_KEYS = {"runtime_seconds"}


def validate_core_metric_schema(result: Mapping[str, Any]) -> None:
    _require_keys(result, REQUIRED_TOP_LEVEL_KEYS, "core metric result")
    if result["schema_version"] != CORE_METRICS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {result['schema_version']}")
    _require_keys(result["edge_ground_truth"], REQUIRED_EDGE_GROUND_TRUTH_KEYS, "edge_ground_truth")
    _require_keys(result["sparsity"], REQUIRED_SPARSITY_KEYS, "sparsity")
    _require_keys(result["fidelity"], REQUIRED_FIDELITY_KEYS, "fidelity")
    _require_keys(result["runtime"], REQUIRED_RUNTIME_KEYS, "runtime")
    if result["fidelity"]["metric_name"] != "remove_delta":
        raise ValueError("fidelity.metric_name must be remove_delta")
    if result["sparsity"]["protocol"] != "top_k_candidate_edges":
        raise ValueError("sparsity.protocol must be top_k_candidate_edges")
    if result["sparsity"]["top_k"] != 25:
        raise ValueError("Phase 4B gate locks top_k at 25")


def _require_keys(payload: Mapping[str, Any], keys: set[str], context: str) -> None:
    missing = keys - set(payload)
    if missing:
        raise ValueError(f"Missing keys in {context}: {sorted(missing)}")
