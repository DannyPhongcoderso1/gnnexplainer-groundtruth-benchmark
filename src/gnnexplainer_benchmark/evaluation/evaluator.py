from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data

from gnnexplainer_benchmark.evaluation.edge_ground_truth import (
    candidate_edge_scores_and_labels,
    compute_edge_ground_truth_metrics,
)
from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity
from gnnexplainer_benchmark.evaluation.schema import (
    CORE_METRICS_SCHEMA_VERSION,
    validate_core_metric_schema,
)
from gnnexplainer_benchmark.evaluation.sparsity import compute_sparsity, select_top_k_edges


def evaluate_node_explanation(
    *,
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    edge_scores: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
    top_k: int,
    runtime_seconds: float,
    explainer: str = "unknown",
    target: dict[str, Any] | None = None,
    candidate_edge_scope: str = "computation_subgraph",
) -> dict[str, Any]:
    candidate_scores, candidate_ground_truth = candidate_edge_scores_and_labels(
        edge_scores=edge_scores.detach().cpu(),
        ground_truth_edge_mask=data.edge_mask.detach().cpu(),
        candidate_edge_mask=candidate_edge_mask.detach().cpu(),
    )
    edge_metrics = compute_edge_ground_truth_metrics(candidate_scores, candidate_ground_truth)
    selected_edge_mask = select_top_k_edges(
        edge_scores=edge_scores.detach().cpu(),
        candidate_edge_mask=candidate_edge_mask.detach().cpu(),
        top_k=top_k,
    ).to(data.edge_index.device)
    sparsity = compute_sparsity(
        candidate_edge_mask=candidate_edge_mask.detach().cpu(),
        selected_edge_mask=selected_edge_mask.detach().cpu(),
        top_k=top_k,
    )
    fidelity = compute_fidelity(
        model=model,
        data=data,
        node_id=node_id,
        selected_edge_mask=selected_edge_mask,
    )

    positive_edge_count = int(candidate_ground_truth.sum().item())
    negative_edge_count = int(candidate_ground_truth.numel() - positive_edge_count)
    ground_truth_edge_prevalence = positive_edge_count / sparsity.candidate_edge_count
    result = {
        "schema_version": CORE_METRICS_SCHEMA_VERSION,
        "explainer": explainer,
        "node_id": node_id,
        "target": target or {},
        "edge_ground_truth": {
            "candidate_edge_scope": candidate_edge_scope,
            "candidate_edge_count": sparsity.candidate_edge_count,
            "candidate_positive_edge_count": positive_edge_count,
            "candidate_negative_edge_count": negative_edge_count,
            "ground_truth_edge_prevalence": ground_truth_edge_prevalence,
            "edge_mapping_validated": True,
            "edge_auroc": edge_metrics["edge_auroc"].value,
            "edge_auroc_valid": edge_metrics["edge_auroc"].valid,
            "edge_auroc_reason": edge_metrics["edge_auroc"].reason,
            "edge_ap": edge_metrics["edge_ap"].value,
            "edge_ap_valid": edge_metrics["edge_ap"].valid,
            "edge_ap_reason": edge_metrics["edge_ap"].reason,
        },
        "sparsity": {
            "protocol": sparsity.protocol,
            "top_k": sparsity.top_k,
            "selected_edge_count": sparsity.selected_edge_count,
            "sparsity": sparsity.sparsity,
        },
        "fidelity": {
            "protocol": fidelity.protocol,
            "metric_name": fidelity.metric_name,
            "original": {
                "prediction": fidelity.original_prediction,
                "probability": fidelity.original_probability,
            },
            "remove_result": {
                "prediction": fidelity.remove_prediction,
                "probability": fidelity.remove_probability,
                "remove_delta": fidelity.remove_delta,
                "changed_prediction": fidelity.remove_changed_prediction,
            },
            "retain_result": {
                "prediction": fidelity.retain_prediction,
                "probability": fidelity.retain_probability,
                "retain_delta": fidelity.retain_delta,
                "probability_ratio": fidelity.retain_probability_ratio,
                "matches_original_prediction": fidelity.retain_matches_original_prediction,
            },
        },
        "runtime": {
            "runtime_seconds": runtime_seconds,
        },
    }
    validate_core_metric_schema(result)
    return result
