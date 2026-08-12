from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from gnnexplainer_benchmark.evaluation.edge_ground_truth import (
    candidate_edge_scores_and_labels,
    compute_edge_ground_truth_metrics,
)
from gnnexplainer_benchmark.evaluation.fidelity import FidelityResult, compute_fidelity
from gnnexplainer_benchmark.evaluation.sparsity import select_top_k_edges


@dataclass(frozen=True)
class CalibrationRun:
    edge_auroc: float | None
    edge_ap: float | None
    remove_delta: float
    retain_probability: float


def assert_gcnconv_cached_false(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, GCNConv) and bool(module.cached):
            raise ValueError("Fidelity perturbation requires GCNConv(cached=False)")


def exact_ground_truth_candidate_edge_mask(
    data: Data,
    candidate_edge_mask: torch.Tensor,
) -> torch.Tensor:
    if data.edge_mask.numel() != candidate_edge_mask.numel():
        raise ValueError("data.edge_mask and candidate_edge_mask must align")
    return data.edge_mask.detach().cpu().bool() & candidate_edge_mask.detach().cpu().bool()


def oracle_edge_scores(data: Data, candidate_edge_mask: torch.Tensor) -> torch.Tensor:
    scores = torch.zeros(data.edge_index.size(1), dtype=torch.float)
    scores[exact_ground_truth_candidate_edge_mask(data.cpu(), candidate_edge_mask)] = 1.0
    return scores


def summarize_runs(runs: list[CalibrationRun]) -> dict[str, dict[str, float | None]]:
    if not runs:
        raise ValueError("runs must not be empty")
    return {
        "edge_auroc": _mean_std([run.edge_auroc for run in runs]),
        "edge_ap": _mean_std([run.edge_ap for run in runs]),
        "remove_delta": _mean_std([run.remove_delta for run in runs]),
        "retain_probability": _mean_std([run.retain_probability for run in runs]),
    }


def random_top_k_calibration(
    *,
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    candidate_edge_mask: torch.Tensor,
    top_k: int,
    repeats: int,
    seed: int,
) -> tuple[list[CalibrationRun], dict[str, dict[str, float | None]]]:
    runs: list[CalibrationRun] = []
    candidate_cpu = candidate_edge_mask.detach().cpu().bool()
    for idx in range(repeats):
        generator = torch.Generator().manual_seed(seed + idx)
        edge_scores = torch.zeros(data.edge_index.size(1), dtype=torch.float)
        random_scores = torch.rand(int(candidate_cpu.sum().item()), generator=generator)
        edge_scores[candidate_cpu] = random_scores
        selected = select_top_k_edges(edge_scores, candidate_cpu, top_k).to(data.edge_index.device)
        metrics = _edge_metrics_for_scores(edge_scores, data.clone().cpu(), candidate_cpu)
        fidelity = compute_fidelity(model, data, node_id, selected)
        runs.append(
            CalibrationRun(
                edge_auroc=metrics["edge_auroc"],
                edge_ap=metrics["edge_ap"],
                remove_delta=fidelity.remove_delta,
                retain_probability=fidelity.retain_probability,
            )
        )
    return runs, summarize_runs(runs)


def edge_and_fidelity_for_selected_mask(
    *,
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    candidate_edge_mask: torch.Tensor,
    edge_scores: torch.Tensor,
    selected_edge_mask: torch.Tensor,
) -> dict[str, Any]:
    metrics = _edge_metrics_for_scores(
        edge_scores.detach().cpu(),
        data.clone().cpu(),
        candidate_edge_mask,
    )
    fidelity = compute_fidelity(
        model=model,
        data=data,
        node_id=node_id,
        selected_edge_mask=selected_edge_mask.to(data.edge_index.device),
    )
    return {
        **metrics,
        "fidelity": fidelity_to_dict(fidelity),
    }


def fidelity_to_dict(fidelity: FidelityResult) -> dict[str, Any]:
    return {
        "original_prediction": fidelity.original_prediction,
        "original_probability": fidelity.original_probability,
        "remove_prediction": fidelity.remove_prediction,
        "remove_probability": fidelity.remove_probability,
        "remove_delta": fidelity.remove_delta,
        "retain_prediction": fidelity.retain_prediction,
        "retain_probability": fidelity.retain_probability,
        "retain_delta": fidelity.retain_delta,
        "retain_probability_ratio": fidelity.retain_probability_ratio,
    }


def _edge_metrics_for_scores(
    edge_scores: torch.Tensor,
    data: Data,
    candidate_edge_mask: torch.Tensor,
) -> dict[str, float | None]:
    candidate_scores, candidate_ground_truth = candidate_edge_scores_and_labels(
        edge_scores=edge_scores,
        ground_truth_edge_mask=data.edge_mask.detach().cpu(),
        candidate_edge_mask=candidate_edge_mask.detach().cpu(),
    )
    edge_metrics = compute_edge_ground_truth_metrics(candidate_scores, candidate_ground_truth)
    return {
        "edge_auroc": edge_metrics["edge_auroc"].value,
        "edge_ap": edge_metrics["edge_ap"].value,
    }


def _mean_std(values: list[float | None]) -> dict[str, float | None]:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return {"mean": None, "std": None}
    if len(valid_values) == 1:
        return {"mean": valid_values[0], "std": 0.0}
    return {"mean": mean(valid_values), "std": stdev(valid_values)}
