from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class NodePrediction:
    prediction: int
    probability: float
    logits: list[float]


@dataclass(frozen=True)
class FidelityResult:
    protocol: str
    metric_name: str
    original_prediction: int
    original_probability: float
    remove_prediction: int
    remove_probability: float
    remove_delta: float
    remove_probability_delta: float
    retain_prediction: int
    retain_probability: float
    retain_delta: float
    retain_probability_delta: float
    fid_plus: int
    fid_minus: int
    one_minus_fid_minus: int
    remove_probability_drop: float
    retain_probability_ratio: float | None
    remove_changed_prediction: bool
    retain_matches_original_prediction: bool


def predict_node(
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    class_id: int | None = None,
) -> NodePrediction:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = logits.softmax(dim=-1)
    prediction = int(logits[node_id].argmax(dim=-1).item())
    measured_class = prediction if class_id is None else class_id
    return NodePrediction(
        prediction=prediction,
        probability=float(probabilities[node_id, measured_class].item()),
        logits=[float(value) for value in logits[node_id].detach().cpu().tolist()],
    )


def compute_fidelity(
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    selected_edge_mask: torch.Tensor,
) -> FidelityResult:
    if selected_edge_mask.numel() != data.edge_index.size(1):
        raise ValueError("selected_edge_mask must align with data.edge_index")
    if selected_edge_mask.dtype != torch.bool:
        raise ValueError("selected_edge_mask must be boolean")

    original = predict_node(model, data, node_id)
    original_class = original.prediction

    remove_data = data.clone()
    remove_data.edge_index = data.edge_index[:, ~selected_edge_mask]
    remove = predict_node(model, remove_data, node_id, class_id=original_class)

    retain_data = data.clone()
    retain_data.edge_index = data.edge_index[:, selected_edge_mask]
    retain = predict_node(model, retain_data, node_id, class_id=original_class)

    retain_ratio = None
    if original.probability != 0.0:
        retain_ratio = retain.probability / original.probability

    return FidelityResult(
        protocol="remove_top_k_and_retain_top_k_edges",
        metric_name="remove_delta",
        original_prediction=original.prediction,
        original_probability=original.probability,
        remove_prediction=remove.prediction,
        remove_probability=remove.probability,
        remove_delta=original.probability - remove.probability,
        remove_probability_delta=original.probability - remove.probability,
        retain_prediction=retain.prediction,
        retain_probability=retain.probability,
        retain_delta=original.probability - retain.probability,
        retain_probability_delta=original.probability - retain.probability,
        fid_plus=fid_plus(original.prediction, remove.prediction),
        fid_minus=fid_minus(original.prediction, retain.prediction),
        one_minus_fid_minus=1 - fid_minus(original.prediction, retain.prediction),
        remove_probability_drop=original.probability - remove.probability,
        retain_probability_ratio=retain_ratio,
        remove_changed_prediction=remove.prediction != original.prediction,
        retain_matches_original_prediction=retain.prediction == original.prediction,
    )


def fid_plus(original_prediction: int, remove_prediction: int) -> int:
    return int(remove_prediction != original_prediction)


def fid_minus(original_prediction: int, retain_prediction: int) -> int:
    return int(retain_prediction != original_prediction)
