from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data

SPLIT_MASKS: dict[str, str] = {
    "train": "train_mask",
    "validation": "val_mask",
    "test": "test_mask",
}


def split_mask(data: Data, split: str) -> torch.Tensor:
    if split not in SPLIT_MASKS:
        raise ValueError(f"Unknown split: {split}")
    attr = SPLIT_MASKS[split]
    if not hasattr(data, attr):
        raise ValueError(f"Missing split mask: {attr}")
    mask = getattr(data, attr)
    if mask.dtype != torch.bool:
        raise ValueError(f"{attr} must be a boolean tensor")
    return mask


def accuracy_for_mask(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    total = int(mask.sum().item())
    if total == 0:
        return {"accuracy": None, "correct": 0, "total": 0}
    correct = int((predictions[mask] == labels[mask]).sum().item())
    return {"accuracy": correct / total, "correct": correct, "total": total}


def per_class_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> dict[str, dict[str, Any]]:
    metrics = {}
    for class_id in range(num_classes):
        class_mask = mask & (labels == class_id)
        metrics[str(class_id)] = accuracy_for_mask(predictions, labels, class_mask)
    return metrics


def confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> list[list[int]]:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    labels_for_mask = labels[mask].tolist()
    predictions_for_mask = predictions[mask].tolist()
    for true_label, prediction in zip(labels_for_mask, predictions_for_mask, strict=True):
        matrix[int(true_label), int(prediction)] += 1
    return matrix.tolist()


def prediction_distribution(
    predictions: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> dict[str, int]:
    values = predictions[mask]
    return {
        str(class_id): int((values == class_id).sum().item())
        for class_id in range(num_classes)
    }


def correctly_predicted_motif_nodes(
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    data: Data,
    split: str,
) -> list[dict[str, Any]]:
    mask = split_mask(data, split)
    motif_mask = data.node_mask.bool()
    target_mask = mask & motif_mask & (predictions == data.y)
    rows: list[dict[str, Any]] = []
    for node_id in torch.nonzero(target_mask, as_tuple=False).view(-1).tolist():
        prediction = int(predictions[node_id].item())
        rows.append(
            {
                "node_id": int(node_id),
                "label": int(data.y[node_id].item()),
                "prediction": prediction,
                "confidence": float(probabilities[node_id, prediction].item()),
            }
        )
    return sorted(rows, key=lambda row: (-row["confidence"], row["node_id"]))


def build_model_quality_report(
    logits: torch.Tensor,
    data: Data,
) -> dict[str, Any]:
    logits = logits.detach().cpu()
    data = data.cpu()
    probabilities = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)
    num_classes = int(data.y.max().item()) + 1

    splits: dict[str, Any] = {}
    for split in SPLIT_MASKS:
        mask = split_mask(data, split)
        motif_mask = mask & data.node_mask.bool()
        motif_correct_mask = motif_mask & (predictions == data.y)
        splits[split] = {
            "overall_accuracy": accuracy_for_mask(predictions, data.y, mask),
            "per_class_accuracy": per_class_accuracy(predictions, data.y, mask, num_classes),
            "confusion_matrix": confusion_matrix(predictions, data.y, mask, num_classes),
            "motif_nodes": int(motif_mask.sum().item()),
            "correctly_predicted_motif_nodes": int(motif_correct_mask.sum().item()),
            "prediction_distribution": prediction_distribution(predictions, mask, num_classes),
        }

    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_classes": num_classes,
        "splits": splits,
        "correctly_predicted_motif_test_nodes": correctly_predicted_motif_nodes(
            probabilities,
            predictions,
            data,
            "test",
        ),
    }
