from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    accuracy: float
    correct: int
    total: int
    motif_correct: int
    motif_total: int


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        raise ValueError("Cannot evaluate an empty mask")
    return torch.nn.functional.cross_entropy(logits[mask], labels[mask])


def evaluate_logits(logits: torch.Tensor, data: Data, mask: torch.Tensor) -> EvaluationMetrics:
    loss = masked_cross_entropy(logits, data.y, mask)
    predicted = logits.argmax(dim=-1)
    correct = int((predicted[mask] == data.y[mask]).sum().item())
    total = int(mask.sum().item())
    motif_mask = mask & (data.y > 0)
    motif_total = int(motif_mask.sum().item())
    motif_correct = int((predicted[motif_mask] == data.y[motif_mask]).sum().item())
    return EvaluationMetrics(
        loss=float(loss.item()),
        accuracy=correct / total,
        correct=correct,
        total=total,
        motif_correct=motif_correct,
        motif_total=motif_total,
    )


def majority_baseline_accuracy(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    eval_mask: torch.Tensor,
) -> float:
    train_labels = labels[train_mask]
    majority_label = torch.mode(train_labels).values
    correct = int((labels[eval_mask] == majority_label).sum().item())
    total = int(eval_mask.sum().item())
    return correct / total


def random_baseline_accuracy(num_classes: int) -> float:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return 1.0 / num_classes
