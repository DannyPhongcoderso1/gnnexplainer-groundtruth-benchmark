from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class TargetNode:
    node_id: int
    label: int
    prediction: int
    confidence: float


def correctly_predicted_motif_test_nodes(logits: torch.Tensor, data: Data) -> list[TargetNode]:
    probabilities = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)
    target_mask = data.test_mask & (data.y > 0) & (predictions == data.y)
    node_ids = torch.nonzero(target_mask, as_tuple=False).view(-1).tolist()
    targets = []
    for node_id in node_ids:
        prediction = int(predictions[node_id].item())
        targets.append(
            TargetNode(
                node_id=int(node_id),
                label=int(data.y[node_id].item()),
                prediction=prediction,
                confidence=float(probabilities[node_id, prediction].item()),
            )
        )
    return sorted(targets, key=lambda item: (-item.confidence, item.node_id))


def write_target_nodes(path: Path, targets: list[TargetNode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "label", "prediction", "confidence"],
        )
        writer.writeheader()
        for target in targets:
            writer.writerow(
                {
                    "node_id": target.node_id,
                    "label": target.label,
                    "prediction": target.prediction,
                    "confidence": target.confidence,
                }
            )


def select_target_node(logits: torch.Tensor, data: Data) -> TargetNode:
    targets = correctly_predicted_motif_test_nodes(logits, data)
    if not targets:
        raise ValueError("No correctly predicted motif test nodes are available")
    return targets[0]
