from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class ExplainerTarget:
    node_id: int
    label: int
    prediction: int
    confidence: float
    split: str
    is_ground_truth_motif: bool


def select_correct_motif_targets(
    logits: torch.Tensor,
    data: Data,
    split: str = "test",
) -> list[ExplainerTarget]:
    if split != "test":
        raise ValueError("Phase 4A target selection is restricted to the test split")
    if not hasattr(data, "test_mask"):
        raise ValueError("Missing test_mask")
    if not hasattr(data, "node_mask"):
        raise ValueError("Missing node_mask")

    logits = logits.detach().cpu()
    data = data.cpu()
    probabilities = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)
    target_mask = data.test_mask & data.node_mask.bool() & (predictions == data.y)
    targets: list[ExplainerTarget] = []
    for node_id in torch.nonzero(target_mask, as_tuple=False).view(-1).tolist():
        prediction = int(predictions[node_id].item())
        targets.append(
            ExplainerTarget(
                node_id=int(node_id),
                label=int(data.y[node_id].item()),
                prediction=prediction,
                confidence=float(probabilities[node_id, prediction].item()),
                split=split,
                is_ground_truth_motif=True,
            )
        )
    return sorted(targets, key=lambda target: (-target.confidence, target.node_id))


def split_explainer_targets(
    targets: list[ExplainerTarget],
    seed: int,
    validation_fraction: float = 0.25,
) -> tuple[list[ExplainerTarget], list[ExplainerTarget]]:
    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled = list(targets)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return [], sorted(shuffled, key=lambda target: target.node_id)

    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation_targets = sorted(shuffled[:validation_count], key=lambda target: target.node_id)
    test_targets = sorted(shuffled[validation_count:], key=lambda target: target.node_id)
    validate_non_overlapping_targets(validation_targets, test_targets)
    return validation_targets, test_targets


def validate_non_overlapping_targets(
    validation_targets: list[ExplainerTarget],
    test_targets: list[ExplainerTarget],
) -> None:
    validation_ids = {target.node_id for target in validation_targets}
    test_ids = {target.node_id for target in test_targets}
    overlap = validation_ids & test_ids
    if overlap:
        raise ValueError(f"Validation/test target overlap: {sorted(overlap)}")


def write_explainer_targets(path: Path, targets: list[ExplainerTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "node_id",
                "label",
                "prediction",
                "confidence",
                "split",
                "is_ground_truth_motif",
            ],
        )
        writer.writeheader()
        for target in targets:
            writer.writerow(
                {
                    "node_id": target.node_id,
                    "label": target.label,
                    "prediction": target.prediction,
                    "confidence": target.confidence,
                    "split": target.split,
                    "is_ground_truth_motif": target.is_ground_truth_motif,
                }
            )
