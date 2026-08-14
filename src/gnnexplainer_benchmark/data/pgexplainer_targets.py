from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask


@dataclass(frozen=True)
class PGExplainerTarget:
    node_id: int
    ground_truth_label: int
    frozen_model_prediction: int
    target_for_pgexplainer: int
    confidence: float
    base_model_split: str
    is_ground_truth_motif: bool
    gcn_prediction_correct: bool
    candidate_edge_count: int
    candidate_edge_mapping_valid: bool


def split_mask(data: Data, split: str) -> torch.Tensor:
    if split == "train":
        mask_name = "train_mask"
    elif split in {"validation", "val"}:
        mask_name = "val_mask"
    elif split == "test":
        mask_name = "test_mask"
    else:
        raise ValueError(f"Unsupported split: {split}")
    if not hasattr(data, mask_name):
        raise ValueError(f"Missing {mask_name}")
    mask = getattr(data, mask_name)
    if mask.dtype != torch.bool:
        raise ValueError(f"{mask_name} must be boolean")
    return mask.detach().cpu()


def select_pgexplainer_targets(
    *,
    logits: torch.Tensor,
    data: Data,
    split: str,
    num_hops: int,
    excluded_node_ids: set[int] | None = None,
) -> list[PGExplainerTarget]:
    if not hasattr(data, "node_mask"):
        raise ValueError("Missing node_mask")
    excluded_node_ids = excluded_node_ids or set()
    logits = logits.detach().cpu()
    data = data.cpu()
    probabilities = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)
    base_mask = split_mask(data, split)
    target_mask = base_mask & data.node_mask.bool() & (predictions == data.y)

    targets: list[PGExplainerTarget] = []
    for node_id in torch.nonzero(target_mask, as_tuple=False).view(-1).tolist():
        node_id = int(node_id)
        if node_id in excluded_node_ids:
            continue
        candidate_edge_mapping_valid = True
        try:
            candidate_edge_mask = computation_subgraph_edge_mask(data, node_id, num_hops)
            candidate_edge_count = int(candidate_edge_mask.sum().item())
            if candidate_edge_count <= 0:
                candidate_edge_mapping_valid = False
        except Exception:
            candidate_edge_count = 0
            candidate_edge_mapping_valid = False
        if not candidate_edge_mapping_valid:
            continue
        prediction = int(predictions[node_id].item())
        targets.append(
            PGExplainerTarget(
                node_id=node_id,
                ground_truth_label=int(data.y[node_id].item()),
                frozen_model_prediction=prediction,
                target_for_pgexplainer=prediction,
                confidence=float(probabilities[node_id, prediction].item()),
                base_model_split="validation" if split == "val" else split,
                is_ground_truth_motif=True,
                gcn_prediction_correct=True,
                candidate_edge_count=candidate_edge_count,
                candidate_edge_mapping_valid=True,
            )
        )
    return sorted(targets, key=lambda target: target.node_id)


def validate_pgexplainer_target_splits(
    train_targets: list[PGExplainerTarget],
    validation_targets: list[PGExplainerTarget],
    test_node_ids: set[int],
) -> None:
    train_ids = {target.node_id for target in train_targets}
    validation_ids = {target.node_id for target in validation_targets}
    if train_ids & validation_ids:
        raise ValueError(
            f"PGExplainer train/validation overlap: {sorted(train_ids & validation_ids)}"
        )
    if train_ids & test_node_ids:
        raise ValueError(f"PGExplainer train/test leakage: {sorted(train_ids & test_node_ids)}")
    if validation_ids & test_node_ids:
        raise ValueError(
            f"PGExplainer validation/test leakage: {sorted(validation_ids & test_node_ids)}"
        )


def write_pgexplainer_targets(path: Path, targets: list[PGExplainerTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "node_id",
        "ground_truth_label",
        "frozen_model_prediction",
        "target_for_pgexplainer",
        "confidence",
        "base_model_split",
        "is_ground_truth_motif",
        "gcn_prediction_correct",
        "candidate_edge_count",
        "candidate_edge_mapping_valid",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target in targets:
            writer.writerow({field: getattr(target, field) for field in fieldnames})
