from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch_geometric.data import Data

from gnnexplainer_benchmark.models import GCN
from gnnexplainer_benchmark.training.checkpointing import save_checkpoint
from gnnexplainer_benchmark.training.evaluation import (
    evaluate_logits,
    majority_baseline_accuracy,
    random_baseline_accuracy,
)
from gnnexplainer_benchmark.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_accuracy: float
    test_accuracy: float
    test_correct: int
    test_total: int
    motif_test_correct: int
    motif_test_total: int
    majority_baseline_accuracy: float
    random_baseline_accuracy: float
    checkpoint_path: str
    history_path: str
    summary_path: str


def load_training_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError("Config must contain a top-level 'model' mapping")
    return config


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = config.get("training", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(requested))


def build_gcn_model(data: Data, config: dict[str, Any]) -> GCN:
    model_config = config["model"]
    out_channels = int(data.y.max().item()) + 1
    return GCN(
        in_channels=int(data.x.size(1)),
        hidden_channels=int(model_config["hidden_channels"]),
        out_channels=out_channels,
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        cached=bool(model_config["cached"]),
    )


def train_gcn(
    data: Data,
    config: dict[str, Any],
    checkpoint_path: Path,
    history_path: Path,
    summary_path: Path,
    dataset_checksum: str,
) -> TrainingResult:
    resolved_config = deepcopy(config)
    seed = int(resolved_config["model"]["seed"])
    seed_everything(seed)
    device = resolve_device(resolved_config)
    data = data.to(device)
    model = build_gcn_model(data, resolved_config).to(device)

    training_config = resolved_config.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.01)),
        weight_decay=float(training_config.get("weight_decay", 5e-4)),
    )
    class_weights = None
    if training_config.get("class_weighting"):
        class_weights = _class_weights(data.y, data.train_mask)
    max_epochs = int(training_config.get("max_epochs", 300))
    patience = int(training_config.get("patience", max_epochs))
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            train_metrics = evaluate_logits(logits, data, data.train_mask)
            val_metrics = evaluate_logits(logits, data, data.val_mask)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
            }
        )

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_metrics = evaluate_logits(logits, data, data.test_mask)

    save_checkpoint(
        checkpoint_path,
        model,
        resolved_config,
        seed,
        best_epoch,
        best_val_accuracy,
        dataset_checksum,
    )
    _write_history(history_path, history)
    num_classes = int(data.y.max().item()) + 1
    majority_accuracy = majority_baseline_accuracy(data.y, data.train_mask, data.test_mask)
    random_accuracy = random_baseline_accuracy(num_classes)
    result = TrainingResult(
        best_epoch=best_epoch,
        best_validation_accuracy=best_val_accuracy,
        test_accuracy=test_metrics.accuracy,
        test_correct=test_metrics.correct,
        test_total=test_metrics.total,
        motif_test_correct=test_metrics.motif_correct,
        motif_test_total=test_metrics.motif_total,
        majority_baseline_accuracy=majority_accuracy,
        random_baseline_accuracy=random_accuracy,
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
        summary_path=str(summary_path),
    )
    _write_summary(summary_path, result, resolved_config, dataset_checksum, device)
    return result


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)


def _class_weights(labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    train_labels = labels[train_mask]
    num_classes = int(labels.max().item()) + 1
    counts = torch.bincount(train_labels, minlength=num_classes).float()
    weights = train_labels.numel() / (num_classes * counts.clamp_min(1.0))
    return weights.to(labels.device)


def _write_summary(
    path: Path,
    result: TrainingResult,
    config: dict[str, Any],
    dataset_checksum: str,
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(result),
        "resolved_config": config,
        "dataset_checksum": dataset_checksum,
        "device": str(device),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
