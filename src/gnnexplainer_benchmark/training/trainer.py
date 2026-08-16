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

from gnnexplainer_benchmark.models import GCN, GraphSAGE
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
    best_validation_metric: float
    best_validation_metric_name: str
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


def build_model(data: Data, config: dict[str, Any]) -> torch.nn.Module:
    model_config = config["model"]
    model_type = str(model_config.get("type", "GCN"))
    out_channels = int(data.y.max().item()) + 1
    common_args = {
        "in_channels": int(data.x.size(1)),
        "hidden_channels": int(model_config["hidden_channels"]),
        "out_channels": out_channels,
        "num_layers": int(model_config["num_layers"]),
        "dropout": float(model_config["dropout"]),
    }
    if model_type == "GCN":
        return GCN(**common_args, cached=bool(model_config["cached"]))
    if model_type == "GraphSAGE":
        return GraphSAGE(**common_args, aggr=str(model_config.get("aggr", "mean")))
    raise ValueError(f"Unsupported model type: {model_type}")


def build_gcn_model(data: Data, config: dict[str, Any]) -> GCN:
    model = build_model(data, config)
    if not isinstance(model, GCN):
        raise TypeError("Config did not build a GCN model")
    return model


def build_graphsage_model(data: Data, config: dict[str, Any]) -> GraphSAGE:
    model = build_model(data, config)
    if not isinstance(model, GraphSAGE):
        raise TypeError("Config did not build a GraphSAGE model")
    return model


def train_gcn(
    data: Data,
    config: dict[str, Any],
    checkpoint_path: Path,
    history_path: Path,
    summary_path: Path,
    dataset_checksum: str,
) -> TrainingResult:
    return train_model(
        data=data,
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        summary_path=summary_path,
        dataset_checksum=dataset_checksum,
    )


def train_graphsage(
    data: Data,
    config: dict[str, Any],
    checkpoint_path: Path,
    history_path: Path,
    summary_path: Path,
    dataset_checksum: str,
) -> TrainingResult:
    return train_model(
        data=data,
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        summary_path=summary_path,
        dataset_checksum=dataset_checksum,
    )


def train_model(
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
    model = build_model(data, resolved_config).to(device)

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
    monitor = str(resolved_config.get("checkpoint", {}).get("monitor", "val_accuracy"))
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_metric = -1.0
    best_val_accuracy_at_best = -1.0
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
            train_balanced_accuracy = balanced_accuracy(logits, data, data.train_mask)
            val_balanced_accuracy = balanced_accuracy(logits, data, data.val_mask)
            monitor_value = metric_value(
                monitor,
                train_metrics,
                val_metrics,
                train_balanced_accuracy,
                val_balanced_accuracy,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "train_motif_accuracy": motif_accuracy(train_metrics),
                "train_balanced_accuracy": train_balanced_accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_motif_accuracy": motif_accuracy(val_metrics),
                "val_balanced_accuracy": val_balanced_accuracy,
                "monitor_metric": monitor,
                "monitor_value": monitor_value,
            }
        )

        if monitor_value > best_validation_metric:
            best_validation_metric = monitor_value
            best_val_accuracy_at_best = val_metrics.accuracy
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
        best_validation_metric,
        dataset_checksum,
    )
    _write_history(history_path, history)
    num_classes = int(data.y.max().item()) + 1
    majority_accuracy = majority_baseline_accuracy(data.y, data.train_mask, data.test_mask)
    random_accuracy = random_baseline_accuracy(num_classes)
    result = TrainingResult(
        best_epoch=best_epoch,
        best_validation_metric=best_validation_metric,
        best_validation_metric_name=monitor,
        best_validation_accuracy=best_val_accuracy_at_best,
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
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_motif_accuracy",
                "train_balanced_accuracy",
                "val_loss",
                "val_accuracy",
                "val_motif_accuracy",
                "val_balanced_accuracy",
                "monitor_metric",
                "monitor_value",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def _class_weights(labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    train_labels = labels[train_mask]
    num_classes = int(labels.max().item()) + 1
    counts = torch.bincount(train_labels, minlength=num_classes).float()
    weights = train_labels.numel() / (num_classes * counts.clamp_min(1.0))
    return weights.to(labels.device)


def motif_accuracy(metrics: Any) -> float:
    if metrics.motif_total == 0:
        return 0.0
    return metrics.motif_correct / metrics.motif_total


def metric_value(
    name: str,
    train_metrics: Any,
    val_metrics: Any,
    train_balanced_accuracy: float,
    val_balanced_accuracy: float,
) -> float:
    values = {
        "train_accuracy": train_metrics.accuracy,
        "train_motif_accuracy": motif_accuracy(train_metrics),
        "val_accuracy": val_metrics.accuracy,
        "val_motif_accuracy": motif_accuracy(val_metrics),
        "train_balanced_accuracy": train_balanced_accuracy,
        "val_balanced_accuracy": val_balanced_accuracy,
        "val_loss": -val_metrics.loss,
    }
    if name not in values:
        raise ValueError(f"Unsupported checkpoint monitor metric: {name}")
    return float(values[name])


def balanced_accuracy(logits: torch.Tensor, data: Data, mask: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    labels = data.y
    num_classes = int(labels.max().item()) + 1
    recalls = []
    for class_id in range(num_classes):
        class_mask = mask & (labels == class_id)
        total = int(class_mask.sum().item())
        if total == 0:
            continue
        correct = int((predictions[class_mask] == labels[class_mask]).sum().item())
        recalls.append(correct / total)
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


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
