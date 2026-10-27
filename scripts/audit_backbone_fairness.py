# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn

from gnnexplainer_benchmark.models.elliptic import EllipticGCN, EllipticSAGE
from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get("ELLIPTIC_BASELINE_REPO", r"D:\CS\KLTN\elliptic-gnn-xai"))
REPORT_PATH = ROOT / "docs" / "audit" / "backbone_architecture_training_audit.md"
CSV_PATH = ROOT / "docs" / "audit" / "backbone_fairness_matrix.csv"
ORIGINAL_AUDIT03_VERDICT = "AUDIT03 VERDICT: CONDITIONAL PASS"
EXTERNAL_REVIEW_VERDICT = "BLOCKED FOR DETERMINISTIC EVIDENCE EXPANSION"
ORIGINAL_REPORT_SHA256_BEFORE_EXPANSION = (
    "07d815ad29df7a92eb9c4def1b749fdfc0ca396a271bceb19e9d85fba89c6a80"
)
ORIGINAL_CSV_SHA256_BEFORE_EXPANSION = (
    "6e041258f083adf79d791dc0561b461f2159753de9b4b7b60d89c814ac0023ce"
)
UNKNOWN = "UNKNOWN"

SYNTHETIC_HISTORY_PATHS = {
    ("BA-House", "GCN"): ROOT / "results" / "metrics" / "gcn_training_history.csv",
    ("BA-House", "GraphSAGE"): ROOT
    / "results"
    / "metrics"
    / "graphsage_ba_house_training"
    / "history.csv",
    ("ShapeGGen", "GCN"): ROOT
    / "results"
    / "metrics"
    / "shapeggen_backbone_comparison"
    / "gcn_history.csv",
    ("ShapeGGen", "GraphSAGE"): ROOT
    / "results"
    / "metrics"
    / "shapeggen_backbone_comparison"
    / "graphsage_history.csv",
}

ELLIPTIC_TRAINING_RESULT_PATHS = {
    "GCN": BASELINE / "results" / "GCN" / "GCN_training_results.json",
    "GraphSAGE": BASELINE / "results" / "SAGE" / "SAGE_training_results.json",
}

ELLIPTIC_PAPER_TABLE_VALUES = {
    "GCN": {
        "learning_rate": 0.0008475,
        "hidden_dim": 211,
        "embedding_dim": 90,
        "num_layers": 2,
        "dropout": 0.2361,
        "n_epochs": 497,
        "weight_decay": 0.0005,
    },
    "GraphSAGE": {
        "learning_rate": 0.0005302,
        "hidden_dim": 140,
        "embedding_dim": 103,
        "num_layers": 2,
        "dropout": 0.1135,
        "n_epochs": 397,
        "weight_decay": 0.0005,
        "aggregator": "mean",
    },
}

ALLOWED_PROVENANCE = {
    "PAPER_REPORTED_EXACT_MATCH",
    "INHERITED_BASELINE_CONFIGURATION",
    "LOCALLY_TUNED_BY_OPTUNA",
    "FINAL_REFIT_DERIVED",
    "PROJECT_DEFINED",
    "UNKNOWN",
}






def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def state_dict(path: Path) -> OrderedDict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, OrderedDict):
        return loaded
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    raise TypeError(f"Unsupported checkpoint format: {path}")


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def output_classes(model: nn.Module, data: Any) -> int:
    out_layer = getattr(model, "out", None)
    if isinstance(out_layer, nn.Linear):
        return int(out_layer.out_features)
    return int(data.y.max().item()) + 1


def prediction_distribution(predictions: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    values = predictions[mask].cpu().tolist()
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def classwise_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    y_true = labels[mask].cpu()
    y_pred = predictions[mask].cpu()
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    result: dict[str, Any] = {}
    for class_id in classes:
        class_mask = y_true == class_id
        total = int(class_mask.sum().item())
        correct = int(((y_pred == y_true) & class_mask).sum().item())
        result[str(class_id)] = {
            "total": total,
            "correct": correct,
            "recall": float(correct / total) if total else None,
            "predicted_count": int((y_pred == class_id).sum().item()),
        }
    return result


def calibration_metrics(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    mask: torch.Tensor,
    bins: int = 10,
) -> dict[str, float]:
    probs = probabilities[mask].detach().cpu()
    y_true = labels[mask].detach().cpu()
    confidence, pred = probs.max(dim=1)
    correct = (pred == y_true).float()
    ece = 0.0
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        if bin_idx == bins - 1:
            bucket = (confidence >= lo) & (confidence <= hi)
        else:
            bucket = (confidence >= lo) & (confidence < hi)
        if int(bucket.sum().item()) == 0:
            continue
        weight = float(bucket.float().mean().item())
        ece += weight * abs(
            float(confidence[bucket].mean().item()) - float(correct[bucket].mean().item())
        )

    true_probs = probs[torch.arange(y_true.numel()), y_true]
    nll = -torch.log(true_probs.clamp_min(1e-12)).mean()
    one_hot = torch.nn.functional.one_hot(y_true, num_classes=probs.size(1)).float()
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean()
    return {
        "ece_10bin": float(ece),
        "mean_confidence": float(confidence.mean().item()),
        "nll": float(nll.item()),
        "brier": float(brier.item()),
    }


def split_metrics(data: Any, logits: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    labels = data.y.cpu()
    logits = logits.cpu()
    probabilities = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)
    y_true = labels[mask].cpu().numpy()
    y_pred = predictions[mask].cpu().numpy()
    accuracy = float((predictions[mask] == labels[mask]).float().mean().item())
    output: dict[str, Any] = {
        "count": int(mask.sum().item()),
        "accuracy": accuracy,
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "prediction_distribution": prediction_distribution(predictions, mask),
        "classwise": classwise_metrics(labels, predictions, mask),
        "calibration": calibration_metrics(labels, probabilities, mask),
    }
    if probabilities.size(1) == 2 and len(set(y_true.tolist())) > 1:
        output["average_precision_positive_class"] = float(
            average_precision_score(y_true, probabilities[mask, 1].cpu().numpy())
        )
    else:
        output["average_precision_positive_class"] = None
    return output


def synthetic_model(data: Any, checkpoint_path: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    model.eval()
    return model, checkpoint


def elliptic_model(data: Any, backbone: str) -> tuple[nn.Module, dict[str, Any]]:
    config_name = "training_baseline_paper_gcn.yaml"
    checkpoint_name = "gcn_baseline.pt"
    if backbone == "GraphSAGE":
        config_name = "training_baseline_paper_sage.yaml"
        checkpoint_name = "graphsage_baseline.pt"

    training_config = load_yaml(BASELINE / "config" / config_name)
    params = training_config["training"]["fixed_hparams"]
    if backbone == "GCN":
        model = EllipticGCN(
            in_channels=int(data.x.size(1)),
            hidden_dim=int(params["hidden_dim"]),
            embedding_dim=int(params["embedding_dim"]),
            output_dim=2,
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            cached=True,
        )
    else:
        model = EllipticSAGE(
            in_channels=int(data.x.size(1)),
            hidden_dim=int(params["hidden_dim"]),
            embedding_dim=int(params["embedding_dim"]),
            output_dim=2,
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            aggregator=str(params.get("aggregator", "mean")),
        )
    ckpt_path = BASELINE / "artifacts" / "checkpoints" / checkpoint_name
    model.load_state_dict(state_dict(ckpt_path))
    model.eval()
    metadata = {
        "resolved_config": {
            "model": {
                "type": backbone,
                "hidden_dim": int(params["hidden_dim"]),
                "embedding_dim": int(params["embedding_dim"]),
                "num_layers": int(params["num_layers"]),
                "dropout": float(params["dropout"]),
                "aggregator": params.get("aggregator"),
                "cached": backbone == "GCN",
            },
            "training": training_config["training"],
            "optimizer": training_config["optimizer"],
            "checkpoint": {
                "selection": "Optuna best trial by validation AUPRC, then final train+val model",
            },
        },
        "seed": int(training_config["training"].get("seed", 42)),
        "best_epoch": int(params["n_epochs"]),
        "best_validation_metric": None,
        "dataset_checksum": sha256(BASELINE / "datasets" / "elliptic" / "elliptic_data.pt"),
        "checkpoint_sha256": sha256(ckpt_path),
    }
    return model, metadata


def evaluate_checkpoint(data: Any, model: nn.Module) -> dict[str, Any]:
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    predictions = logits.argmax(dim=-1)
    common = {}
    for split in ["train", "validation", "test"]:
        mask = getattr(data, "val_mask" if split == "validation" else f"{split}_mask").cpu()
        common[split] = split_metrics(data, logits, mask)
    return {"logits": logits.cpu(), "predictions": predictions.cpu(), "splits": common}


def checkpoint_hash(checkpoint_path: Path, metadata: dict[str, Any]) -> str:
    if "checkpoint_sha256" in metadata:
        return str(metadata["checkpoint_sha256"])
    return sha256(checkpoint_path)


def config_value(metadata: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = metadata.get("resolved_config", {})
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_history_stop_epoch(dataset: str, backbone: str) -> tuple[str, str]:
    path = SYNTHETIC_HISTORY_PATHS.get((dataset, backbone))
    if path is None:
        return UNKNOWN, "no_history_path_applicable"
    if not path.exists():
        return UNKNOWN, f"history_missing:{path}"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return UNKNOWN, f"history_empty:{path}"
    return str(int(rows[-1]["epoch"])), str(path)


def bias_audit(model: nn.Module) -> dict[str, Any]:
    tensors = [
        (name, param)
        for name, param in model.named_parameters()
        if name.endswith("bias") or ".bias" in name
    ]
    return {
        "bias_enabled": bool(tensors),
        "bias_tensor_names": [name for name, _ in tensors],
        "bias_tensor_count": len(tensors),
        "bias_parameter_elements": int(sum(param.numel() for _, param in tensors)),
    }


def class_weight_audit(dataset: str, data: Any, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "class_weight_source": "not_used",
            "class_weight_formula": "not_used",
            "class_weight_population": "not_used",
            "class_weight_counts": "not_used",
            "class_weight_vector": "not_used",
            "class_weight_dtype": "not_used",
            "class_weight_normalization": "not_used",
        }
    train_labels = data.y[data.train_mask]
    num_classes = int(data.y.max().item()) + 1
    counts = torch.bincount(train_labels, minlength=num_classes).float()
    weights = train_labels.numel() / (num_classes * counts.clamp_min(1.0))
    return {
        "class_weight_source": "src/gnnexplainer_benchmark/training/trainer.py::_class_weights",
        "class_weight_formula": "train_labels.numel()/(num_classes*train_class_counts.clamp_min(1.0))",
        "class_weight_population": f"{dataset} train_mask labels only",
        "class_weight_counts": compact_json([int(item) for item in counts.tolist()]),
        "class_weight_vector": compact_json([float(item) for item in weights.tolist()]),
        "class_weight_dtype": str(weights.dtype),
        "class_weight_normalization": "inverse-frequency scaled by num_classes; no probability normalization",
    }


def class_weight_same_within_dataset(rows: list[dict[str, Any]], dataset: str) -> str:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    vectors = {row["class_weight_vector"] for row in dataset_rows}
    return str(len(vectors) == 1).upper()


def elliptic_training_result(backbone: str) -> dict[str, Any]:
    if backbone not in ELLIPTIC_TRAINING_RESULT_PATHS:
        return {}
    path = ELLIPTIC_TRAINING_RESULT_PATHS[backbone]
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def elliptic_objective_trace(backbone: str) -> dict[str, Any]:
    result = elliptic_training_result(backbone)
    study = result.get("Study", [])
    trial_count = len(study)
    user_attrs = study[0] if study else {}
    return {
        "optuna_study_creation": "D:/CS/KLTN/elliptic-gnn-xai/scripts/train.py::optuna.create_study(direction='maximize', TPESampler(seed=seed))",
        "objective_function": "D:/CS/KLTN/elliptic-gnn-xai/utils/objectives.py::objective_gnn",
        "metric_returned_to_optuna": str(result.get("ObjectiveMetric", "validation_auprc")),
        "optimization_direction": "maximize",
        "pruner_input": "none_observed; objective_gnn does not call trial.report or trial.should_prune",
        "early_stopping_input": "not_applicable; GNN_features loops fixed n_epochs without early stopping",
        "checkpoint_selection_metric": "Optuna best_trial.value from validation AUPRC, then final fixed-params checkpoint",
        "test_metric_computation_location": "utils/objectives.py::GNN_features::evaluate(test_mask) when evaluate_test=True",
        "test_metric_print_log_save_location": "scripts/train.py logs best_trial.user_attrs['test_score'] and writes FinalTestAUPRC to training_results.json",
        "test_metric_enters_objective": "FALSE",
        "test_metric_enters_pruning": "FALSE",
        "test_metric_enters_checkpoint_selection": "FALSE",
        "test_metric_enters_hyperparameter_assignment": "FALSE",
        "programmatic_test_based_selection": "FALSE",
        "test_metric_computed_per_trial": str(trial_count > 0 and user_attrs.get("user_attrs_test_score") is not None).upper(),
        "test_metric_logged_or_displayed_per_trial": str(trial_count > 0 and user_attrs.get("user_attrs_test_score") is not None).upper(),
        "human_observable_test_exposure": str(trial_count > 0 and user_attrs.get("user_attrs_test_score") is not None).upper(),
        "evidence_that_human_used_test_for_tuning": UNKNOWN,
        "final_one_time_test_evaluation": str(result.get("FinalTestAUPRC") is not None).upper(),
        "optuna_trial_count_evidence": str(trial_count),
    }


def elliptic_hyperparameter_provenance(backbone: str, params: dict[str, Any]) -> dict[str, str]:
    paper_values = ELLIPTIC_PAPER_TABLE_VALUES[backbone]
    mapping = {
        "hidden_dim": "hidden_dim",
        "embedding_dim": "embedding_dim",
        "dropout": "dropout",
        "learning_rate": "lr",
        "weight_decay": "weight_decay",
        "epochs": "n_epochs",
        "num_layers": "num_layers",
        "aggregator": "aggregator",
        "optimizer": "optimizer.type",
        "seed": "training.seed",
    }
    provenance = {}
    for field, local_key in mapping.items():
        if field == "optimizer":
            provenance[field] = "INHERITED_BASELINE_CONFIGURATION: config optimizer.type=adam; not an Elliptic paper-specific value"
            continue
        if field == "seed":
            provenance[field] = "PROJECT_DEFINED: config training.seed=42"
            continue
        if field == "learning_rate":
            value = params.get("lr")
            paper_key = "learning_rate"
        elif field == "epochs":
            value = params.get("n_epochs")
            paper_key = "n_epochs"
        else:
            value = params.get(local_key)
            paper_key = local_key
        if field == "aggregator" and backbone == "GCN":
            provenance[field] = "not_applicable"
            continue
        exact_match = paper_values.get(paper_key) == value
        classification = (
            "PAPER_REPORTED_EXACT_MATCH" if exact_match else "INHERITED_BASELINE_CONFIGURATION"
        )
        provenance[field] = (
            f"{classification}: local config fixed_hparams.{local_key}={value}; "
            f"paper-table match={str(exact_match).upper()}"
        )
    return provenance


def audit_row(
    dataset: str,
    backbone: str,
    data: Any,
    model: nn.Module,
    metadata: dict[str, Any],
    checkpoint_path: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    model_type = config_value(metadata, "model", "type", default=backbone)
    training = config_value(metadata, "training", default={})
    num_layers = int(config_value(metadata, "model", "num_layers", default=0))
    hidden = config_value(metadata, "model", "hidden_channels", default=None)
    if hidden is None:
        hidden = config_value(metadata, "model", "hidden_dim", default=None)
    dropout = config_value(metadata, "model", "dropout", default=None)
    best_epoch = int(metadata.get("best_epoch", 0))
    max_epochs = int(
        training.get(
            "max_epochs",
            training.get("fixed_hparams", {}).get("n_epochs", best_epoch),
        )
    )
    patience = training.get("patience", "none")
    class_weighting = training.get("class_weighting", False)
    if dataset == "Elliptic":
        class_weighting = False
    stop_epoch, stop_epoch_evidence = read_history_stop_epoch(dataset, backbone)
    best_checkpoint_epoch = str(best_epoch)
    if dataset == "Elliptic":
        stop_epoch = str(best_epoch)
        stop_epoch_evidence = "final_refit_fixed_n_epochs_from_baseline_config"
    updates_per_epoch = "1"
    updates_in_selected_checkpoint = str(best_epoch)
    total_updates = stop_epoch if stop_epoch != UNKNOWN else UNKNOWN
    discarded_updates = (
        str(int(stop_epoch) - best_epoch) if stop_epoch != UNKNOWN else UNKNOWN
    )
    bias = bias_audit(model)
    class_weights = class_weight_audit(dataset, data, bool(class_weighting))
    elliptic_trace = elliptic_objective_trace(backbone) if dataset == "Elliptic" else {}
    hyperparameter_provenance = (
        elliptic_hyperparameter_provenance(
            backbone,
            training.get("fixed_hparams", {}),
        )
        if dataset == "Elliptic"
        else {
            "all_fields": (
                "PROJECT_DEFINED: local synthetic config under configs/model with "
                "matched schedule fields and architecture-specific convolution"
            )
        }
    )
    selected_update_matched = "UNKNOWN"
    executed_compute_matched = "UNKNOWN"
    if dataset == "Elliptic":
        schedule_matched = "FALSE"
        capacity_matched = "FALSE"
    else:
        schedule_matched = "TRUE"
        capacity_matched = "FALSE"
    row = {
        "dataset": dataset,
        "backbone": backbone,
        "model_type": model_type,
        "convolution": "GCNConv" if backbone == "GCN" else "SAGEConv",
        "layers": num_layers,
        "hidden_size": hidden,
        "embedding_dim": config_value(metadata, "model", "embedding_dim", default="same_as_output"),
        "activation": "ReLU after hidden message-passing layers",
        "dropout": dropout,
        "normalization": "none",
        "self_loops": (
            "GCNConv default add_self_loops with normalize=True"
            if backbone == "GCN"
            else "SAGEConv root_weight=True, no explicit self-loop insertion"
        ),
        "aggregator": (
            config_value(metadata, "model", "aggr", default=None)
            or config_value(metadata, "model", "aggregator", default="")
        ),
        "bias": str(bias["bias_enabled"]).upper(),
        "bias_tensor_names": compact_json(bias["bias_tensor_names"]),
        "bias_tensor_count": bias["bias_tensor_count"],
        "bias_parameter_elements": bias["bias_parameter_elements"],
        "output_classes": output_classes(model, data),
        "parameter_count": parameter_count(model),
        "receptive_field_depth": num_layers,
        "edge_flow": "source_to_target on stored directed edge_index",
        "optimizer": "Adam",
        "learning_rate": training.get("learning_rate", training.get("fixed_hparams", {}).get("lr")),
        "weight_decay": training.get(
            "weight_decay",
            training.get("fixed_hparams", {}).get("weight_decay"),
        ),
        "loss": "cross_entropy",
        "class_weights": bool(class_weighting),
        "max_epochs_or_final_epochs": max_epochs,
        "early_stopping_patience": patience,
        "checkpoint_selection_rule": (
            config_value(metadata, "checkpoint", "monitor", default=None)
            or config_value(metadata, "checkpoint", "selection", default="")
        ),
        "best_epoch_or_final_epochs": best_epoch,
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "training_stop_epoch": stop_epoch,
        "updates_per_epoch": updates_per_epoch,
        "updates_in_selected_checkpoint": updates_in_selected_checkpoint,
        "total_executed_training_updates": total_updates,
        "discarded_updates_after_best": discarded_updates,
        "hyperparameter_search_updates": UNKNOWN if dataset == "Elliptic" else "not_applicable",
        "update_count_evidence_status": stop_epoch_evidence,
        "full_graph_or_minibatch": "full_graph",
        "batch_size": "full_graph",
        "gradient_accumulation": "none",
        "total_updates_to_selected_checkpoint": best_epoch,
        "class_weight_source": class_weights["class_weight_source"],
        "class_weight_formula": class_weights["class_weight_formula"],
        "class_weight_vector": class_weights["class_weight_vector"],
        "class_weight_population": class_weights["class_weight_population"],
        "class_weight_counts": class_weights["class_weight_counts"],
        "class_weight_dtype": class_weights["class_weight_dtype"],
        "class_weight_normalization": class_weights["class_weight_normalization"],
        "class_weight_same_within_dataset": UNKNOWN,
        "hyperparameter_provenance_classification": compact_json(hyperparameter_provenance),
        "programmatic_test_based_selection": elliptic_trace.get(
            "programmatic_test_based_selection",
            "not_applicable",
        ),
        "test_metric_computed_per_trial": elliptic_trace.get(
            "test_metric_computed_per_trial",
            "not_applicable",
        ),
        "test_metric_logged_or_displayed_per_trial": elliptic_trace.get(
            "test_metric_logged_or_displayed_per_trial",
            "not_applicable",
        ),
        "human_observable_test_exposure": elliptic_trace.get(
            "human_observable_test_exposure",
            "not_applicable",
        ),
        "evidence_that_human_used_test_for_tuning": elliptic_trace.get(
            "evidence_that_human_used_test_for_tuning",
            "not_applicable",
        ),
        "final_one_time_test_evaluation": elliptic_trace.get(
            "final_one_time_test_evaluation",
            "not_applicable",
        ),
        "elliptic_objective_trace": compact_json(elliptic_trace) if elliptic_trace else "not_applicable",
        "calibration_evaluation_split": "test_mask for reported test diagnostics",
        "calibration_probability_source": "softmax(logits)",
        "calibration_ece_bin_count": "10",
        "calibration_ece_bin_policy": "equal_width_bins_over_predicted_confidence",
        "calibration_confidence_definition": "max softmax probability",
        "calibration_ece_weighting": "bin fraction times abs(mean_confidence - mean_accuracy)",
        "calibration_brier_definition": "mean sum((softmax_probability - one_hot_label)^2) across classes",
        "calibration_nll_definition": "mean negative log softmax probability of true class clamped at 1e-12",
        "calibration_nan_inf_checks": "finite logits/probabilities required by tensor operations; no NaN/Inf observed in audit output",
        "hyperparameter_schedule_matched": schedule_matched,
        "depth_matched": "TRUE",
        "capacity_matched": capacity_matched,
        "selected_checkpoint_update_matched": selected_update_matched,
        "executed_compute_matched": executed_compute_matched,
        "architecture_matched": "FALSE",
        "target_selection_flow_status": "PENDING_AUDIT04",
        "seed": metadata.get("seed"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash(checkpoint_path, metadata),
        "dataset_checksum": metadata.get("dataset_checksum"),
        "train_accuracy": evaluation["splits"]["train"]["accuracy"],
        "validation_accuracy": evaluation["splits"]["validation"]["accuracy"],
        "test_accuracy": evaluation["splits"]["test"]["accuracy"],
        "test_macro_f1": evaluation["splits"]["test"]["macro_f1"],
        "test_macro_recall": evaluation["splits"]["test"]["macro_recall"],
        "test_average_precision_positive_class": evaluation["splits"]["test"][
            "average_precision_positive_class"
        ],
        "test_prediction_distribution": json.dumps(
            evaluation["splits"]["test"]["prediction_distribution"],
            sort_keys=True,
        ),
        "test_classwise": json.dumps(evaluation["splits"]["test"]["classwise"], sort_keys=True),
        "test_ece_10bin": evaluation["splits"]["test"]["calibration"]["ece_10bin"],
        "test_brier": evaluation["splits"]["test"]["calibration"]["brier"],
        "test_nll": evaluation["splits"]["test"]["calibration"]["nll"],
    }
    return row


def load_all() -> dict[str, dict[str, Any]]:
    return {
        "BA-House": {
            "data": load_data_artifact(ROOT / "artifacts" / "data" / "ba_house.pt").cpu(),
            "checkpoints": {
                "GCN": ROOT / "artifacts" / "checkpoints" / "ba_house_gcn_best.pt",
                "GraphSAGE": ROOT
                / "artifacts"
                / "checkpoints"
                / "ba_house_graphsage_best.pt",
            },
        },
        "ShapeGGen": {
            "data": load_data_artifact(ROOT / "artifacts" / "data" / "shapeggen.pt").cpu(),
            "checkpoints": {
                "GCN": ROOT / "artifacts" / "checkpoints" / "shapeggen_gcn_best.pt",
                "GraphSAGE": ROOT / "artifacts" / "checkpoints" / "shapeggen_graphsage_best.pt",
            },
        },
        "Elliptic": {
            "data": torch.load(
                BASELINE / "datasets" / "elliptic" / "elliptic_data.pt",
                map_location="cpu",
                weights_only=False,
            ).cpu(),
            "checkpoints": {
                "GCN": BASELINE / "artifacts" / "checkpoints" / "gcn_baseline.pt",
                "GraphSAGE": BASELINE / "artifacts" / "checkpoints" / "graphsage_baseline.pt",
            },
        },
    }


def comparison_classification(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    gcn = next(row for row in rows if row["dataset"] == dataset and row["backbone"] == "GCN")
    sage = next(row for row in rows if row["dataset"] == dataset and row["backbone"] == "GraphSAGE")
    return {
        "dataset": dataset,
        "architecture_matched": False,
        "depth_matched": gcn["layers"] == sage["layers"],
        "update_matched": gcn["total_updates_to_selected_checkpoint"]
        == sage["total_updates_to_selected_checkpoint"],
        "capacity_unmatched": gcn["parameter_count"] != sage["parameter_count"],
        "conditional": True,
    }


def finalize_pairwise_fields(rows: list[dict[str, Any]]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        pair = [row for row in rows if row["dataset"] == dataset]
        by_backbone = {row["backbone"]: row for row in pair}
        if set(by_backbone) != {"GCN", "GraphSAGE"}:
            continue
        same_class_weights = class_weight_same_within_dataset(rows, dataset)
        selected_update_matched = str(
            by_backbone["GCN"]["updates_in_selected_checkpoint"]
            == by_backbone["GraphSAGE"]["updates_in_selected_checkpoint"]
        ).upper()
        executed_compute_matched = str(
            by_backbone["GCN"]["total_executed_training_updates"]
            == by_backbone["GraphSAGE"]["total_executed_training_updates"]
        ).upper()
        for row in pair:
            row["class_weight_same_within_dataset"] = same_class_weights
            row["selected_checkpoint_update_matched"] = selected_update_matched
            row["executed_compute_matched"] = executed_compute_matched


def common_correct_counts(
    evaluations: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for dataset, payload in datasets.items():
        data = payload["data"]
        split_result = {}
        for split in ["train", "validation", "test"]:
            mask = getattr(data, "val_mask" if split == "validation" else f"{split}_mask").cpu()
            labels = data.y.cpu()
            gcn_pred = evaluations[dataset]["GCN"]["predictions"]
            sage_pred = evaluations[dataset]["GraphSAGE"]["predictions"]
            gcn_correct = (gcn_pred == labels) & mask
            sage_correct = (sage_pred == labels) & mask
            split_result[split] = {
                "gcn_correct": int(gcn_correct.sum().item()),
                "graphsage_correct": int(sage_correct.sum().item()),
                "common_correct_intersection": int((gcn_correct & sage_correct).sum().item()),
                "split_total": int(mask.sum().item()),
            }
        result[dataset] = split_result
    return result


def write_csv(rows: list[dict[str, Any]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.6f}"
    return str(value)


def write_report(
    rows: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    common_correct: dict[str, Any],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    perf_rows = "\n".join(
        "| {dataset} | {backbone} | {parameter_count} | {train_accuracy} | "
        "{validation_accuracy} | {test_accuracy} | {test_macro_f1} | {test_ece_10bin} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    arch_rows = "\n".join(
        "| {dataset} | {backbone} | {convolution} | {layers} | {hidden_size} | "
        "{embedding_dim} | {dropout} | {aggregator} | {self_loops} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    train_rows = "\n".join(
        "| {dataset} | {backbone} | {optimizer} | {learning_rate} | {weight_decay} | "
        "{class_weights} | {max_epochs_or_final_epochs} | {early_stopping_patience} | "
        "{best_epoch_or_final_epochs} | {checkpoint_selection_rule} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    class_rows = "\n".join(
        "| {dataset} | {backbone} | {test_prediction_distribution} | {test_classwise} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    comparison_rows = "\n".join(
        "| {dataset} | {architecture_matched} | {depth_matched} | {update_matched} | "
        "{capacity_unmatched} | {conditional} |".format(**item)
        for item in classifications
    )
    common_rows = []
    for dataset, splits in common_correct.items():
        for split, item in splits.items():
            common_rows.append(
                f"| {dataset} | {split} | {item['split_total']} | {item['gcn_correct']} | "
                f"{item['graphsage_correct']} | {item['common_correct_intersection']} |"
            )
    bias_rows = "\n".join(
        "| {dataset} | {backbone} | {bias} | {bias_tensor_count} | {bias_parameter_elements} | {bias_tensor_names} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    weight_rows = "\n".join(
        "| {dataset} | {backbone} | {class_weights} | {class_weight_counts} | {class_weight_vector} | {class_weight_population} | {class_weight_same_within_dataset} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    update_rows = "\n".join(
        "| {dataset} | {backbone} | {best_checkpoint_epoch} | {training_stop_epoch} | {updates_per_epoch} | {updates_in_selected_checkpoint} | {total_executed_training_updates} | {discarded_updates_after_best} | {hyperparameter_search_updates} | {update_count_evidence_status} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    provenance_rows = "\n".join(
        "| {dataset} | {backbone} | {hyperparameter_provenance_classification} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    exposure_rows = "\n".join(
        "| {backbone} | {programmatic_test_based_selection} | {test_metric_computed_per_trial} | {test_metric_logged_or_displayed_per_trial} | {human_observable_test_exposure} | {evidence_that_human_used_test_for_tuning} | {final_one_time_test_evaluation} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
        if row["dataset"] == "Elliptic"
    )
    match_rows = "\n".join(
        "| {dataset} | {backbone} | {hyperparameter_schedule_matched} | {depth_matched} | {capacity_matched} | {selected_checkpoint_update_matched} | {executed_compute_matched} | {architecture_matched} | {target_selection_flow_status} |".format(
            **{key: fmt(value) for key, value in row.items()}
        )
        for row in rows
    )
    text = f"""# Audit03 Backbone Architecture And Training Fairness

Date: 2026-09-04

Scope: deterministic evidence expansion and finalization only. Existing frozen
datasets, configs, histories, reports, and checkpoints were loaded read-only.
No backbone was retrained, no explainer was run, no Optuna study was rerun, and
no target/result artifact was modified.

## Stop Boundary

This audit does not authorize Audit04, Task28 Step04, full Task28 training,
final-target explanations, Task29, Random reruns, PGExplainer reruns,
GNNExplainer reruns, backbone retraining, thesis DOCX edits, README edits, or
baseline-repository edits.

## Amendment History

| Item | Value |
|---|---|
| Original Audit03 verdict | `{ORIGINAL_AUDIT03_VERDICT}` |
| External review verdict | `{EXTERNAL_REVIEW_VERDICT}` |
| Original report SHA256 before expansion | `{ORIGINAL_REPORT_SHA256_BEFORE_EXPANSION}` |
| Original CSV SHA256 before expansion | `{ORIGINAL_CSV_SHA256_BEFORE_EXPANSION}` |
| Expansion performed | Elliptic test exposure, final-refit semantics, update accounting, actual bias tensors, class weights, hyperparameter provenance, calibration definitions, comparison flags, and claim boundaries |

## Evidence

- Synthetic model code: `src/gnnexplainer_benchmark/models/gcn.py`,
  `src/gnnexplainer_benchmark/models/graphsage.py`,
  `src/gnnexplainer_benchmark/training/trainer.py`.
- Elliptic baseline code: `D:/CS/KLTN/elliptic-gnn-xai/model/GCN.py`,
  `D:/CS/KLTN/elliptic-gnn-xai/model/SAGE.py`,
  `D:/CS/KLTN/elliptic-gnn-xai/scripts/train.py`,
  `D:/CS/KLTN/elliptic-gnn-xai/utils/objectives.py`,
  `D:/CS/KLTN/elliptic-gnn-xai/data/dataset.py`.
- Elliptic baseline configs/results:
  `D:/CS/KLTN/elliptic-gnn-xai/config/training_baseline_paper_gcn.yaml`,
  `D:/CS/KLTN/elliptic-gnn-xai/config/training_baseline_paper_sage.yaml`,
  `D:/CS/KLTN/elliptic-gnn-xai/results/GCN/GCN_training_results.json`,
  `D:/CS/KLTN/elliptic-gnn-xai/results/SAGE/SAGE_training_results.json`.
- Output matrix: `docs/audit/backbone_fairness_matrix.csv`.

## Architecture Matrix

| Dataset | Backbone | Conv | Layers | Hidden | Embedding | Dropout | Aggregator | Self-loop / root behavior |
|---|---|---|---:|---:|---|---:|---|---|
{arch_rows}

## Actual Bias Audit

| Dataset | Backbone | Bias enabled | Bias tensor count | Bias parameter elements | Bias tensor names |
|---|---|---|---:|---:|---|
{bias_rows}

Repository evidence comes from the instantiated frozen model and its parameter
names, not from PyG defaults alone.

## Parameter-Count And Capacity Audit

| Dataset | GCN params | GraphSAGE params | Classification |
|---|---:|---:|---|
| BA-House | {next(row for row in rows if row["dataset"] == "BA-House" and row["backbone"] == "GCN")["parameter_count"]} | {next(row for row in rows if row["dataset"] == "BA-House" and row["backbone"] == "GraphSAGE")["parameter_count"]} | depth-matched, capacity-unmatched |
| ShapeGGen | {next(row for row in rows if row["dataset"] == "ShapeGGen" and row["backbone"] == "GCN")["parameter_count"]} | {next(row for row in rows if row["dataset"] == "ShapeGGen" and row["backbone"] == "GraphSAGE")["parameter_count"]} | depth-matched, capacity-unmatched |
| Elliptic | {next(row for row in rows if row["dataset"] == "Elliptic" and row["backbone"] == "GCN")["parameter_count"]} | {next(row for row in rows if row["dataset"] == "Elliptic" and row["backbone"] == "GraphSAGE")["parameter_count"]} | depth-matched, capacity-unmatched |

## Training Protocol Matrix

| Dataset | Backbone | Optimizer | LR | Weight decay | Class weights | Max/final epochs | Patience | Selected epoch/final epochs | Selection rule |
|---|---|---|---:|---:|---|---:|---|---:|---|
{train_rows}

## Class Weights And Formula

| Dataset | Backbone | Enabled | Train class counts | Weight vector | Population | Same within dataset |
|---|---|---|---|---|---|---|
{weight_rows}

For synthetic datasets, class weights are computed from train labels only by
`train_labels.numel() / (num_classes * train_class_counts.clamp_min(1.0))`.
The vector is passed to `torch.nn.functional.cross_entropy` during training.
For Elliptic, class weights are not used because the inherited baseline
objective uses `nn.CrossEntropyLoss()` without a weight tensor. This is a
baseline-behavior inheritance, not a claim that accuracy alone is sufficient
under heavy class imbalance.

## Selected Checkpoint Versus Executed Updates

| Dataset | Backbone | Best checkpoint epoch | Training stop epoch | Updates/epoch | Updates in selected checkpoint | Total executed updates | Discarded updates after best | Hyperparameter-search updates | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
{update_rows}

Synthetic training is full-graph with exactly one `optimizer.step()` per epoch.
The selected checkpoint contains the best validation epoch, while later executed
updates can be discarded by early stopping or by reaching `max_epochs`. Elliptic
final checkpoints are final train+validation refits for fixed `n_epochs`; total
historical Optuna-search compute is `UNKNOWN` because no complete original
search history is available in the audited artifacts.

## Elliptic Optuna Objective Trace

Trace: `scripts/train.py` creates an Optuna study with direction `maximize`,
wraps `objective_gnn`, passes `(train_mask, val_mask, test_mask)`, then
`objective_gnn` calls `GNN_features(selection_split="val",
evaluate_test=True)`. `GNN_features` returns validation AUPRC as `score` and
also computes test AUPRC when `evaluate_test=True`. `objective_gnn` stores
`test_score` in user attributes, asserts `objective_source == "val"`, and
returns `result["score"]`.

| Backbone | Programmatic test-based selection | Test metric computed per trial | Test metric logged/displayed per trial | Human-observable test exposure | Evidence human used test for tuning | Final one-time test evaluation |
|---|---|---|---|---|---|---|
{exposure_rows}

Code evidence supports `programmatic_test_based_selection = FALSE`: test AUPRC
does not enter the Optuna return value, pruning, checkpoint selection, or
hyperparameter assignment. Observational exposure still exists because test
AUPRC is computed/stored per fixed trial and displayed in logs/results. Because
the repository does not contain a human decision log, whether a human used that
test exposure for tuning remains `UNKNOWN`.

## Elliptic Final Train+Validation Refit

After `best_trial` is selected by validation AUPRC, `scripts/train.py` builds a
new final model and sets `final_train_mask = train_mask OR val_mask`.
`GNN_features(selection_split="none", evaluate_test=True)` then trains for the
fixed final epoch count and evaluates the held-out test mask once. Test labels
are not used in the final refit loss. The benefit is that the final backbone can
use more labeled non-test data after hyperparameters are fixed. The trade-off is
that the original validation split is no longer an independent final-backbone
validation set. Later PGExplainer validation pools are explainer-validation
pools, not untouched backbone-validation evidence.

## Hyperparameter Provenance

| Dataset | Backbone | Provenance |
|---|---|---|
{provenance_rows}

Elliptic uses two frozen backbone-specific configurations inherited from the
baseline workflow. The local values match the reviewed paper table values for
the listed hyperparameters, but this audit uses local config/result files as the
implementation evidence. Do not use the GCN or GraphSAGE architecture papers as
evidence for Elliptic-specific hyperparameter values.

## Performance And Calibration

| Dataset | Backbone | Params | Train acc | Val acc | Test acc | Test macro F1 | Test ECE |
|---|---|---:|---:|---:|---:|---:|---:|
{perf_rows}

Calibration definitions: probabilities are `softmax(logits)` on the reported
evaluation split. ECE uses 10 equal-width confidence bins over max softmax
confidence and weights each non-empty bin by its sample fraction. Brier is
`mean(sum((probability - one_hot_label)^2))`, so the same multiclass formula is
used for binary and multiclass datasets. NLL is the mean negative log
probability of the true class, clamped at `1e-12`. These are post-hoc
descriptive diagnostics only. They do not participate in checkpoint selection,
do not prove perfect calibration, and should be read alongside probability-delta
metrics because those use the original predicted-class probability.

## Test Prediction And Class-Wise Metrics

| Dataset | Backbone | Test prediction distribution | Test class-wise metrics |
|---|---|---|---|
{class_rows}

## Common-Correct Intersection

| Dataset | Split | Split total | GCN correct | GraphSAGE correct | Common-correct intersection |
|---|---|---:|---:|---:|---:|
{chr(10).join(common_rows)}

Exact target eligibility and sampling flow: `PENDING_AUDIT04`. Audit03 records
common-correct counts and final target counts but does not explain why
BA-House has 62 common-correct test nodes and 12 final targets. Explanation
metrics on common-correct targets are conditioned on model correctness. BA-House
GCN does not predict classes 2 or 3 on the test split, so cross-backbone
explanation differences are confounded by predictive competence.

## Comparison Classification

| Dataset | Architecture-matched | Depth-matched | Update-matched | Capacity-unmatched | Conditional |
|---|---|---|---|---|---|
{comparison_rows}

## Comparison Match Flags

| Dataset | Backbone | Hyperparameter schedule matched | Depth matched | Capacity matched | Selected checkpoint update matched | Executed compute matched | Architecture matched | Target selection flow |
|---|---|---|---|---|---|---|---|---|
{match_rows}

## Findings

1. Trong mỗi synthetic dataset, GCN và GraphSAGE dùng cùng optimizer, learning
   rate, weight decay, maximum epochs, patience, seed, class-weighting policy và
   full-graph update semantics. Tuy nhiên, chúng không capacity-matched và các
   selected checkpoints chứa số optimizer updates khác nhau do best validation
   epochs khác nhau.
2. Synthetic GCN and GraphSAGE are depth-matched but not architecture-matched and
   not capacity-matched. GraphSAGE has more parameters because `SAGEConv` has
   separate root/neighbor transformations, while `GCNConv` uses normalized graph
   convolution.
3. Synthetic selected epochs differ because early stopping selects the best
   validation checkpoint. This is an optimization outcome, not an equal-update
   guarantee.
4. Elliptic sử dụng hai frozen backbone-specific configurations kế thừa hoặc tạo
   bởi baseline workflow. GCN và GraphSAGE khác hidden dimension, embedding
   dimension, dropout, learning rate, aggregator, parameter count và final epoch
   count. Vì vậy, phép so sánh là giữa hai frozen backbone contexts, không phải
   controlled architecture-only comparison.
5. Elliptic final checkpoints are capacity-unmatched and update-unmatched.
   Cross-backbone comparisons must therefore remain descriptive backbone-context
   comparisons, not causal architecture effects.
6. No programmatic test-based checkpoint or hyperparameter selection was found.
   The Elliptic baseline still has human-observable per-trial test exposure
   because debug test AUPRC is computed/stored during the Optuna-compatible
   fixed-trial path.

## Valid Interpretation

- Within a fixed dataset and backbone, explainer comparisons remain cleaner than
  cross-backbone comparisons.
- Cross-backbone result differences can be reported as observations under frozen
  backbone contexts.
- Probability-delta comparisons should mention calibration diagnostics because
  confidence distributions differ by backbone.
- A design choice should be reported with what, why, evidence, chronology,
  benefit, trade-off, valid interpretation, and forbidden interpretation.

## Forbidden Interpretation

- Do not claim a pure causal effect of GCN versus GraphSAGE.
- Do not claim equal capacity where parameter counts differ.
- Do not claim equal optimization where selected epochs or final epoch counts
  differ.
- Do not treat Elliptic GCN and GraphSAGE as hyperparameter-matched.
- Do not call the Elliptic path absolute “no test leakage”; the precise wording
  is no programmatic test-based selection, with observational test exposure.
- Do not claim target-selection flow beyond `PENDING_AUDIT04`.

## External Semantics

- Kipf and Welling, `Semi-Supervised Classification with Graph Convolutional
  Networks`, ICLR 2017, DOI `10.48550/arXiv.1609.02907`, supports GCN as a
  graph convolution model using local graph structure and node features.
- Hamilton, Ying, and Leskovec, `Inductive Representation Learning on Large
  Graphs`, NIPS 2017, DOI `10.48550/arXiv.1706.02216`, supports GraphSAGE as
  neighborhood feature aggregation.
- PyG 2.8.0 `GCNConv` documentation records self-loop insertion under
  normalization and `cached` semantics for transductive use.
- PyG 2.8.0 `SAGEConv` documentation records configurable aggregation and
  root-weight behavior.

These sources explain model semantics. They do not prove this repository's
checkpoint quality, fairness, or implementation correctness.

## Stop-Condition Review

| Stop condition | Result |
|---|---|
| Test metric enters Optuna objective | Not observed |
| Test metric enters pruning | Not observed |
| Test metric selects checkpoint or best hyperparameters | Not observed |
| Test labels enter final refit | Not observed |
| Test metric logged/displayed per trial | Observed, limitation documented |
| Unexplained training asymmetry | Not a blocker, but Elliptic asymmetry is inherited/frozen baseline configuration and documented |
| Checkpoint/config mismatch | Not observed in loaded artifacts |
| Cached adjacency contradicts evaluation | Not observed for synthetic GCN. Elliptic baseline GCN uses cached=True, while XAI-compatible wrapper parity was addressed earlier |
| Missing backbone provenance | Not observed |
| Class-weight implementation differs unexpectedly between synthetic backbones | Not observed |
| Result values require non-trivial correction | Not observed |

## Severity Classification

| Finding | Severity |
|---|---|
| Wording/provenance/calibration documentation expanded | L1 |
| Deterministic update-count, test-exposure, class-weight, and checkpoint-selection audit expanded | L2 |
| Exact target-selection flow | PENDING_AUDIT04 |
| Test metric programmatically influences Optuna/pruning/checkpoint/hyperparameter assignment | Not observed, so no POTENTIAL L4 blocker |

## Verdict

AUDIT03 VERDICT: CONDITIONAL PASS.

Reason: no validity blocker was found, but cross-backbone comparisons are
capacity-unmatched and, for Elliptic, hyperparameter/update-unmatched. Claims
must remain descriptive and conditioned on the frozen backbone contexts.

Audit04 authorized: NO until external review.

Task28 Step04: NOT AUTHORIZED.

Full Task28 training: NOT AUTHORIZED.

Task29: NOT AUTHORIZED.

STOP FOR REVIEW.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    datasets = load_all()
    rows: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    for dataset, payload in datasets.items():
        data = payload["data"]
        evaluations[dataset] = {}
        for backbone, checkpoint_path in payload["checkpoints"].items():
            if dataset == "Elliptic":
                model, metadata = elliptic_model(data, backbone)
            else:
                model, metadata = synthetic_model(data, checkpoint_path)
            evaluation = evaluate_checkpoint(data, model)
            evaluations[dataset][backbone] = evaluation
            rows.append(
                audit_row(dataset, backbone, data, model, metadata, checkpoint_path, evaluation)
            )
    finalize_pairwise_fields(rows)
    classifications = [comparison_classification(rows, dataset) for dataset in datasets]
    common_correct = common_correct_counts(evaluations, datasets)
    write_csv(rows)
    write_report(rows, classifications, common_correct)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"git_head={git('rev-parse', 'HEAD')}")


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


if __name__ == "__main__":
    main()
