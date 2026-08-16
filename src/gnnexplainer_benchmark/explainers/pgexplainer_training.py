from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
import torch_geometric
import yaml
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import PGExplainer

from gnnexplainer_benchmark.data.pgexplainer_targets import (
    PGExplainerTarget,
    validate_pgexplainer_target_splits,
)
from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.evaluation.budget import selected_edge_count_for_ratio
from gnnexplainer_benchmark.evaluation.edge_ground_truth import (
    candidate_edge_scores_and_labels,
    compute_edge_ground_truth_metrics,
)
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.explainers.pgexplainer_smoke import model_parameter_hash
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)
from gnnexplainer_benchmark.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class PGExplainerTrainingResult:
    artifact_path: Path
    manifest_path: Path
    training_log_path: Path
    validation_health_path: Path
    validation_summary_path: Path
    epochs_completed: int
    final_loss: float
    training_runtime_seconds: float
    model_parameters_unchanged: bool
    checkpoint_reload_ok: bool
    validation_success_count: int
    validation_failure_count: int


def read_pgexplainer_targets(path: Path) -> list[PGExplainerTarget]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    targets = [
        PGExplainerTarget(
            node_id=int(row["node_id"]),
            ground_truth_label=int(row["ground_truth_label"]),
            frozen_model_prediction=int(row["frozen_model_prediction"]),
            target_for_pgexplainer=int(row["target_for_pgexplainer"]),
            confidence=float(row["confidence"]),
            base_model_split=row["base_model_split"],
            is_ground_truth_motif=str(row["is_ground_truth_motif"]).lower() == "true",
            gcn_prediction_correct=str(row["gcn_prediction_correct"]).lower() == "true",
            candidate_edge_count=int(row["candidate_edge_count"]),
            candidate_edge_mapping_valid=(
                str(row["candidate_edge_mapping_valid"]).lower() == "true"
            ),
        )
        for row in rows
    ]
    node_ids = [target.node_id for target in targets]
    if node_ids != sorted(node_ids):
        raise ValueError(f"{path} must be sorted by node_id")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"{path} contains duplicate node ids")
    return targets


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_pgexplainer_from_config(config: dict[str, Any]) -> PGExplainer:
    explainer_config = config["explainer"]
    regularization = explainer_config["regularization"]
    temperature = explainer_config["temperature"]
    kwargs = {
        "edge_size": float(regularization["edge_size"]),
        "edge_ent": float(regularization["edge_ent"]),
        "temp": [float(temperature["start"]), float(temperature["end"])],
    }
    for name, coefficient in explainer_config["implementation_coefficients"].items():
        kwargs[name] = float(coefficient["value"])
    return PGExplainer(
        epochs=int(explainer_config["epochs"]),
        lr=float(explainer_config["learning_rate"]),
        **kwargs,
    )


def wrap_pgexplainer(model: torch.nn.Module, algorithm: PGExplainer) -> Explainer:
    return Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="phenomenon",
        edge_mask_type="object",
        node_mask_type=None,
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


def materialize_pgexplainer(
    algorithm: PGExplainer,
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_id: int,
) -> None:
    with torch.no_grad():
        node_embeddings = algorithm._get_embeddings(model, x, edge_index)
        inputs = algorithm._get_inputs(node_embeddings, edge_index, node_id)
        algorithm.mlp(inputs[:1])


def train_pgexplainer_gcn(
    *,
    dataset_path: Path = Path("artifacts/data/ba_house.pt"),
    dataset_config_path: Path = Path("configs/dataset/ba_house.yaml"),
    checkpoint_path: Path = Path("artifacts/checkpoints/ba_house_gcn_best.pt"),
    explainer_config_path: Path = Path("configs/explainer/pgexplainer.yaml"),
    evaluation_config_path: Path = Path("configs/evaluation/pgexplainer_gcn.yaml"),
    train_targets_path: Path = Path("configs/targets/pgexplainer_train_targets.csv"),
    validation_targets_path: Path = Path("configs/targets/pgexplainer_val_targets.csv"),
    final_test_targets_path: Path = Path("results/metrics/explainer_test_targets.csv"),
    artifact_path: Path = Path("artifacts/explainers/pgexplainer_gcn_ba_house_seed42.pt"),
    manifest_path: Path = Path(
        "artifacts/explainers/pgexplainer_gcn_ba_house_seed42_manifest.json"
    ),
    training_log_path: Path = Path("results/metrics/pgexplainer_gcn_seed42_training_log.csv"),
    validation_health_path: Path = Path(
        "results/metrics/pgexplainer_gcn_seed42_validation_health.csv"
    ),
    validation_summary_path: Path = Path(
        "results/metrics/pgexplainer_gcn_seed42_validation_summary.json"
    ),
    device: str = "cpu",
    seed_override: int | None = None,
) -> PGExplainerTrainingResult:
    explainer_config = load_yaml(explainer_config_path)
    evaluation_config = load_yaml(evaluation_config_path)
    if seed_override is not None:
        explainer_config["explainer"]["seed"] = seed_override
    seed = int(explainer_config["explainer"]["seed"])
    epochs = int(explainer_config["explainer"]["epochs"])
    seed_everything(seed)
    dataset_config = load_yaml(dataset_config_path)

    train_targets = read_pgexplainer_targets(train_targets_path)
    validation_targets = read_pgexplainer_targets(validation_targets_path)
    final_test_targets = read_final_test_node_ids(final_test_targets_path)
    validate_pgexplainer_target_splits(train_targets, validation_targets, final_test_targets)

    torch_device = torch.device(device)
    data_cpu = load_data_artifact(dataset_path)
    data = data_cpu.clone().to(torch_device)
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        frozen_predictions = logits.argmax(dim=-1).detach()
    validate_targets_match_frozen_predictions(train_targets, frozen_predictions)
    validate_targets_match_frozen_predictions(validation_targets, frozen_predictions)

    gcn_hash_before = model_parameter_hash(model)
    algorithm = build_pgexplainer_from_config(explainer_config).to(torch_device)
    explainer = wrap_pgexplainer(model, algorithm)
    materialize_pgexplainer(algorithm, model, data.x, data.edge_index, train_targets[0].node_id)
    optimizer_excludes_gcn = optimizer_excludes_gcn_parameters(algorithm, model)
    if not optimizer_excludes_gcn:
        raise RuntimeError("PGExplainer optimizer unexpectedly includes frozen model parameters")

    epoch_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        losses = []
        for target in train_targets:
            loss = algorithm.train(
                epoch,
                model,
                data.x,
                data.edge_index,
                target=frozen_predictions,
                index=target.node_id,
            )
            if not np.isfinite(loss):
                raise RuntimeError(f"Non-finite PGExplainer loss at epoch {epoch}: {loss}")
            losses.append(float(loss))
        epoch_rows.append(
            {
                "epoch": epoch,
                "mean_loss": mean(losses),
                "median_loss": median(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "num_train_instances": len(losses),
                "learning_rate": float(explainer_config["explainer"]["learning_rate"]),
                "temperature": algorithm._get_temperature(epoch),
                "runtime_seconds": time.perf_counter() - epoch_start,
                "seed": seed,
            }
        )
    training_runtime = time.perf_counter() - start
    gcn_hash_after = model_parameter_hash(model)
    model_parameters_unchanged = gcn_hash_before == gcn_hash_after
    if not model_parameters_unchanged:
        raise RuntimeError("Frozen model parameters changed during PGExplainer training")

    save_pgexplainer_artifact(
        artifact_path=artifact_path,
        algorithm=algorithm,
        explainer_config=explainer_config,
        evaluation_config=evaluation_config,
        train_targets=train_targets,
        validation_targets=validation_targets,
        seed=seed,
    )
    checkpoint_reload_ok = verify_pgexplainer_reload(
        artifact_path,
        explainer_config,
        model,
        data.x,
        data.edge_index,
        train_targets[0].node_id,
    )
    pgexplainer_checkpoint_sha256 = sha256_file(artifact_path)
    write_epoch_log(training_log_path, epoch_rows)
    validation_rows = run_validation_health_check(
        explainer=explainer,
        model=model,
        data_cpu=data_cpu,
        data=data,
        validation_targets=validation_targets,
        frozen_predictions=frozen_predictions,
        num_hops=int(checkpoint["resolved_config"]["model"]["num_layers"]),
        selected_edge_ratio=float(evaluation_config["evaluation"]["selected_edge_ratio"]),
    )
    write_validation_health(validation_health_path, validation_rows)
    validation_summary = summarize_validation_health(validation_rows)
    validation_summary_path.parent.mkdir(parents=True, exist_ok=True)
    validation_summary_path.write_text(
        json.dumps(validation_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "artifact_path": str(artifact_path),
        "pgexplainer_checkpoint_sha256": pgexplainer_checkpoint_sha256,
        "run_status": "success",
        "dataset_path": str(dataset_path),
        "dataset_config_path": str(dataset_config_path),
        "dataset_seed": int(dataset_config["dataset"]["seed"]),
        "dataset_sha256": sha256_file(dataset_path),
        "model_type": str(checkpoint["resolved_config"]["model"].get("type", "GCN")),
        "model_checkpoint_path": str(checkpoint_path),
        "model_checkpoint_sha256": sha256_file(checkpoint_path),
        "gcn_checkpoint_path": str(checkpoint_path),
        "gcn_checkpoint_sha256": sha256_file(checkpoint_path),
        "gcn_model_hash_before": gcn_hash_before,
        "gcn_model_hash_after": gcn_hash_after,
        "gcn_model_hash_before_training": gcn_hash_before,
        "gcn_model_hash_after_training": gcn_hash_after,
        "gcn_model_parameters_unchanged": model_parameters_unchanged,
        "model_weights_unchanged": model_parameters_unchanged,
        "pgexplainer_config_path": str(explainer_config_path),
        "pgexplainer_config_hash": config_hash(explainer_config),
        "pgexplainer_hyperparameters": pgexplainer_hyperparameters(explainer_config),
        "provenance_classification": {
            "epochs": "PAPER / SUPPLEMENT",
            "learning_rate": "PAPER / SUPPLEMENT",
            "edge_size": "PAPER / SUPPLEMENT",
            "edge_ent": "PAPER / SUPPLEMENT",
            "temperature": "PAPER / SUPPLEMENT",
            "bias": "PYG_IMPLEMENTATION",
            "selected_edge_ratio": "THESIS_PROTOCOL",
            "target_semantics": "THESIS_PROTOCOL",
        },
        "evaluation_config_path": str(evaluation_config_path),
        "evaluation_config_hash": config_hash(evaluation_config),
        "train_target_file": str(train_targets_path),
        "train_target_file_hash": sha256_file(train_targets_path),
        "train_target_path": str(train_targets_path),
        "train_target_sha256": sha256_file(train_targets_path),
        "train_target_count": len(train_targets),
        "validation_target_file": str(validation_targets_path),
        "validation_target_file_hash": sha256_file(validation_targets_path),
        "validation_target_path": str(validation_targets_path),
        "validation_target_sha256": sha256_file(validation_targets_path),
        "validation_target_count": len(validation_targets),
        "final_test_target_file": str(final_test_targets_path),
        "final_test_target_path": str(final_test_targets_path),
        "final_test_target_sha256": sha256_file(final_test_targets_path),
        "final_test_target_count": len(final_test_targets),
        "final_test_node_count_used_for_training": 0,
        "final_test_targets_used_for_training": 0,
        "environment": environment_metadata(),
        "environment_audit_reference": "docs/pgexplainer_installed_api_audit.json",
        "torch_geometric_version": torch_geometric.__version__,
        "seed": seed,
        "epochs": epochs,
        "epochs_completed": epochs,
        "final_loss": epoch_rows[-1]["mean_loss"],
        "final_mean_loss": epoch_rows[-1]["mean_loss"],
        "training_runtime_seconds": training_runtime,
        "explainer_training_runtime_seconds": training_runtime,
        "inference_runtime_seconds": None,
        "evaluation_runtime_seconds": None,
        "optimizer_excludes_gcn_parameters": optimizer_excludes_gcn,
        "checkpoint_reload_ok": checkpoint_reload_ok,
        "validation_health": validation_summary,
        "training_code_base_commit": current_git_commit(),
        "training_code_worktree_state": (
            "Task-15 implementation was uncommitted at successful training time; "
            "this commit records the base HEAD before Task-15 commit."
        ),
        "git_commit": current_git_commit(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PGExplainerTrainingResult(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        training_log_path=training_log_path,
        validation_health_path=validation_health_path,
        validation_summary_path=validation_summary_path,
        epochs_completed=epochs,
        final_loss=float(epoch_rows[-1]["mean_loss"]),
        training_runtime_seconds=training_runtime,
        model_parameters_unchanged=model_parameters_unchanged,
        checkpoint_reload_ok=checkpoint_reload_ok,
        validation_success_count=sum(row["status"] == "success" for row in validation_rows),
        validation_failure_count=sum(row["status"] != "success" for row in validation_rows),
    )


def read_final_test_node_ids(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["node_id"]) for row in csv.DictReader(handle)}


def validate_targets_match_frozen_predictions(
    targets: list[PGExplainerTarget],
    frozen_predictions: torch.Tensor,
) -> None:
    for target in targets:
        actual = int(frozen_predictions[target.node_id].item())
        if target.target_for_pgexplainer != actual:
            raise ValueError(
                f"Target {target.node_id} uses {target.target_for_pgexplainer}, "
                f"but frozen GCN predicts {actual}"
            )


def optimizer_excludes_gcn_parameters(
    algorithm: PGExplainer,
    model: torch.nn.Module,
) -> bool:
    gcn_parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    return not (optimizer_parameter_ids & gcn_parameter_ids)


def save_pgexplainer_artifact(
    *,
    artifact_path: Path,
    algorithm: PGExplainer,
    explainer_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    train_targets: list[PGExplainerTarget],
    validation_targets: list[PGExplainerTarget],
    seed: int,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm_state_dict": algorithm.state_dict(),
            "algorithm_current_epoch": algorithm._curr_epoch,
            "algorithm_class": "torch_geometric.explain.algorithm.PGExplainer",
            "explainer_config": explainer_config,
            "evaluation_config": evaluation_config,
            "seed": seed,
            "train_node_ids": [target.node_id for target in train_targets],
            "validation_node_ids": [target.node_id for target in validation_targets],
        },
        artifact_path,
    )


def pgexplainer_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    explainer = config["explainer"]
    return {
        "epochs": int(explainer["epochs"]),
        "learning_rate": float(explainer["learning_rate"]),
        "edge_size": float(explainer["regularization"]["edge_size"]),
        "edge_ent": float(explainer["regularization"]["edge_ent"]),
        "temperature_start": float(explainer["temperature"]["start"]),
        "temperature_end": float(explainer["temperature"]["end"]),
        "bias": float(explainer["implementation_coefficients"]["bias"]["value"]),
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def verify_pgexplainer_reload(
    artifact_path: Path,
    explainer_config: dict[str, Any],
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    materialization_node_id: int,
) -> bool:
    payload = torch.load(artifact_path, map_location=x.device, weights_only=False)
    reloaded = build_pgexplainer_from_config(explainer_config).to(x.device)
    wrap_pgexplainer(model, reloaded)
    materialize_pgexplainer(reloaded, model, x, edge_index, materialization_node_id)
    reloaded.load_state_dict(payload["algorithm_state_dict"])
    reloaded._curr_epoch = int(payload.get("algorithm_current_epoch", reloaded.epochs - 1))
    return all(torch.isfinite(parameter).all().item() for parameter in reloaded.parameters())


def write_epoch_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "mean_loss",
                "median_loss",
                "min_loss",
                "max_loss",
                "num_train_instances",
                "learning_rate",
                "temperature",
                "runtime_seconds",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_validation_health_check(
    *,
    explainer: Explainer,
    model: torch.nn.Module,
    data_cpu: Any,
    data: Any,
    validation_targets: list[PGExplainerTarget],
    frozen_predictions: torch.Tensor,
    num_hops: int,
    selected_edge_ratio: float,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for target in validation_targets:
        try:
            start = time.perf_counter()
            explanation = explainer(
                data.x,
                data.edge_index,
                target=frozen_predictions,
                index=target.node_id,
            )
            runtime = time.perf_counter() - start
            edge_mask = explanation.edge_mask.detach().cpu()
            candidate_edge_mask = computation_subgraph_edge_mask(data_cpu, target.node_id, num_hops)
            candidate_scores, candidate_labels = candidate_edge_scores_and_labels(
                edge_scores=edge_mask,
                ground_truth_edge_mask=data_cpu.edge_mask.detach().cpu(),
                candidate_edge_mask=candidate_edge_mask,
            )
            edge_metrics = compute_edge_ground_truth_metrics(candidate_scores, candidate_labels)
            rows.append(
                {
                    "node_id": target.node_id,
                    "candidate_edge_count": int(candidate_edge_mask.sum().item()),
                    "selected_edge_count": selected_edge_count_for_ratio(
                        int(candidate_edge_mask.sum().item()),
                        selected_edge_ratio,
                    ),
                    "edge_mask_finite": bool(torch.isfinite(edge_mask).all().item()),
                    "edge_mask_min": float(edge_mask.min().item()),
                    "edge_mask_max": float(edge_mask.max().item()),
                    "edge_mask_std": float(edge_mask.std(unbiased=False).item()),
                    "edge_mask_non_degenerate": bool(edge_mask.std(unbiased=False).item() > 0.0),
                    "edge_auroc": edge_metrics["edge_auroc"].value,
                    "edge_ap": edge_metrics["edge_ap"].value,
                    "runtime_seconds": runtime,
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "node_id": target.node_id,
                    "candidate_edge_count": None,
                    "selected_edge_count": None,
                    "edge_mask_finite": False,
                    "edge_mask_min": None,
                    "edge_mask_max": None,
                    "edge_mask_std": None,
                    "edge_mask_non_degenerate": False,
                    "edge_auroc": None,
                    "edge_ap": None,
                    "runtime_seconds": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return rows


def write_validation_health(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "node_id",
                "candidate_edge_count",
                "selected_edge_count",
                "edge_mask_finite",
                "edge_mask_min",
                "edge_mask_max",
                "edge_mask_std",
                "edge_mask_non_degenerate",
                "edge_auroc",
                "edge_ap",
                "runtime_seconds",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_validation_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["status"] == "success"]
    return {
        "validation_node_count": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(rows) - len(success_rows),
        "all_masks_finite": all(row["edge_mask_finite"] for row in success_rows),
        "all_masks_non_degenerate": all(row["edge_mask_non_degenerate"] for row in success_rows),
        "edge_auroc_mean": safe_mean(row["edge_auroc"] for row in success_rows),
        "edge_ap_mean": safe_mean(row["edge_ap"] for row in success_rows),
    }


def safe_mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return mean(numbers) if numbers else None


def current_git_commit() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
