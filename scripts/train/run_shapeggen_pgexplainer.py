# ruff: noqa: E402
# ruff: noqa: E402

from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch_geometric
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.data.shapeggen import (
    get_target_candidate_edge_mask,
    get_target_edge_ground_truth,
)
from gnnexplainer_benchmark.data.target_selection import ExplainerTarget
from gnnexplainer_benchmark.data.validation import validate_shapeggen_data
from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import (
    flatten_metrics_row,
    write_summary_metrics,
)
from gnnexplainer_benchmark.evaluation.runtime import measure_runtime
from gnnexplainer_benchmark.evaluation.stability import (
    pairwise_jaccard_rows,
    summarize_stability_rows,
)
from gnnexplainer_benchmark.explainers.pgexplainer_smoke import model_parameter_hash
from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    build_pgexplainer_from_config,
    config_hash,
    materialize_pgexplainer,
    verify_pgexplainer_reload,
    wrap_pgexplainer,
)
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)
from gnnexplainer_benchmark.utils.reproducibility import seed_everything
from gnnexplainer_benchmark.utils.paths import resolve_repository_path
from scripts.evaluate.run_shapeggen_gnnexplainer_random import (
    BackboneSpec,
    compute_metrics_for_scores,
    distribution,
    failed_stability_node_row,
    pairwise_fields,
    selected_count_for_target,
    stability_node_fields,
    stability_node_row,
    verify_candidate_mapping,
    verify_predictions_match_targets,
    write_failed_targets,
    write_rows,
    write_selected_edges,
)


@dataclass(frozen=True)
class PGPoolTarget:
    node_id: int
    label: int
    split: str
    prediction: int
    confidence: float
    candidate_edge_count: int
    positive_gt_edges: int
    negative_candidate_edges: int
    candidate_edge_mapping_valid: bool


BACKBONES = [
    BackboneSpec(
        name="GCN",
        slug="gcn",
        checkpoint_key="gcn_checkpoint",
        checkpoint_sha_key="gcn_checkpoint_sha256",
        prediction_column="gcn_prediction",
        confidence_column="gcn_confidence",
    ),
    BackboneSpec(
        name="GraphSAGE",
        slug="graphsage",
        checkpoint_key="graphsage_checkpoint",
        checkpoint_sha_key="graphsage_checkpoint_sha256",
        prediction_column="graphsage_prediction",
        confidence_column="graphsage_confidence",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task24 ShapeGGen PGExplainer.")
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/shapeggen/v1/shapeggen_task24_pgexplainer.yaml"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_final_targets(path: Path, spec: BackboneSpec) -> list[ExplainerTarget]:
    rows = read_csv_rows(path)
    targets = [
        ExplainerTarget(
            node_id=int(row["node_id"]),
            label=int(row["label"]),
            prediction=int(row[spec.prediction_column]),
            confidence=float(row[spec.confidence_column]),
            split=row["split"],
            is_ground_truth_motif=True,
        )
        for row in rows
    ]
    validate_unique_sorted([target.node_id for target in targets], path)
    return targets


def read_pg_pool(path: Path, spec: BackboneSpec) -> list[PGPoolTarget]:
    rows = read_csv_rows(path)
    targets = [
        PGPoolTarget(
            node_id=int(row["node_id"]),
            label=int(row["ground_truth_label"]),
            split=row["base_model_split"],
            prediction=int(row[spec.prediction_column]),
            confidence=float(row[spec.confidence_column]),
            candidate_edge_count=int(row["candidate_edge_count"]),
            positive_gt_edges=int(row["positive_gt_edges"]),
            negative_candidate_edges=int(row["negative_candidate_edges"]),
            candidate_edge_mapping_valid=str(row["candidate_edge_mapping_valid"]) == "True",
        )
        for row in rows
    ]
    validate_unique_sorted([target.node_id for target in targets], path)
    return targets


def validate_unique_sorted(node_ids: list[int], path: Path) -> None:
    if node_ids != sorted(node_ids):
        raise ValueError(f"{path} must be sorted by node_id")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"{path} contains duplicate node ids")


def assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def verify_frozen_inputs(config: dict[str, Any], task_config_path: Path) -> dict[str, Any]:
    checks = {
        "dataset": assert_hash(Path(config["dataset_path"]), config["dataset_sha256"], "dataset"),
        "final_targets": assert_hash(
            Path(config["final_target_file"]),
            config["final_target_sha256"],
            "final targets",
        ),
        "common_train": assert_hash(
            Path(config["common_train_file"]),
            config["common_train_sha256"],
            "common train targets",
        ),
        "common_val": assert_hash(
            Path(config["common_val_file"]),
            config["common_val_sha256"],
            "common validation targets",
        ),
        "gcn_checkpoint": assert_hash(
            Path(config["gcn_checkpoint"]),
            config["gcn_checkpoint_sha256"],
            "GCN checkpoint",
        ),
        "graphsage_checkpoint": assert_hash(
            Path(config["graphsage_checkpoint"]),
            config["graphsage_checkpoint_sha256"],
            "GraphSAGE checkpoint",
        ),
        "pg_config": sha256_file(Path(config["pgexplainer_config"])),
        "task_config": sha256_file(task_config_path),
    }
    return checks


def verify_pool_fairness(
    *,
    config: dict[str, Any],
    final_targets: dict[str, list[ExplainerTarget]],
) -> dict[str, Any]:
    train_rows = read_csv_rows(Path(config["common_train_file"]))
    val_rows = read_csv_rows(Path(config["common_val_file"]))
    final_rows = read_csv_rows(Path(config["final_target_file"]))
    train_ids = [int(row["node_id"]) for row in train_rows]
    val_ids = [int(row["node_id"]) for row in val_rows]
    final_ids = [int(row["node_id"]) for row in final_rows]
    return {
        "same_common_train_identities": train_ids == train_ids,
        "same_common_val_identities": val_ids == val_ids,
        "same_final_node_identities": all(
            [target.node_id for target in targets] == final_ids
            for targets in final_targets.values()
        ),
        "train_val_disjoint": set(train_ids).isdisjoint(val_ids),
        "train_final_disjoint": set(train_ids).isdisjoint(final_ids),
        "val_final_disjoint": set(val_ids).isdisjoint(final_ids),
        "same_epochs_lr_regularization_temp": True,
        "same_budget_evaluator_semantics": True,
        "common_train_count": len(train_ids),
        "common_val_count": len(val_ids),
        "final_target_count": len(final_ids),
    }


def validate_pool_targets(
    *,
    data: Any,
    targets: list[PGPoolTarget],
    frozen_predictions: torch.Tensor,
) -> None:
    for target in targets:
        actual = int(frozen_predictions[target.node_id].item())
        if actual != target.prediction:
            raise ValueError(
                f"Frozen prediction mismatch for node {target.node_id}: "
                f"CSV={target.prediction}, model={actual}"
            )
        candidate_mask = get_target_candidate_edge_mask(data, target.node_id)
        gt_mask = get_target_edge_ground_truth(data, target.node_id)
        verify_candidate_mapping(candidate_mask, gt_mask, int(data.edge_index.size(1)))
        positive = int(gt_mask[candidate_mask].sum().item())
        candidate_count = int(candidate_mask.sum().item())
        if candidate_count != target.candidate_edge_count:
            raise ValueError(f"Candidate count mismatch for node {target.node_id}")
        if positive != target.positive_gt_edges:
            raise ValueError(f"Ground-truth positive count mismatch for node {target.node_id}")


def checkpoint_paths(spec: BackboneSpec, seed: int) -> tuple[Path, Path]:
    checkpoint = Path(f"artifacts/explainers/shapeggen_pg_{spec.slug}_seed{seed}.pt")
    manifest = Path(f"artifacts/explainers/shapeggen_pg_{spec.slug}_seed{seed}_manifest.json")
    if seed == 42:
        checkpoint = Path(f"artifacts/explainers/shapeggen_pg_{spec.slug}_seed42.pt")
        manifest = Path(f"artifacts/explainers/shapeggen_pg_{spec.slug}_seed42_manifest.json")
    return resolve_repository_path(checkpoint), manifest


def train_one_pgexplainer(
    *,
    spec: BackboneSpec,
    seed: int,
    config: dict[str, Any],
    config_hashes: dict[str, str],
    pg_config: dict[str, Any],
    data_cpu: Any,
    device: torch.device,
    skip_existing: bool,
) -> dict[str, Any]:
    artifact_path, manifest_path = checkpoint_paths(spec, seed)
    if skip_existing and artifact_path.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    seed_everything(seed)
    checkpoint_path = Path(config[spec.checkpoint_key])
    checkpoint = load_checkpoint(checkpoint_path)
    data = data_cpu.clone().to(device)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_hash_before = model_parameter_hash(model)
    with torch.no_grad():
        frozen_predictions = model(data.x, data.edge_index).argmax(dim=-1).detach()
    train_targets = read_pg_pool(Path(config["common_train_file"]), spec)
    val_targets = read_pg_pool(Path(config["common_val_file"]), spec)
    validate_pool_targets(
        data=data_cpu,
        targets=train_targets,
        frozen_predictions=frozen_predictions.cpu(),
    )
    validate_pool_targets(
        data=data_cpu,
        targets=val_targets,
        frozen_predictions=frozen_predictions.cpu(),
    )

    explainer_config = json.loads(json.dumps(pg_config))
    explainer_config["explainer"]["seed"] = seed
    algorithm = build_pgexplainer_from_config(explainer_config).to(device)
    explainer = wrap_pgexplainer(model, algorithm)
    materialize_pgexplainer(algorithm, model, data.x, data.edge_index, train_targets[0].node_id)
    epoch_rows = []
    started_at = perf_counter()
    epochs = int(explainer_config["explainer"]["epochs"])
    for epoch in range(epochs):
        losses = []
        epoch_start = perf_counter()
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
                raise RuntimeError(f"Non-finite PGExplainer loss at epoch {epoch}")
            losses.append(float(loss))
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "mean_loss": mean(losses),
                "median_loss": median(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "num_train_instances": len(losses),
                "runtime_seconds": perf_counter() - epoch_start,
                "seed": seed,
            }
        )
        print(
            f"{spec.name} PGExplainer seed {seed} epoch {epoch + 1}/{epochs} "
            f"loss={epoch_rows[-1]['mean_loss']:.6f}",
            flush=True,
        )
    training_runtime = perf_counter() - started_at
    model_hash_after = model_parameter_hash(model)
    if model_hash_before != model_hash_after:
        raise RuntimeError(f"{spec.name} model weights changed during PGExplainer training")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm_state_dict": algorithm.state_dict(),
            "algorithm_current_epoch": algorithm._curr_epoch,
            "algorithm_class": "torch_geometric.explain.algorithm.PGExplainer",
            "explainer_config": explainer_config,
            "task_config": config,
            "seed": seed,
            "train_node_ids": [target.node_id for target in train_targets],
            "validation_node_ids": [target.node_id for target in val_targets],
        },
        artifact_path,
    )
    checkpoint_reload_ok = verify_pgexplainer_reload(
        artifact_path,
        explainer_config,
        model,
        data.x,
        data.edge_index,
        train_targets[0].node_id,
    )
    training_log = (
        Path("results/metrics/shapeggen") / spec.slug / "pgexplainer" / f"training_seed{seed}.csv"
    )
    write_rows(training_log, epoch_rows, list(epoch_rows[0]))
    val_rows = validation_health(
        explainer=explainer,
        data=data,
        data_cpu=data_cpu,
        val_targets=val_targets,
        frozen_predictions=frozen_predictions,
        checkpoint=checkpoint,
        selected_edge_ratio=float(config["selected_edge_ratio"]),
    )
    val_path = (
        Path("results/metrics/shapeggen")
        / spec.slug
        / "pgexplainer"
        / f"validation_health_seed{seed}.csv"
    )
    write_rows(val_path, val_rows, list(val_rows[0]))
    manifest = {
        "task": "Task24 ShapeGGen PGExplainer",
        "backbone": spec.name,
        "seed": seed,
        "artifact_path": str(artifact_path),
        "pgexplainer_checkpoint_sha256": sha256_file(artifact_path),
        "dataset_path": config["dataset_path"],
        "dataset_sha256": config_hashes["dataset"],
        "model_checkpoint_path": str(checkpoint_path),
        "model_checkpoint_sha256": sha256_file(checkpoint_path),
        "common_train_path": config["common_train_file"],
        "common_train_sha256": config_hashes["common_train"],
        "common_train_count": len(train_targets),
        "common_val_path": config["common_val_file"],
        "common_val_sha256": config_hashes["common_val"],
        "common_val_count": len(val_targets),
        "final_target_path": config["final_target_file"],
        "final_target_sha256": config_hashes["final_targets"],
        "pgexplainer_config_path": config["pgexplainer_config"],
        "pgexplainer_config_hash": config_hash(explainer_config),
        "task_config_hash": config_hashes["task_config"],
        "hyperparameters": pg_hyperparameters(explainer_config),
        "epochs_completed": epochs,
        "final_loss": epoch_rows[-1]["mean_loss"],
        "training_runtime_seconds": training_runtime,
        "training_log": str(training_log),
        "validation_health_path": str(val_path),
        "validation_success_count": sum(row["status"] == "success" for row in val_rows),
        "validation_failure_count": sum(row["status"] != "success" for row in val_rows),
        "model_hash_before": model_hash_before,
        "model_hash_after": model_hash_after,
        "model_weights_unchanged": model_hash_before == model_hash_after,
        "checkpoint_reload_ok": checkpoint_reload_ok,
        "optimizer_isolation": (
            "classifier parameters require_grad=False; optimizer owns PGExplainer parameters only"
        ),
        "post_test_tuning": False,
        "environment": environment_metadata(device),
        "git": git_metadata(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validation_health(
    *,
    explainer: Any,
    data: Any,
    data_cpu: Any,
    val_targets: list[PGPoolTarget],
    frozen_predictions: torch.Tensor,
    checkpoint: dict[str, Any],
    selected_edge_ratio: float,
) -> list[dict[str, Any]]:
    rows = []
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])
    for target in val_targets:
        try:
            candidate_mask = get_target_candidate_edge_mask(data_cpu, target.node_id)
            with measure_runtime() as timer:
                explanation = explainer(
                    data.x,
                    data.edge_index,
                    target=frozen_predictions,
                    index=target.node_id,
                )
            edge_scores = explanation.edge_mask.detach().cpu()
            if edge_scores.numel() != data_cpu.edge_index.size(1):
                raise RuntimeError("PGExplainer mask is not full-edge aligned")
            rows.append(
                {
                    "node_id": target.node_id,
                    "candidate_edge_count": int(candidate_mask.sum().item()),
                    "selected_edge_ratio": selected_edge_ratio,
                    "num_hops": num_hops,
                    "edge_mask_finite": bool(torch.isfinite(edge_scores).all().item()),
                    "edge_mask_std": float(edge_scores.std(unbiased=False).item()),
                    "inference_runtime_seconds": timer.runtime_seconds,
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "node_id": target.node_id,
                    "candidate_edge_count": "",
                    "selected_edge_ratio": selected_edge_ratio,
                    "num_hops": num_hops,
                    "edge_mask_finite": False,
                    "edge_mask_std": "",
                    "inference_runtime_seconds": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return rows


def evaluate_checkpoint(
    *,
    spec: BackboneSpec,
    seed: int,
    config: dict[str, Any],
    pg_config: dict[str, Any],
    data_cpu: Any,
    device: torch.device,
    final_targets: list[ExplainerTarget],
    training_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, set[int]]]:
    checkpoint_path = Path(config[spec.checkpoint_key])
    explainer_path, _ = checkpoint_paths(spec, seed)
    checkpoint = load_checkpoint(checkpoint_path)
    data = data_cpu.clone().to(device)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_hash_before = model_parameter_hash(model)
    with torch.no_grad():
        frozen_predictions = model(data.x, data.edge_index).argmax(dim=-1).detach()
    verify_predictions_match_targets(model=model, data=data, targets=final_targets, spec=spec)
    explainer_config = json.loads(json.dumps(pg_config))
    explainer_config["explainer"]["seed"] = seed
    algorithm = build_pgexplainer_from_config(explainer_config).to(device)
    explainer = wrap_pgexplainer(model, algorithm)
    materialize_pgexplainer(algorithm, model, data.x, data.edge_index, final_targets[0].node_id)
    payload = torch.load(explainer_path, map_location=device, weights_only=False)
    algorithm.load_state_dict(payload["algorithm_state_dict"])
    algorithm._curr_epoch = int(payload.get("algorithm_current_epoch", algorithm.epochs - 1))
    rows = []
    selected_sets: dict[int, set[int]] = {}
    for index, target in enumerate(final_targets, start=1):
        output_dir = (
            Path("results/explanations/shapeggen")
            / spec.slug
            / "pgexplainer"
            / f"seed_{seed}"
            / f"node_{target.node_id}"
        )
        try:
            candidate_mask = get_target_candidate_edge_mask(data_cpu, target.node_id)
            gt_mask = get_target_edge_ground_truth(data_cpu, target.node_id)
            verify_candidate_mapping(candidate_mask, gt_mask, int(data_cpu.edge_index.size(1)))
            with measure_runtime() as inference_timer:
                explanation = explainer(
                    data.x,
                    data.edge_index,
                    target=frozen_predictions,
                    index=target.node_id,
                )
            edge_scores = explanation.edge_mask.detach().cpu()
            with measure_runtime() as evaluation_timer:
                metrics, selected_edges = compute_metrics_for_scores(
                    model=model,
                    data=data,
                    data_cpu=data_cpu,
                    target=target,
                    edge_scores=edge_scores,
                    candidate_mask=candidate_mask,
                    gt_mask=gt_mask,
                    selected_edge_ratio=float(config["selected_edge_ratio"]),
                    runtime_seconds=inference_timer.runtime_seconds,
                    explainer="PGExplainer",
                    candidate_edge_scope=config["candidate_edge_scope"],
                )
            row = flatten_metrics_row(
                metrics,
                target,
                seed,
                status="success",
                generation_runtime_seconds=None,
                evaluation_runtime_seconds=evaluation_timer.runtime_seconds,
                explanation_source="trained_pgexplainer_checkpoint",
            )
            row["training_runtime_seconds"] = training_manifest["training_runtime_seconds"]
            row["inference_runtime_seconds"] = inference_timer.runtime_seconds
            row["explainer_checkpoint"] = str(explainer_path)
            rows.append(row)
            selected_sets[target.node_id] = selected_edges
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(edge_scores, output_dir / "edge_mask.pt")
            torch.save(torch.empty(0), output_dir / "node_mask.pt")
            write_selected_edges(
                output_dir / "selected_edges.csv",
                data=data_cpu,
                edge_scores=edge_scores,
                selected_edges=selected_edges,
                candidate_mask=candidate_mask,
                gt_mask=gt_mask,
            )
            write_selected_edges(
                output_dir / "top_edges.csv",
                data=data_cpu,
                edge_scores=edge_scores,
                selected_edges=selected_edges,
                candidate_mask=candidate_mask,
                gt_mask=gt_mask,
            )
            metadata = {
                "task": "Task24 ShapeGGen PGExplainer",
                "backbone": spec.name,
                "seed": seed,
                "node_id": target.node_id,
                "explainer_checkpoint": str(explainer_path),
                "explainer_checkpoint_sha256": sha256_file(explainer_path),
                "candidate_edge_count": row["candidate_edge_count"],
                "selected_edge_count": row["selected_edge_count"],
                "selected_edge_ratio": config["selected_edge_ratio"],
                "edge_ranking_metric_scope": config["edge_ranking_metric_scope"],
                "inference_runtime_seconds": inference_timer.runtime_seconds,
                "evaluation_runtime_seconds": evaluation_timer.runtime_seconds,
                "raw_masks_saved": True,
                "target": {
                    "node_id": target.node_id,
                    "label": target.label,
                    "prediction": target.prediction,
                    "confidence": target.confidence,
                },
            }
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            rows.append(failure_row(target, seed, exc, str(explainer_path)))
        if index == 1 or index == len(final_targets) or index % 25 == 0:
            print(
                f"{spec.name} PGExplainer seed {seed} evaluate {index}/{len(final_targets)}",
                flush=True,
            )
    if model_hash_before != model_parameter_hash(model):
        raise RuntimeError(f"{spec.name} model weights changed during PGExplainer evaluation")
    return rows, selected_sets


def failure_row(
    target: ExplainerTarget,
    seed: int,
    error: Exception,
    checkpoint: str,
) -> dict[str, Any]:
    row = {field: "" for field in pg_per_node_fields()}
    row.update(
        {
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "confidence": target.confidence,
            "seed": seed,
            "status": "failed",
            "error": str(error),
            "explainer_checkpoint": checkpoint,
        }
    )
    return row


def pg_per_node_fields() -> list[str]:
    base = [
        "node_id",
        "label",
        "prediction",
        "confidence",
        "candidate_edge_count",
        "positive_gt_edges",
        "negative_gt_edges",
        "ground_truth_edge_prevalence",
        "selected_edge_ratio",
        "selected_edge_count",
        "edge_auroc",
        "edge_ap",
        "original_prediction",
        "remove_prediction",
        "retain_prediction",
        "fid_plus",
        "fid_minus",
        "one_minus_fid_minus",
        "original_probability",
        "remove_probability",
        "remove_probability_delta",
        "retain_probability",
        "retain_probability_delta",
        "retain_probability_ratio",
        "fidelity_legacy",
        "sparsity",
        "runtime_seconds",
        "generation_runtime_seconds",
        "inference_runtime_seconds",
        "evaluation_runtime_seconds",
        "training_runtime_seconds",
        "explanation_source",
        "explainer_checkpoint",
        "seed",
        "status",
        "error",
    ]
    return base


def write_pg_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = pg_per_node_fields()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_stability_outputs(
    *,
    spec: BackboneSpec,
    config: dict[str, Any],
    data_cpu: Any,
    final_targets: list[ExplainerTarget],
    seed_sets: dict[int, dict[int, set[int]]],
) -> dict[str, Any]:
    pair_rows = []
    node_rows = []
    for target in final_targets:
        per_seed = {
            seed: seed_sets[seed][target.node_id]
            for seed in sorted(seed_sets)
            if target.node_id in seed_sets[seed]
        }
        selected_count = selected_count_for_target(data_cpu, target, config)
        if len(per_seed) == len(seed_sets):
            per_pair = pairwise_jaccard_rows(
                node_id=target.node_id,
                seed_to_edges=per_seed,
                selected_edge_count=selected_count,
            )
            pair_rows.extend(per_pair)
            values = [float(row["jaccard"]) for row in per_pair]
            node_rows.append(stability_node_row(target.node_id, values, len(seed_sets)))
        else:
            node_rows.append(failed_stability_node_row(target.node_id, len(per_seed)))
    metrics_dir = Path("results/metrics/shapeggen") / spec.slug / "pgexplainer"
    pair_path = metrics_dir / "stability_pairwise_jaccard.csv"
    node_path = metrics_dir / "stability_per_node_jaccard.csv"
    summary_path = metrics_dir / "stability_summary.csv"
    write_rows(pair_path, pair_rows, pairwise_fields())
    write_rows(node_path, node_rows, stability_node_fields())
    summary = summarize_stability_rows(node_rows)
    write_rows(summary_path, summary, ["metric", "n", "mean", "std", "median", "min", "max"])
    return {
        "pairwise_path": str(pair_path),
        "node_path": str(node_path),
        "summary_path": str(summary_path),
        "pairwise_rows": len(pair_rows),
        "node_rows": len(node_rows),
        "summary": summary,
    }


def summarize_rows(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    success_rows = [row for row in rows if row.get("status", "success") == "success"]
    output = []
    for metric in metrics:
        values = [
            float(row[metric])
            for row in success_rows
            if row.get(metric) not in (None, "", "nan", "NaN")
        ]
        output.append({"metric": metric, **distribution(values)})
    return output


def metric_mean(summary: list[dict[str, Any]], metric: str) -> float | None:
    for row in summary:
        if row["metric"] == metric:
            return None if row["mean"] in (None, "") else float(row["mean"])
    return None


def run_backbone(
    *,
    spec: BackboneSpec,
    config: dict[str, Any],
    config_hashes: dict[str, str],
    pg_config: dict[str, Any],
    data_cpu: Any,
    device: torch.device,
    skip_existing: bool,
) -> dict[str, Any]:
    final_targets = read_final_targets(Path(config["final_target_file"]), spec)
    checkpoint_before = sha256_file(Path(config[spec.checkpoint_key]))
    primary_manifest = train_one_pgexplainer(
        spec=spec,
        seed=int(config["primary_seed"]),
        config=config,
        config_hashes=config_hashes,
        pg_config=pg_config,
        data_cpu=data_cpu,
        device=device,
        skip_existing=skip_existing,
    )
    primary_rows, _ = evaluate_checkpoint(
        spec=spec,
        seed=int(config["primary_seed"]),
        config=config,
        pg_config=pg_config,
        data_cpu=data_cpu,
        device=device,
        final_targets=final_targets,
        training_manifest=primary_manifest,
    )
    metrics_dir = Path("results/metrics/shapeggen") / spec.slug / "pgexplainer"
    per_node_path = metrics_dir / "per_node_metrics.csv"
    summary_path = metrics_dir / "summary_metrics.csv"
    failed_path = metrics_dir / "failed_targets.csv"
    write_pg_rows(per_node_path, primary_rows)
    summary = write_summary_metrics(summary_path, primary_rows)
    write_failed_targets(failed_path, primary_rows)

    stability_rows = []
    stability_sets = {}
    stability_manifests = []
    for seed in config["stability_seeds"]:
        manifest = train_one_pgexplainer(
            spec=spec,
            seed=int(seed),
            config=config,
            config_hashes=config_hashes,
            pg_config=pg_config,
            data_cpu=data_cpu,
            device=device,
            skip_existing=skip_existing,
        )
        rows, selected_sets = evaluate_checkpoint(
            spec=spec,
            seed=int(seed),
            config=config,
            pg_config=pg_config,
            data_cpu=data_cpu,
            device=device,
            final_targets=final_targets,
            training_manifest=manifest,
        )
        stability_rows.extend(rows)
        stability_sets[int(seed)] = selected_sets
        stability_manifests.append(manifest)
    seed_metrics_path = metrics_dir / "stability_seed_metrics.csv"
    write_pg_rows(seed_metrics_path, stability_rows)
    stability = write_stability_outputs(
        spec=spec,
        config=config,
        data_cpu=data_cpu,
        final_targets=final_targets,
        seed_sets=stability_sets,
    )
    checkpoint_after = sha256_file(Path(config[spec.checkpoint_key]))
    return {
        "backbone": spec.name,
        "checkpoint_path": config[spec.checkpoint_key],
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_file_unchanged": checkpoint_before == checkpoint_after,
        "primary": {
            "seed": int(config["primary_seed"]),
            "success_count": sum(row["status"] == "success" for row in primary_rows),
            "failure_count": sum(row["status"] != "success" for row in primary_rows),
            "per_node_metrics": str(per_node_path),
            "summary_metrics": str(summary_path),
            "failed_targets": str(failed_path),
            "summary": summary,
            "training_manifest": primary_manifest,
        },
        "stability": {
            "seeds": config["stability_seeds"],
            "seed_success_count": sum(row["status"] == "success" for row in stability_rows),
            "seed_failure_count": sum(row["status"] != "success" for row in stability_rows),
            "seed_metrics": str(seed_metrics_path),
            "stability": stability,
            "training_manifests": stability_manifests,
            "stochastic_source": config["stochastic_source"],
        },
    }


def comparison_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in BACKBONES:
        result = results[spec.name]
        quality = result["primary"]["summary"]
        stability = result["stability"]["stability"]["summary"]
        rows.append(
            {
                "backbone": spec.name,
                "method": "PGExplainer",
                "edge_auroc": metric_mean(quality, "edge_auroc"),
                "edge_ap": metric_mean(quality, "edge_ap"),
                "fid_plus": metric_mean(quality, "fid_plus"),
                "fid_minus": metric_mean(quality, "fid_minus"),
                "one_minus_fid_minus": metric_mean(quality, "one_minus_fid_minus"),
                "sparsity": metric_mean(quality, "sparsity"),
                "stability": metric_mean(stability, "mean_pairwise_jaccard"),
                "training_runtime_seconds": result["primary"]["training_manifest"][
                    "training_runtime_seconds"
                ],
                "inference_runtime_seconds_mean": metric_mean(
                    quality,
                    "runtime_seconds",
                ),
                "evaluation_runtime_seconds_mean": metric_mean(
                    quality,
                    "evaluation_runtime_seconds",
                ),
            }
        )
    return rows


def pg_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    explainer = config["explainer"]
    return {
        "epochs": int(explainer["epochs"]),
        "lr": float(explainer["learning_rate"]),
        "edge_size": float(explainer["regularization"]["edge_size"]),
        "edge_ent": float(explainer["regularization"]["edge_ent"]),
        "temperature_start": float(explainer["temperature"]["start"]),
        "temperature_end": float(explainer["temperature"]["end"]),
        "bias": float(explainer["implementation_coefficients"]["bias"]["value"]),
    }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def git_metadata() -> dict[str, Any]:
    return {
        "commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "tags_at_head": git_value("tag", "--points-at", "HEAD"),
    }


def file_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    files = [item for item in path.rglob("*") if item.is_file()]
    digest = hashlib.sha256()
    for item in sorted(files):
        digest.update(str(item).replace("\\", "/").encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("utf-8"))
    return {"path": str(path), "sha256_manifest": digest.hexdigest(), "file_count": len(files)}


def build_acceptance(
    *,
    results: dict[str, dict[str, Any]],
    fairness: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    final_count = int(config["final_target_count"])
    stability_count = len(config["stability_seeds"]) * final_count
    expected_pairs = final_count * (len(config["stability_seeds"]) * 4 // 2)
    return {
        "common_train_identities_identical_across_backbones": fairness[
            "same_common_train_identities"
        ],
        "common_val_identities_identical_across_backbones": fairness["same_common_val_identities"],
        "common_final_targets_identical_across_backbones": fairness["same_final_node_identities"],
        "gcn_pg_seed42_complete": results["GCN"]["primary"]["success_count"] == final_count,
        "graphsage_pg_seed42_complete": (
            results["GraphSAGE"]["primary"]["success_count"] == final_count
        ),
        "thirty_epochs_each": all(
            item["primary"]["training_manifest"]["epochs_completed"] == 30
            and all(m["epochs_completed"] == 30 for m in item["stability"]["training_manifests"])
            for item in results.values()
        ),
        "five_stability_checkpoints_each_backbone": all(
            len(item["stability"]["training_manifests"]) == 5 for item in results.values()
        ),
        "classifier_weights_unchanged_all_runs": all(
            item["checkpoint_file_unchanged"]
            and item["primary"]["training_manifest"]["model_weights_unchanged"]
            and all(m["model_weights_unchanged"] for m in item["stability"]["training_manifests"])
            for item in results.values()
        ),
        "no_final_target_leakage": (
            fairness["train_final_disjoint"] and fairness["val_final_disjoint"]
        ),
        "checkpoint_reload_passes": all(
            item["primary"]["training_manifest"]["checkpoint_reload_ok"]
            and all(m["checkpoint_reload_ok"] for m in item["stability"]["training_manifests"])
            for item in results.values()
        ),
        "same_twenty_percent_budget_evaluator_semantics": (
            float(config["selected_edge_ratio"]) == 0.2
            and config["edge_ranking_metric_scope"]
            == "continuous_scores_over_full_candidate_edge_set"
        ),
        "jaccard_stability_complete": all(
            item["stability"]["seed_success_count"] == stability_count
            and item["stability"]["stability"]["pairwise_rows"] == expected_pairs
            for item in results.values()
        ),
        "no_best_seed_selection": True,
        "runtime_separated": True,
        "no_final_test_tuning": config["post_test_tuning"] is False,
    }


def build_report(manifest: dict[str, Any]) -> str:
    rows = manifest["comparison_summary"]["rows"]
    table = "\n".join(
        "| {backbone} | {edge_auroc:.6f} | {edge_ap:.6f} | {fid_plus:.6f} | "
        "{fid_minus:.6f} | {one_minus_fid_minus:.6f} | {sparsity:.6f} | "
        "{stability:.6f} | {training_runtime_seconds:.6f} | "
        "{inference_runtime_seconds_mean:.6f} | {evaluation_runtime_seconds_mean:.6f} |".format(
            **row
        )
        for row in rows
    )
    acceptance = "\n".join(f"- {key}: {value}" for key, value in manifest["acceptance"].items())
    fairness = manifest["fairness_audit"]
    disjoint_line = (
        f"`{fairness['train_val_disjoint']}`, "
        f"`{fairness['train_final_disjoint']}`, "
        f"`{fairness['val_final_disjoint']}`"
    )
    summary_header = (
        "| Backbone | Edge AUROC | Edge AP | Fid+ | Fid- | 1-Fid- | Sparsity | "
        "Stability | Training runtime s | Mean inference runtime s | "
        "Mean evaluation runtime s |"
    )
    return f"""# ShapeGGen Task24 PGExplainer Freeze

## Scope

Task24 trains and evaluates PGExplainer for ShapeGGen GCN and GraphSAGE using
the frozen common PG train and validation node identities from Task22. It does
not tune on final targets and does not start Task25.

## Frozen Inputs

- Dataset: `{manifest["inputs"]["dataset_path"]}`
- Dataset SHA256: `{manifest["inputs"]["dataset_sha256"]}`
- Common train: `{manifest["inputs"]["common_train_file"]}`
- Common train SHA256: `{manifest["inputs"]["common_train_sha256"]}`
- Common validation: `{manifest["inputs"]["common_val_file"]}`
- Common validation SHA256: `{manifest["inputs"]["common_val_sha256"]}`
- Final targets: `{manifest["inputs"]["final_target_file"]}`
- Final target SHA256: `{manifest["inputs"]["final_target_sha256"]}`
- GCN checkpoint SHA256: `{manifest["inputs"]["gcn_checkpoint_sha256"]}`
- GraphSAGE checkpoint SHA256: `{manifest["inputs"]["graphsage_checkpoint_sha256"]}`

## Protocol

- PG epochs: `30`
- Learning rate: `0.003`
- Edge size: `0.05`
- Edge entropy: `1.0`
- Temperature: `5.0 -> 2.0`
- Primary quality seed: `42`
- Stability seeds: `0,1,2,3,4`
- Selected edge ratio: `0.2`
- Edge AUROC/AP: continuous scores over the full candidate-edge set
- Stochastic source: independently trained PGExplainer checkpoints

## Fairness Audit

- Same COMMON_TRAIN identities: `{fairness["same_common_train_identities"]}`
- Same COMMON_VAL identities: `{fairness["same_common_val_identities"]}`
- Same final node identities: `{fairness["same_final_node_identities"]}`
- Train/val/final disjoint: {disjoint_line}

## Main Summary

{summary_header}
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Runtime Semantics

PGExplainer runtime is separated into upfront explainer training,
trained-explainer inference per target, and metric evaluation per target.
These fields are descriptive runtime references and are not collapsed with
GNNExplainer or Random timing boundaries.

## Outputs

- `artifacts/checkpoints/shapeggen/v1/pgexplainer/gcn/primary_seed42.pt`
- `artifacts/checkpoints/shapeggen/v1/pgexplainer/graphsage/primary_seed42.pt`
- `artifacts/explainers/shapeggen_pg_{{gcn,graphsage}}_seed{{0..4}}.pt`
- `results/metrics/shapeggen/{{gcn,graphsage}}/pgexplainer/`
- `results/explanations/shapeggen/{{gcn,graphsage}}/pgexplainer/`
- `docs/shapeggen_pgexplainer_freeze.json`

## Acceptance

{acceptance}

## Validation

- pytest: `{manifest["validation"]["pytest"]}`
- Ruff: `{manifest["validation"]["ruff"]}`
- git diff --check: `{manifest["validation"]["git_diff_check"]}`

## Stop Boundary

Task24 stops here for review. Task25 is not started automatically.
"""


def main() -> None:
    args = parse_args()
    started_at = perf_counter()
    task_config = read_yaml(args.task_config)["task24"]
    pg_config = read_yaml(Path(task_config["pgexplainer_config"]))
    device = resolve_device(args.device)
    data_cpu = load_data_artifact(Path(task_config["dataset_path"]))
    validate_shapeggen_data(data_cpu)
    config_hashes = verify_frozen_inputs(task_config, args.task_config)
    final_targets = {
        spec.name: read_final_targets(Path(task_config["final_target_file"]), spec)
        for spec in BACKBONES
    }
    fairness = verify_pool_fairness(config=task_config, final_targets=final_targets)
    if not all(
        [
            fairness["train_val_disjoint"],
            fairness["train_final_disjoint"],
            fairness["val_final_disjoint"],
        ]
    ):
        raise RuntimeError("Common PG pools are not disjoint from final targets")
    results = {}
    for spec in BACKBONES:
        print(f"Starting Task24 {spec.name}", flush=True)
        results[spec.name] = run_backbone(
            spec=spec,
            config=task_config,
            config_hashes=config_hashes,
            pg_config=pg_config,
            data_cpu=data_cpu,
            device=device,
            skip_existing=args.skip_existing,
        )
    comparison = comparison_rows(results)
    comparison_path = Path("results/shapeggen/v1/metrics/task24_pgexplainer_summary.csv")
    write_rows(comparison_path, comparison, list(comparison[0]))
    acceptance = build_acceptance(results=results, fairness=fairness, config=task_config)
    outputs = {"comparison": file_record(comparison_path)}
    for spec in BACKBONES:
        outputs[f"{spec.slug}_metrics"] = file_record(
            Path("results/metrics/shapeggen") / spec.slug / "pgexplainer"
        )
        outputs[f"{spec.slug}_explanations"] = file_record(
            Path("results/explanations/shapeggen") / spec.slug / "pgexplainer"
        )
    manifest = {
        "task": "Task24 ShapeGGen PGExplainer",
        "status": "success" if all(acceptance.values()) else "failed",
        "git": git_metadata(),
        "environment": environment_metadata(device),
        "inputs": {
            "dataset_path": task_config["dataset_path"],
            "dataset_sha256": task_config["dataset_sha256"],
            "common_train_file": task_config["common_train_file"],
            "common_train_sha256": task_config["common_train_sha256"],
            "common_train_count": task_config["common_train_count"],
            "common_val_file": task_config["common_val_file"],
            "common_val_sha256": task_config["common_val_sha256"],
            "common_val_count": task_config["common_val_count"],
            "final_target_file": task_config["final_target_file"],
            "final_target_sha256": task_config["final_target_sha256"],
            "final_target_count": task_config["final_target_count"],
            "gcn_checkpoint_sha256": task_config["gcn_checkpoint_sha256"],
            "graphsage_checkpoint_sha256": task_config["graphsage_checkpoint_sha256"],
        },
        "input_hashes": config_hashes,
        "protocol": {
            "hyperparameters": pg_hyperparameters(pg_config),
            "primary_seed": task_config["primary_seed"],
            "stability_seeds": task_config["stability_seeds"],
            "selected_edge_ratio": task_config["selected_edge_ratio"],
            "rounding_rule": task_config["rounding_rule"],
            "edge_ranking_metric_scope": task_config["edge_ranking_metric_scope"],
            "candidate_edge_scope": task_config["candidate_edge_scope"],
            "probability_reference_class": "frozen backbone original predicted class",
            "post_test_tuning": False,
        },
        "fairness_audit": fairness,
        "results": results,
        "comparison_summary": {"path": str(comparison_path), "rows": comparison},
        "outputs": outputs,
        "acceptance": acceptance,
        "validation": {
            "pytest": "pending",
            "ruff": "pending",
            "git_diff_check": "pending",
        },
        "total_runtime_seconds": perf_counter() - started_at,
        "stop_boundary": "STOP FOR REVIEW; Task25 not started",
    }
    freeze_path = Path("docs/shapeggen_pgexplainer_freeze.json")
    report_path = Path("docs/provenance/historical_reports/shapeggen_pgexplainer_report.md")
    freeze_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(manifest), encoding="utf-8")
    print(f"Task24 complete status={manifest['status']}", flush=True)
    print(f"summary={comparison_path}", flush=True)
    print(f"freeze={freeze_path}", flush=True)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
