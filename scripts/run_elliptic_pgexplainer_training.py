from __future__ import annotations

# ruff: noqa: E402,E501
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch_geometric
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_elliptic_task26 import ROOT, build_models, load_data, logits, sha256

from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.explainers.pgexplainer_smoke import model_parameter_hash
from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    build_pgexplainer_from_config,
    materialize_pgexplainer,
    verify_pgexplainer_reload,
    wrap_pgexplainer,
)
from gnnexplainer_benchmark.utils.reproducibility import seed_everything

ALLOWED_BACKBONES = {"gcn", "graphsage"}
ALLOWED_SEEDS = {42, 0, 1, 2, 3, 4}


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    slug: str
    model_key: str
    checkpoint_key: str
    checkpoint_sha_key: str
    prediction_column: str
    confidence_column: str


@dataclass(frozen=True)
class PoolTarget:
    node_id: int
    label: int
    split: str
    prediction: int
    confidence: float
    candidate_edge_count: int
    candidate_mapping_valid: bool


@dataclass(frozen=True)
class RuntimeLedger:
    training_seconds: float
    validation_seconds: float
    serialization_seconds: float


SPECS = {
    "gcn": BackboneSpec(
        name="GCN",
        slug="gcn",
        model_key="gcn_compatible",
        checkpoint_key="gcn_checkpoint",
        checkpoint_sha_key="gcn_checkpoint_sha256",
        prediction_column="gcn_prediction",
        confidence_column="gcn_confidence",
    ),
    "graphsage": BackboneSpec(
        name="GraphSAGE",
        slug="graphsage",
        model_key="graphsage",
        checkpoint_key="graphsage_checkpoint",
        checkpoint_sha_key="graphsage_checkpoint_sha256",
        prediction_column="graphsage_prediction",
        confidence_column="graphsage_confidence",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Elliptic PGExplainer checkpoint for exactly one backbone and one seed."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/elliptic_pgexplainer_training.yaml"),
    )
    parser.add_argument("--backbone", required=True, choices=sorted(ALLOWED_BACKBONES))
    parser.add_argument("--seed", required=True, type=int, choices=sorted(ALLOWED_SEEDS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and exit without training.")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="Stop cleanly after saving the specified epoch checkpoint; intended for resume smoke tests.",
    )
    parser.add_argument(
        "--allow-debug-limits",
        action="store_true",
        help="Allow tiny train/validation limits for implementation smoke tests only.",
    )
    parser.add_argument("--debug-train-limit", type=int, default=None)
    parser.add_argument("--debug-val-limit", type=int, default=None)
    parser.add_argument(
        "--reset-incomplete",
        action="store_true",
        help="Delete only this incomplete backbone/seed run directory before training.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_text(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def read_pool(path: Path, spec: BackboneSpec) -> list[PoolTarget]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = [
        PoolTarget(
            node_id=int(row["node_id"]),
            label=int(row["label"]),
            split=row["split"],
            prediction=int(row[spec.prediction_column]),
            confidence=float(row[spec.confidence_column]),
            candidate_edge_count=int(row["candidate_edge_count"]),
            candidate_mapping_valid=str(row["candidate_mapping_valid"]).lower() == "true",
        )
        for row in rows
    ]
    ids = [target.node_id for target in targets]
    if ids != sorted(ids):
        raise RuntimeError(f"{path} must remain sorted by node_id")
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path} contains duplicate node IDs")
    return targets


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        return ""


def config_identity(config: dict[str, Any], pg_config: dict[str, Any], spec: BackboneSpec, seed: int, hashes: dict[str, str]) -> str:
    payload = {
        "task": "TASK28_ELLIPTIC_PGEXPLAINER_TRAINING",
        "backbone": spec.slug,
        "seed": seed,
        "epochs": int(config["epochs"]),
        "dataset_sha256": hashes["dataset"],
        "common_train_sha256": hashes["common_train"],
        "common_val_sha256": hashes["common_val"],
        "backbone_checkpoint_sha256": hashes[spec.checkpoint_key],
        "pgexplainer_config_hash": config_hash(pg_config),
        "selected_edge_ratio": float(config["selected_edge_ratio"]),
        "target_semantics": config["target_semantics"],
        "checkpoint_selection": config["checkpoint_selection"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def verify_frozen_inputs(config: dict[str, Any], config_path: Path) -> dict[str, str]:
    if "final_target_file" in config:
        raise RuntimeError("Training config must not include final_target_file")
    hashes = {
        "task_config": sha256(config_path),
        "dataset": assert_hash(Path(config["dataset_path"]), config["dataset_sha256"], "Elliptic dataset"),
        "common_train": assert_hash(Path(config["common_train_file"]), config["common_train_sha256"], "Common train IDs"),
        "common_val": assert_hash(Path(config["common_val_file"]), config["common_val_sha256"], "Common validation IDs"),
        "pgexplainer_config": assert_hash(Path(config["pgexplainer_config"]), config["pgexplainer_config_sha256"], "PGExplainer config"),
        "gcn_checkpoint": assert_hash(Path(config["gcn_checkpoint"]), config["gcn_checkpoint_sha256"], "GCN checkpoint"),
        "graphsage_checkpoint": assert_hash(Path(config["graphsage_checkpoint"]), config["graphsage_checkpoint_sha256"], "GraphSAGE checkpoint"),
    }
    return hashes


def validate_pools(config: dict[str, Any], train_targets: list[PoolTarget], val_targets: list[PoolTarget]) -> None:
    if len(train_targets) != int(config["common_train_count"]):
        raise RuntimeError("Common train count mismatch")
    if len(val_targets) != int(config["common_val_count"]):
        raise RuntimeError("Common validation count mismatch")
    train_ids = {target.node_id for target in train_targets}
    val_ids = {target.node_id for target in val_targets}
    if not train_ids.isdisjoint(val_ids):
        raise RuntimeError("Common train and validation pools overlap")
    if not all(target.candidate_mapping_valid for target in train_targets + val_targets):
        raise RuntimeError("Candidate mapping invalid in train/validation pools")


def validate_targets_match_predictions(targets: list[PoolTarget], frozen_predictions: torch.Tensor) -> None:
    for target in targets:
        actual = int(frozen_predictions[target.node_id].item())
        if actual != target.prediction:
            raise RuntimeError(
                f"Frozen prediction mismatch for node {target.node_id}: CSV={target.prediction}, model={actual}"
            )


def run_dir(config: dict[str, Any], spec: BackboneSpec, seed: int) -> Path:
    return Path(config["artifacts_root"]) / f"elliptic_pg_{spec.slug}_seed{seed}_run"


def final_artifact_path(config: dict[str, Any], spec: BackboneSpec, seed: int) -> Path:
    return Path(config["artifacts_root"]) / f"elliptic_pg_{spec.slug}_seed{seed}.pt"


def manifest_path(config: dict[str, Any], spec: BackboneSpec, seed: int) -> Path:
    return Path(config["artifacts_root"]) / f"elliptic_pg_{spec.slug}_seed{seed}_manifest.json"


def latest_epoch_state(directory: Path, expected_identity: str) -> tuple[int, dict[str, Any] | None]:
    states = sorted((directory / "checkpoints").glob("epoch_*.pt"))
    for path in reversed(states):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Corrupted checkpoint cannot be loaded: {path}: {exc}") from exc
        if payload.get("config_identity") != expected_identity:
            raise RuntimeError(f"Incompatible resume checkpoint: {path}")
        epoch = int(payload["epoch_completed"])
        return epoch, payload
    return 0, None


def restore_rng(payload: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(payload["rng_state_cpu"])
    np.random.set_state(payload["rng_state_numpy"])
    if device.type == "cuda" and payload.get("rng_state_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["rng_state_cuda"])


def rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None, "sum": 0.0}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def validate_pg_mask_health(
    *,
    explainer: Any,
    data: Any,
    data_cpu: Any,
    frozen_predictions: torch.Tensor,
    val_targets: list[PoolTarget],
    limit: int | None,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    start = perf_counter()
    for target in val_targets[:limit]:
        explanation = explainer(
            data.x,
            data.edge_index,
            target=frozen_predictions,
            index=target.node_id,
        )
        edge_mask = explanation.edge_mask.detach().cpu()
        candidate_mask = computation_subgraph_edge_mask(data_cpu, target.node_id, num_hops=2)
        rows.append(
            {
                "node_id": target.node_id,
                "candidate_edge_count": int(candidate_mask.sum().item()),
                "edge_mask_count": int(edge_mask.numel()),
                "edge_mask_finite": bool(torch.isfinite(edge_mask).all().item()),
                "edge_mask_std": float(edge_mask.std(unbiased=False).item()),
                "status": "success",
                "error": "",
            }
        )
    return rows, perf_counter() - start


def environment(device: torch.device) -> dict[str, Any]:
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


def reset_incomplete_run(directory: Path) -> None:
    root = (ROOT / "artifacts" / "explainers").resolve()
    target = directory.resolve()
    if root not in target.parents:
        raise RuntimeError(f"Refusing to reset path outside artifacts/explainers: {target}")
    complete_marker = target / "COMPLETE.json"
    if complete_marker.exists():
        raise RuntimeError("Refusing to reset a completed run")
    if target.exists():
        shutil.rmtree(target)


def write_epoch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "mean_loss",
        "median_loss",
        "min_loss",
        "max_loss",
        "num_train_instances",
        "runtime_seconds",
        "serialization_seconds",
        "seed",
        "backbone",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.debug_train_limit is not None and not args.allow_debug_limits:
        raise RuntimeError("--debug-train-limit requires --allow-debug-limits")
    if args.debug_val_limit is not None and not args.allow_debug_limits:
        raise RuntimeError("--debug-val-limit requires --allow-debug-limits")
    full_config = load_yaml(args.config)
    config = full_config["task28_training"]
    pg_config = load_yaml(Path(config["pgexplainer_config"]))
    pg_config = json.loads(json.dumps(pg_config))
    pg_config["explainer"]["seed"] = int(args.seed)
    pg_config["explainer"]["epochs"] = int(config["epochs"])
    spec = SPECS[args.backbone]
    hashes = verify_frozen_inputs(config, args.config)
    train_targets = read_pool(Path(config["common_train_file"]), spec)
    val_targets = read_pool(Path(config["common_val_file"]), spec)
    validate_pools(config, train_targets, val_targets)
    if args.allow_debug_limits:
        train_targets = train_targets[: args.debug_train_limit] if args.debug_train_limit else train_targets
        val_targets = val_targets[: args.debug_val_limit] if args.debug_val_limit else val_targets
    device = resolve_device(args.device)
    identity = config_identity(config, pg_config, spec, int(args.seed), hashes)
    directory = run_dir(config, spec, int(args.seed))
    complete_marker = directory / "COMPLETE.json"
    if complete_marker.exists():
        complete = json.loads(complete_marker.read_text(encoding="utf-8"))
        if complete.get("config_identity") == identity:
            return {"status": "already_complete", "complete_marker": str(complete_marker), "config_identity": identity}
        raise RuntimeError("Completed run exists with incompatible config identity")
    if args.reset_incomplete:
        reset_incomplete_run(directory)
    if args.dry_run:
        return {
            "status": "dry_run_pass",
            "backbone": spec.name,
            "seed": int(args.seed),
            "train_count": len(train_targets),
            "validation_count": len(val_targets),
            "config_identity": identity,
            "hashes": hashes,
            "device": str(device),
        }

    seed_everything(int(args.seed))
    data_cpu = load_data()
    data = data_cpu.clone().to(device)
    model = build_models(data_cpu)[spec.model_key].to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_hash_before = model_parameter_hash(model)
    with torch.no_grad():
        frozen_predictions = logits(model, data).argmax(dim=1).detach()
    validate_targets_match_predictions(train_targets + val_targets, frozen_predictions.cpu())

    algorithm = build_pgexplainer_from_config(pg_config).to(device)
    explainer = wrap_pgexplainer(model, algorithm)
    materialize_pgexplainer(algorithm, model, data.x, data.edge_index, train_targets[0].node_id)
    start_epoch, resume_payload = latest_epoch_state(directory, identity)
    epoch_rows = []
    runtime_training_total = 0.0
    runtime_validation_total = 0.0
    runtime_serialization_total = 0.0
    if resume_payload is not None:
        algorithm.load_state_dict(resume_payload["algorithm_state_dict"])
        algorithm.optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        algorithm._curr_epoch = int(resume_payload.get("algorithm_current_epoch", start_epoch - 1))
        restore_rng(resume_payload, device)
        epoch_rows = list(resume_payload.get("epoch_rows", []))
        ledger = resume_payload.get("runtime_ledger", {})
        runtime_training_total = float(ledger.get("training_seconds", 0.0))
        runtime_validation_total = float(ledger.get("validation_seconds", 0.0))
        runtime_serialization_total = float(ledger.get("serialization_seconds", 0.0))

    jsonl_log = Path(config["metrics_root"]) / spec.slug / "pgexplainer" / f"training_seed{args.seed}.jsonl"
    text_log = Path(config["metrics_root"]) / spec.slug / "pgexplainer" / f"training_seed{args.seed}.txt"
    total_epochs = int(config["epochs"])
    for epoch in range(start_epoch, total_epochs):
        losses = []
        train_start = perf_counter()
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
                raise RuntimeError(f"Non-finite PGExplainer loss at epoch {epoch + 1}: {loss}")
            losses.append(float(loss))
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_training_seconds = perf_counter() - train_start
        runtime_training_total += epoch_training_seconds
        row = {
            "epoch": epoch + 1,
            "mean_loss": mean(losses),
            "median_loss": median(losses),
            "min_loss": min(losses),
            "max_loss": max(losses),
            "num_train_instances": len(losses),
            "runtime_seconds": epoch_training_seconds,
            "serialization_seconds": 0.0,
            "seed": int(args.seed),
            "backbone": spec.name,
        }
        epoch_rows.append(row)
        serialization_start = perf_counter()
        state_payload = {
            "task": "TASK28 Elliptic PGExplainer training state",
            "config_identity": identity,
            "epoch_completed": epoch + 1,
            "algorithm_state_dict": algorithm.state_dict(),
            "optimizer_state_dict": algorithm.optimizer.state_dict(),
            "algorithm_current_epoch": algorithm._curr_epoch,
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "epoch_rows": epoch_rows,
            "runtime_ledger": {
                "training_seconds": runtime_training_total,
                "validation_seconds": runtime_validation_total,
                "serialization_seconds": runtime_serialization_total,
            },
            "hashes": hashes,
            "backbone": spec.name,
            "seed": int(args.seed),
        }
        atomic_torch_save(directory / "checkpoints" / f"epoch_{epoch + 1:03d}.pt", state_payload)
        serialization_seconds = perf_counter() - serialization_start
        runtime_serialization_total += serialization_seconds
        row["serialization_seconds"] = serialization_seconds
        state_payload["epoch_rows"] = epoch_rows
        state_payload["runtime_ledger"] = {
            "training_seconds": runtime_training_total,
            "validation_seconds": runtime_validation_total,
            "serialization_seconds": runtime_serialization_total,
        }
        atomic_torch_save(directory / "checkpoints" / f"epoch_{epoch + 1:03d}.pt", state_payload)
        append_jsonl(jsonl_log, {"event": "epoch_complete", **row, "config_identity": identity})
        append_text(
            text_log,
            f"{spec.name} seed {args.seed} epoch {epoch + 1}/{total_epochs} loss={row['mean_loss']:.6f} train_s={epoch_training_seconds:.3f}",
        )
        write_epoch_csv(Path(config["metrics_root"]) / spec.slug / "pgexplainer" / f"training_seed{args.seed}.csv", epoch_rows)
        if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
            return {"status": "stopped_after_epoch", "epoch_completed": epoch + 1, "config_identity": identity}

    validation_rows, validation_seconds = validate_pg_mask_health(
        explainer=explainer,
        data=data,
        data_cpu=data_cpu,
        frozen_predictions=frozen_predictions,
        val_targets=val_targets,
        limit=None,
    )
    runtime_validation_total += validation_seconds
    if any(row["status"] != "success" or not row["edge_mask_finite"] for row in validation_rows):
        raise RuntimeError("Validation health failed")
    if model_hash_before != model_parameter_hash(model):
        raise RuntimeError(f"{spec.name} backbone weights changed during training")
    artifact = final_artifact_path(config, spec, int(args.seed))
    atomic_torch_save(
        artifact,
        {
            "task": "TASK28 Elliptic PGExplainer",
            "algorithm_state_dict": algorithm.state_dict(),
            "optimizer_state_dict": algorithm.optimizer.state_dict(),
            "algorithm_current_epoch": algorithm._curr_epoch,
            "explainer_config": pg_config,
            "training_config": config,
            "config_identity": identity,
            "seed": int(args.seed),
            "backbone": spec.name,
            "train_node_ids": [target.node_id for target in train_targets],
            "validation_node_ids": [target.node_id for target in val_targets],
            "hashes": hashes,
        },
    )
    reload_ok = verify_pgexplainer_reload(
        artifact,
        pg_config,
        model,
        data.x,
        data.edge_index,
        train_targets[0].node_id,
    )
    if not reload_ok:
        raise RuntimeError("PGExplainer checkpoint reload check failed")
    manifest = {
        "task": "TASK28 Elliptic PGExplainer training",
        "status": "success",
        "backbone": spec.name,
        "seed": int(args.seed),
        "epochs_completed": total_epochs,
        "checkpoint_selection": config["checkpoint_selection"],
        "artifact_path": str(artifact),
        "artifact_sha256": sha256(artifact),
        "manifest_path": str(manifest_path(config, spec, int(args.seed))),
        "state_run_dir": str(directory),
        "config_identity": identity,
        "hashes": hashes,
        "train_count": len(train_targets),
        "validation_count": len(val_targets),
        "target_semantics": config["target_semantics"],
        "model_hash_before": model_hash_before,
        "model_hash_after": model_parameter_hash(model),
        "model_weights_unchanged": True,
        "checkpoint_reload_ok": reload_ok,
        "optimizer_isolation": "classifier parameters require_grad=False; optimizer owns PGExplainer parameters only",
        "training_runtime_seconds": runtime_training_total,
        "validation_runtime_seconds": runtime_validation_total,
        "serialization_runtime_seconds": runtime_serialization_total,
        "epoch_loss_summary": numeric_summary([float(row["mean_loss"]) for row in epoch_rows]),
        "validation_health": {
            "rows": len(validation_rows),
            "success_count": len(validation_rows),
            "failure_count": 0,
            "edge_mask_std_summary": numeric_summary([float(row["edge_mask_std"]) for row in validation_rows]),
        },
        "environment": environment(device),
        "git": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
        },
        "post_test_tuning": False,
        "final_targets_accessed": False,
    }
    atomic_write_json(manifest_path(config, spec, int(args.seed)), manifest)
    atomic_write_json(complete_marker, {"status": "complete", "config_identity": identity, "artifact_sha256": manifest["artifact_sha256"]})
    append_jsonl(jsonl_log, {"event": "complete", "config_identity": identity, "artifact_sha256": manifest["artifact_sha256"]})
    return manifest


def main() -> None:
    args = parse_args()
    result = train_one(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
