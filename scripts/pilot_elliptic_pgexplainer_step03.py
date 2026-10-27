from __future__ import annotations

# ruff: noqa: E402,E501,I001
import hashlib
import json
import platform
import subprocess
import sys
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
from scripts.train.run_elliptic_pgexplainer_training import (
    SPECS,
    atomic_torch_save,
    config_hash,
    config_identity,
    latest_epoch_state,
    read_pool,
    restore_rng,
    validate_targets_match_predictions,
    verify_frozen_inputs,
)

from gnnexplainer_benchmark.explainers.pgexplainer_smoke import model_parameter_hash
from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    build_pgexplainer_from_config,
    materialize_pgexplainer,
    verify_pgexplainer_reload,
    wrap_pgexplainer,
)


CONFIG_PATH = Path("configs/elliptic/elliptic_pgexplainer_training.yaml")
MANIFEST_PATH = Path("docs/elliptic_pgexplainer_step03_pilot_manifest.json")
REPORT_PATH = Path("docs/provenance/historical_reports/elliptic_pgexplainer_step03_pilot_report.md")
PILOT_ROOT = Path("results/metrics/elliptic/task28_pilot")


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        return ""


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def trainable_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return before.keys() != after.keys() or any(
        not torch.equal(before[name], after[name]) for name in before
    )


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def warmup_and_measure(
    *,
    algorithm: Any,
    model: torch.nn.Module,
    data: Any,
    frozen_predictions: torch.Tensor,
    train_node_ids: list[int],
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    losses: list[float] = []
    runtimes: list[float] = []
    sequence = train_node_ids * ((warmup_steps + measured_steps) // len(train_node_ids) + 1)
    for step, node_id in enumerate(sequence[: warmup_steps + measured_steps]):
        start = perf_counter()
        loss = algorithm.train(
            step,
            model,
            data.x,
            data.edge_index,
            target=frozen_predictions,
            index=node_id,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtime = perf_counter() - start
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite PGExplainer pilot loss: {loss}")
        if step >= warmup_steps:
            losses.append(float(loss))
            runtimes.append(runtime)
    return losses, runtimes


def reload_edge_mask_digest(
    *,
    checkpoint_path: Path,
    pg_config: dict[str, Any],
    model: torch.nn.Module,
    data: Any,
    node_id: int,
    frozen_predictions: torch.Tensor,
) -> tuple[str, str]:
    payload = torch.load(checkpoint_path, map_location=data.x.device, weights_only=False)
    fresh = build_pgexplainer_from_config(pg_config).to(data.x.device)
    explainer = wrap_pgexplainer(model, fresh)
    materialize_pgexplainer(fresh, model, data.x, data.edge_index, node_id)
    fresh.load_state_dict(payload["algorithm_state_dict"])
    fresh._curr_epoch = int(payload["algorithm_current_epoch"])
    original = build_pgexplainer_from_config(pg_config).to(data.x.device)
    wrap_pgexplainer(model, original)
    materialize_pgexplainer(original, model, data.x, data.edge_index, node_id)
    original.load_state_dict(payload["algorithm_state_dict"])
    original._curr_epoch = int(payload["algorithm_current_epoch"])
    with torch.no_grad():
        first = explainer(data.x, data.edge_index, target=frozen_predictions, index=node_id).edge_mask.detach().cpu()
        second = explainer(data.x, data.edge_index, target=frozen_predictions, index=node_id).edge_mask.detach().cpu()
    return (
        hashlib.sha256(first.contiguous().numpy().tobytes()).hexdigest(),
        hashlib.sha256(second.contiguous().numpy().tobytes()).hexdigest(),
    )


def run_resume_fixture() -> dict[str, Any]:
    seed_everything(777)
    x = torch.tensor([[1.0, -1.0]])
    y = torch.tensor([[0.25]])

    def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        optimizer.step()

    uninterrupted = torch.nn.Linear(2, 1)
    uninterrupted_optimizer = torch.optim.SGD(uninterrupted.parameters(), lr=0.1)
    step(uninterrupted, uninterrupted_optimizer)
    step(uninterrupted, uninterrupted_optimizer)

    seed_everything(777)
    resumed = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.1)
    step(resumed, resumed_optimizer)
    fixture_dir = PILOT_ROOT / "resume_fixture"
    atomic_torch_save(
        fixture_dir / "checkpoints" / "epoch_001.pt",
        {
            "config_identity": "pilot-fixture",
            "epoch_completed": 1,
            "model_state_dict": resumed.state_dict(),
            "optimizer_state_dict": resumed_optimizer.state_dict(),
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
        },
    )
    epoch, payload = latest_epoch_state(fixture_dir, "pilot-fixture")
    assert payload is not None
    resumed.load_state_dict(payload["model_state_dict"])
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_rng(payload, torch.device("cpu"))
    step(resumed, resumed_optimizer)
    parity = all(
        torch.equal(first, second)
        for first, second in zip(uninterrupted.parameters(), resumed.parameters(), strict=True)
    )
    return {"epoch_resumed_from": epoch, "uninterrupted_resumed_parity": parity}


def run_backbone_pilot(
    *,
    spec_key: str,
    config: dict[str, Any],
    pg_config: dict[str, Any],
    hashes: dict[str, str],
    data_cpu: Any,
    device: torch.device,
) -> dict[str, Any]:
    spec = SPECS[spec_key]
    train_targets = read_pool(Path(config["common_train_file"]), spec)[:8]
    val_targets = read_pool(Path(config["common_val_file"]), spec)[:3]
    seed = 42
    seed_everything(seed)
    pg_config = json.loads(json.dumps(pg_config))
    pg_config["explainer"]["seed"] = seed
    pg_config["explainer"]["epochs"] = int(config["epochs"])
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
    optimizer_parameter_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    backbone_parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_isolated = not (optimizer_parameter_ids & backbone_parameter_ids)
    pg_before = trainable_snapshot(algorithm)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses, runtimes = warmup_and_measure(
        algorithm=algorithm,
        model=model,
        data=data,
        frozen_predictions=frozen_predictions,
        train_node_ids=[target.node_id for target in train_targets],
        warmup_steps=2,
        measured_steps=10,
        device=device,
    )
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        peak_memory = None
    pg_after = trainable_snapshot(algorithm)
    validation_start = perf_counter()
    val_rows = []
    algorithm._curr_epoch = int(config["epochs"]) - 1
    for target in val_targets:
        explanation = explainer(data.x, data.edge_index, target=frozen_predictions, index=target.node_id)
        edge_mask = explanation.edge_mask.detach().cpu()
        val_rows.append(
            {
                "node_id": target.node_id,
                "edge_mask_count": int(edge_mask.numel()),
                "edge_mask_finite": bool(torch.isfinite(edge_mask).all().item()),
                "edge_mask_std": float(edge_mask.std(unbiased=False).item()),
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    validation_seconds = perf_counter() - validation_start
    pilot_checkpoint = PILOT_ROOT / "checkpoints" / f"pilot_{spec.slug}_seed42.pt"
    atomic_torch_save(
        pilot_checkpoint,
        {
            "task": "Task28 Step03 pilot checkpoint",
            "config_identity": config_identity(
                config,
                pg_config,
                spec,
                seed,
                hashes,
                {"allow_debug_limits": True, "train_count": 8, "validation_count": 3, "pilot": True},
            ),
            "algorithm_state_dict": algorithm.state_dict(),
            "optimizer_state_dict": algorithm.optimizer.state_dict(),
            "algorithm_current_epoch": algorithm._curr_epoch,
        },
    )
    reload_ok = verify_pgexplainer_reload(
        pilot_checkpoint,
        pg_config,
        model,
        data.x,
        data.edge_index,
        train_targets[0].node_id,
    )
    if model_hash_before != model_parameter_hash(model):
        raise RuntimeError(f"{spec.name} backbone mutated during pilot")
    return {
        "backbone": spec.name,
        "pilot_train_nodes": [target.node_id for target in train_targets],
        "pilot_val_nodes": [target.node_id for target in val_targets],
        "target_matches_frozen_prediction": True,
        "optimizer_isolation": optimizer_isolated,
        "pgexplainer_parameters_changed": changed(pg_before, pg_after),
        "pgexplainer_parameter_digest_before": tensor_digest(pg_before),
        "pgexplainer_parameter_digest_after": tensor_digest(pg_after),
        "backbone_hash_before": model_hash_before,
        "backbone_hash_after": model_parameter_hash(model),
        "backbone_weights_unchanged": True,
        "loss_summary": numeric_summary(losses),
        "train_step_runtime_seconds": numeric_summary(runtimes),
        "estimated_30_epoch_seconds_for_2000_train_nodes": mean(runtimes) * 2000 * 30,
        "peak_gpu_memory_bytes": peak_memory,
        "validation_runtime_seconds": validation_seconds,
        "validation_rows": val_rows,
        "pilot_checkpoint": str(pilot_checkpoint),
        "pilot_checkpoint_sha256": sha256(pilot_checkpoint),
        "reload_ok": reload_ok,
        "edge_mask_finite": all(row["edge_mask_finite"] for row in val_rows),
        "edge_mask_non_degenerate": all(row["edge_mask_std"] > 0 for row in val_rows),
    }


def build_report(manifest: dict[str, Any]) -> str:
    rows = "\n".join(
        "| {backbone} | {n} | {mean:.6f} | {std:.6f} | {median:.6f} | {min:.6f} | {max:.6f} | {estimate:.2f} | {memory} |".format(
            backbone=item["backbone"],
            n=item["train_step_runtime_seconds"]["n"],
            mean=item["train_step_runtime_seconds"]["mean"],
            std=item["train_step_runtime_seconds"]["std"],
            median=item["train_step_runtime_seconds"]["median"],
            min=item["train_step_runtime_seconds"]["min"],
            max=item["train_step_runtime_seconds"]["max"],
            estimate=item["estimated_30_epoch_seconds_for_2000_train_nodes"],
            memory=item["peak_gpu_memory_bytes"],
        )
        for item in manifest["backbones"].values()
    )
    return f"""# Elliptic PGExplainer Step03 Pilot And Parity

## Scope

This is a train/validation-only pilot for Task28. It uses IDs from the frozen common train and validation pools only. It does not access final targets, does not start full training, and does not produce Task28 final PGExplainer results.

## Timing

| Backbone | n | mean | std | median | min | max | estimated 30 epoch seconds | peak GPU memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Integrity

- Target semantics: frozen backbone original predicted class.
- PGExplainer parameters changed: `{manifest['acceptance']['pgexplainer_parameters_changed']}`
- Backbone weights unchanged: `{manifest['acceptance']['backbone_weights_unchanged']}`
- Optimizer isolation: `{manifest['acceptance']['optimizer_isolation']}`
- Reload parity: `{manifest['acceptance']['reload_parity']}`
- Interrupted/resumed fixture parity: `{manifest['resume_fixture']['uninterrupted_resumed_parity']}`
- Final targets accessed: `{manifest['final_targets_accessed']}`

## Recommendation

`{manifest['recommendation']}`

## Stop Boundary

STOP FOR REVIEW. Step04 is the next gate and full Task28 training is not started here.
"""


def main() -> None:
    start = perf_counter()
    config = load_yaml(CONFIG_PATH)["task28_training"]
    pg_config = load_yaml(Path(config["pgexplainer_config"]))
    hashes = verify_frozen_inputs(config, CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cpu = load_data()
    backbones = {
        key: run_backbone_pilot(
            spec_key=key,
            config=config,
            pg_config=pg_config,
            hashes=hashes,
            data_cpu=data_cpu,
            device=device,
        )
        for key in ("gcn", "graphsage")
    }
    resume_fixture = run_resume_fixture()
    acceptance = {
        "pgexplainer_parameters_changed": all(item["pgexplainer_parameters_changed"] for item in backbones.values()),
        "backbone_weights_unchanged": all(item["backbone_weights_unchanged"] for item in backbones.values()),
        "optimizer_isolation": all(item["optimizer_isolation"] for item in backbones.values()),
        "target_matches_frozen_prediction": all(item["target_matches_frozen_prediction"] for item in backbones.values()),
        "reload_parity": all(item["reload_ok"] for item in backbones.values()),
        "edge_masks_finite": all(item["edge_mask_finite"] for item in backbones.values()),
        "edge_masks_non_degenerate": all(item["edge_mask_non_degenerate"] for item in backbones.values()),
        "resume_fixture_parity": resume_fixture["uninterrupted_resumed_parity"],
        "final_targets_not_accessed": True,
    }
    manifest = {
        "task": "Task28 Step03 PGExplainer pilot and parity",
        "status": "PASS" if all(acceptance.values()) else "BLOCKED",
        "git": {"branch": git_value("branch", "--show-current"), "commit": git_value("rev-parse", "HEAD")},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "input_hashes": hashes,
        "task_config_hash": sha256(CONFIG_PATH),
        "pg_config_hash": config_hash(pg_config),
        "pilot_scope": {"train_nodes_per_backbone": 8, "validation_nodes_per_backbone": 3, "warmup_steps": 2, "measured_train_steps": 10},
        "backbones": backbones,
        "resume_fixture": resume_fixture,
        "acceptance": acceptance,
        "final_targets_accessed": False,
        "full_training_started": False,
        "recommendation": "freeze implementation for Step04 compute plan" if all(acceptance.values()) else "block before Step04",
        "runtime_seconds": perf_counter() - start,
        "stop_boundary": "STOP FOR REVIEW; Step04 not started",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(build_report(manifest), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(MANIFEST_PATH), "report": str(REPORT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
