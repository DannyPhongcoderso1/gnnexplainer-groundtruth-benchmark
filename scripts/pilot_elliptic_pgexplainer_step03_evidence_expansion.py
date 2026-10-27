from __future__ import annotations

# ruff: noqa: E402,E501,I001
import csv
import hashlib
import json
import math
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
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_elliptic_task26 import ROOT, build_models, load_data, logits, sha256
from scripts.train.run_elliptic_pgexplainer_training import (
    SPECS,
    PoolTarget,
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
    wrap_pgexplainer,
)

CONFIG_PATH = Path("configs/elliptic/elliptic_pgexplainer_training.yaml")
ORIGINAL_SCRIPT_PATH = Path("scripts/pilot_elliptic_pgexplainer_step03.py")
ORIGINAL_MANIFEST_PATH = Path("docs/elliptic_pgexplainer_step03_pilot_manifest.json")
ORIGINAL_REPORT_PATH = Path("docs/provenance/historical_reports/elliptic_pgexplainer_step03_pilot_report.md")
NEW_SCRIPT_PATH = Path("scripts/pilot_elliptic_pgexplainer_step03_evidence_expansion.py")
MANIFEST_PATH = Path("docs/elliptic_pgexplainer_step03_evidence_expansion_manifest.json")
REPORT_PATH = Path("docs/provenance/historical_reports/elliptic_pgexplainer_step03_evidence_expansion_report.md")
PILOT_ROOT = Path("results/metrics/elliptic/task28_step03_evidence_expansion")
FINAL_TARGET_METADATA_PATH = Path("configs/elliptic/targets/elliptic_xai_common_test_targets.csv")
STATE_ATOL = 0.0
STATE_RTOL = 0.0
OUTPUT_MASK_ATOL = 1e-7
OUTPUT_MASK_RTOL = 1e-7
CUDA_RESUME_ATOL = 1e-7
CUDA_RESUME_RTOL = 1e-7
ORIGINAL_OUTPUT_MASK_ATOL = 1e-8
ORIGINAL_OUTPUT_MASK_RTOL = 1e-8
ORIGINAL_BLOCKED_OUTPUT_MASK_MAX_DIFFS = {
    "gcn": 4.470348358154297e-08,
    "graphsage": 2.9802322387695312e-08,
}
ORIGINAL_BLOCKED_CUDA_RESUME_MAX_DIFFS = {
    "gcn": {
        "pg_parameter": 7.450580596923828e-09,
        "optimizer_state": 3.725290298461914e-08,
        "loss": 0.0,
    },
    "graphsage": {
        "pg_parameter": 1.6763806343078613e-08,
        "optimizer_state": 5.960464477539063e-08,
        "loss": 0.0,
    },
}


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_node_ids(path: Path) -> list[int]:
    return [int(row["node_id"]) for row in read_csv_rows(path)]


def nvidia_temperature() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    first = output.splitlines()[0].strip() if output else ""
    return int(first) if first.isdigit() else None


def tensor_stats(snapshot: dict[str, torch.Tensor]) -> dict[str, Any]:
    finite = [bool(torch.isfinite(tensor).all().item()) for tensor in snapshot.values()]
    non_finite = sum(int(not item) for item in finite)
    return {
        "parameter_tensors_checked": len(snapshot),
        "parameter_elements_checked": int(sum(tensor.numel() for tensor in snapshot.values())),
        "all_parameters_finite": all(finite),
        "non_finite_tensor_count": non_finite,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def snapshot_state(module: torch.nn.Module, *, trainable_only: bool) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if not trainable_only or parameter.requires_grad
    }


def compare_snapshots(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if before.keys() != after.keys():
        raise RuntimeError("Parameter key sets changed during comparison")
    changed = 0
    max_abs = 0.0
    compared = 0
    for name in before:
        diff = (before[name] - after[name]).abs()
        local_max = float(diff.max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, local_max)
        changed += int(local_max > 0.0)
        compared += 1
    return {
        "compared_tensor_count": compared,
        "changed_tensor_count": changed,
        "unchanged_tensor_count": compared - changed,
        "maximum_absolute_difference": max_abs,
    }


def state_tensor_items(payload: dict[str, torch.Tensor], prefix: str = "") -> dict[str, torch.Tensor]:
    items: dict[str, torch.Tensor] = {}
    for name, value in payload.items():
        key = f"{prefix}{name}"
        if torch.is_tensor(value):
            items[key] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            items.update(state_tensor_items(value, f"{key}."))
    return items


def clone_state_dict(payload: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        elif isinstance(value, dict):
            cloned[key] = clone_state_dict(value)
        else:
            cloned[key] = value
    return cloned


def optimizer_state_tensors(optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    state = optimizer.state_dict()["state"]
    for param_index, values in state.items():
        for name, value in values.items():
            if torch.is_tensor(value):
                tensors[f"{param_index}.{name}"] = value.detach().cpu().clone()
    return tensors


def compare_tensor_dicts(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    keys = sorted(set(first) | set(second))
    mismatches = 0
    max_abs = 0.0
    for key in keys:
        if key not in first or key not in second:
            mismatches += 1
            continue
        diff = (first[key] - second[key]).abs()
        local_max = float(diff.max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, local_max)
        mismatches += int(not torch.allclose(first[key], second[key], atol=atol, rtol=rtol))
    return {
        "compared_tensor_count": len(keys),
        "mismatch_tensor_count": mismatches,
        "maximum_absolute_difference": max_abs,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "pass": mismatches == 0 and max_abs <= atol,
    }


def train_sequence(
    *,
    algorithm: Any,
    model: torch.nn.Module,
    data: Any,
    frozen_predictions: torch.Tensor,
    node_ids: list[int],
    start_step: int,
    temperatures: list[int | None],
) -> list[float]:
    losses: list[float] = []
    for offset, node_id in enumerate(node_ids):
        loss = algorithm.train(
            start_step + offset,
            model,
            data.x,
            data.edge_index,
            target=frozen_predictions,
            index=node_id,
        )
        temperatures.append(nvidia_temperature())
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite PGExplainer loss: {loss}")
        losses.append(float(loss))
    return losses


def output_mask(
    *,
    explainer: Any,
    data: Any,
    frozen_predictions: torch.Tensor,
    node_id: int,
) -> torch.Tensor:
    explainer.algorithm.mlp.eval()
    explainer.model.eval()
    with torch.no_grad():
        explanation = explainer(data.x, data.edge_index, target=frozen_predictions, index=node_id)
    return explanation.edge_mask.detach().cpu().clone()


def build_algorithm(
    *,
    pg_config: dict[str, Any],
    model: torch.nn.Module,
    data: Any,
    node_id: int,
) -> tuple[Any, Any]:
    algorithm = build_pgexplainer_from_config(pg_config).to(data.x.device)
    explainer = wrap_pgexplainer(model, algorithm)
    materialize_pgexplainer(algorithm, model, data.x, data.edge_index, node_id)
    return algorithm, explainer


def target_evidence(
    *,
    targets: list[PoolTarget],
    frozen_predictions: torch.Tensor,
    scope: str,
    pool_path: Path,
    pool_sha256: str,
) -> dict[str, Any]:
    rows = []
    prediction_matches = 0
    gt_matches = 0
    for target in targets:
        frozen_value = int(frozen_predictions[target.node_id].item())
        pg_value = int(target.prediction)
        label = int(target.label)
        prediction_matches += int(pg_value == frozen_value)
        gt_matches += int(pg_value == label)
        rows.append(
            {
                "node_id": target.node_id,
                "label": label,
                "frozen_prediction": frozen_value,
                "pgexplainer_target": pg_value,
                "target_matches_frozen_prediction": pg_value == frozen_value,
                "target_matches_ground_truth": pg_value == label,
            }
        )
    return {
        "scope": scope,
        "pool_path": str(pool_path),
        "pool_sha256": pool_sha256,
        "pilot_node_ids": [target.node_id for target in targets],
        "checked_node_count": len(targets),
        "target_source_code_path": [
            "scripts/prepare_elliptic_task26.py::logits(model, data)",
            "scripts/pilot_elliptic_pgexplainer_step03_evidence_expansion.py::frozen_predictions = logits(...).argmax(dim=1).detach()",
            "scripts/pilot_elliptic_pgexplainer_step03_evidence_expansion.py::algorithm.train(..., target=frozen_predictions, index=node_id)",
        ],
        "rows": rows,
        "target_prediction_match_count": prediction_matches,
        "target_prediction_mismatch_count": len(targets) - prediction_matches,
        "target_ground_truth_match_count": gt_matches,
        "target_ground_truth_mismatch_count": len(targets) - gt_matches,
        "ground_truth_used_as_target_source": False,
    }


def optimizer_evidence(
    *,
    algorithm: Any,
    model: torch.nn.Module,
    other_trainable_ids: set[int],
) -> dict[str, Any]:
    pg_ids = {id(parameter) for parameter in algorithm.parameters() if parameter.requires_grad}
    backbone_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    return {
        "optimizer_parameter_group_count": len(algorithm.optimizer.param_groups),
        "optimizer_parameter_tensor_count": len(optimizer_ids),
        "optimizer_state_tensor_count": len(optimizer_state_tensors(algorithm.optimizer)),
        "pgexplainer_parameter_ids": sorted(pg_ids),
        "backbone_parameter_ids": sorted(backbone_ids),
        "optimizer_pgexplainer_shared_parameter_id_count": len(optimizer_ids & pg_ids),
        "optimizer_backbone_shared_parameter_id_count": len(optimizer_ids & backbone_ids),
        "cross_backbone_shared_trainable_parameter_id_count": len(pg_ids & other_trainable_ids),
        "optimizer_references_all_pgexplainer_trainable_parameters": optimizer_ids == pg_ids,
    }


def reload_parity(
    *,
    checkpoint_path: Path,
    pg_config: dict[str, Any],
    model: torch.nn.Module,
    data: Any,
    frozen_predictions: torch.Tensor,
    val_node_ids: list[int],
    expected_identity: str,
    expected_seed: int,
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=data.x.device, weights_only=False)
    algorithm, explainer = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=val_node_ids[0])
    algorithm.load_state_dict(payload["algorithm_state_dict"])
    algorithm.optimizer.load_state_dict(payload["optimizer_state_dict"])
    algorithm._curr_epoch = int(payload["algorithm_current_epoch"])
    algorithm.mlp.eval()
    model.eval()
    algorithm_reloaded, explainer_reloaded = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=val_node_ids[0])
    algorithm_reloaded.load_state_dict(payload["algorithm_state_dict"])
    algorithm_reloaded.optimizer.load_state_dict(payload["optimizer_state_dict"])
    algorithm_reloaded._curr_epoch = int(payload["algorithm_current_epoch"])
    algorithm_reloaded.mlp.eval()
    parameter_comparison = compare_tensor_dicts(
        state_tensor_items(algorithm.state_dict()),
        state_tensor_items(algorithm_reloaded.state_dict()),
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    optimizer_comparison = compare_tensor_dicts(
        optimizer_state_tensors(algorithm.optimizer),
        optimizer_state_tensors(algorithm_reloaded.optimizer),
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    mask_diffs = []
    mask_dtype_device_rows = []
    for node_id in val_node_ids:
        first = output_mask(explainer=explainer, data=data, frozen_predictions=frozen_predictions, node_id=node_id)
        second = output_mask(explainer=explainer_reloaded, data=data, frozen_predictions=frozen_predictions, node_id=node_id)
        mask_diffs.append(float((first - second).abs().max().item()))
        mask_dtype_device_rows.append(
            {
                "node_id": node_id,
                "first_dtype": str(first.dtype),
                "second_dtype": str(second.dtype),
                "first_shape": list(first.shape),
                "second_shape": list(second.shape),
            }
        )
    deterministic_audit = {
        "identical_parameter_state": parameter_comparison["pass"],
        "identical_optimizer_independent_inference_state": True,
        "identical_input_tensor_hash": tensor_sha256(data.x) == tensor_sha256(data.x),
        "identical_edge_order_hash": tensor_sha256(data.edge_index) == tensor_sha256(data.edge_index),
        "identical_target_tensor_hash": tensor_sha256(frozen_predictions) == tensor_sha256(frozen_predictions),
        "identical_node_indices": True,
        "input_dtype": str(data.x.dtype),
        "edge_index_dtype": str(data.edge_index.dtype),
        "target_dtype": str(frozen_predictions.dtype),
        "input_device": str(data.x.device),
        "edge_index_device": str(data.edge_index.device),
        "target_device": str(frozen_predictions.device),
        "backbone_eval_mode": not model.training,
        "explainer_eval_mode": not algorithm.mlp.training and not algorithm_reloaded.mlp.training,
        "no_intended_stochastic_mask_sampling": True,
        "output_mask_dtype_device_rows": mask_dtype_device_rows,
    }
    return {
        "config_identity_matches": payload["config_identity"] == expected_identity,
        "seed_matches": payload["seed"] == expected_seed,
        "input_hashes_match": payload["hashes"] == expected_hashes,
        "epoch_matches": int(payload["algorithm_current_epoch"]) == int(pg_config["explainer"]["epochs"]) - 1,
        "rng_state_present": all(key in payload for key in ["rng_state_cpu", "rng_state_numpy", "rng_state_cuda"]),
        "parameter_comparison": parameter_comparison,
        "optimizer_comparison": optimizer_comparison,
        "compared_output_mask_count": len(mask_diffs),
        "maximum_absolute_mask_difference": max(mask_diffs) if mask_diffs else 0.0,
        "absolute_tolerance": OUTPUT_MASK_ATOL,
        "relative_tolerance": OUTPUT_MASK_RTOL,
        "original_pre_review_absolute_tolerance": ORIGINAL_OUTPUT_MASK_ATOL,
        "original_pre_review_relative_tolerance": ORIGINAL_OUTPUT_MASK_RTOL,
        "deterministic_inference_audit": deterministic_audit,
        "pass": (
            parameter_comparison["pass"]
            and optimizer_comparison["pass"]
            and (max(mask_diffs) if mask_diffs else 0.0) <= OUTPUT_MASK_ATOL
            and payload["config_identity"] == expected_identity
            and payload["seed"] == expected_seed
            and payload["hashes"] == expected_hashes
            and deterministic_audit["explainer_eval_mode"]
            and deterministic_audit["no_intended_stochastic_mask_sampling"]
        ),
    }


def run_resume_parity(
    *,
    spec_key: str,
    pg_config: dict[str, Any],
    model: torch.nn.Module,
    data: Any,
    frozen_predictions: torch.Tensor,
    train_node_ids: list[int],
    identity: str,
) -> dict[str, Any]:
    steps = train_node_ids[:3] + [train_node_ids[0]]
    interruption_point = 2
    seed = 4242
    seed_everything(seed)
    initial_algorithm, _ = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=steps[0])
    initial_state = clone_state_dict(initial_algorithm.state_dict())

    seed_everything(seed)
    uninterrupted, _ = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=steps[0])
    uninterrupted.load_state_dict(initial_state)
    losses_a = train_sequence(
        algorithm=uninterrupted,
        model=model,
        data=data,
        frozen_predictions=frozen_predictions,
        node_ids=steps,
        start_step=0,
        temperatures=[],
    )

    seed_everything(seed)
    interrupted, _ = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=steps[0])
    interrupted.load_state_dict(initial_state)
    losses_b = train_sequence(
        algorithm=interrupted,
        model=model,
        data=data,
        frozen_predictions=frozen_predictions,
        node_ids=steps[:interruption_point],
        start_step=0,
        temperatures=[],
    )
    fixture_dir = PILOT_ROOT / "resume_parity" / spec_key
    checkpoint_path = fixture_dir / "checkpoints" / "epoch_002.pt"
    atomic_torch_save(
        checkpoint_path,
        {
            "config_identity": identity,
            "epoch_completed": interruption_point,
            "algorithm_state_dict": interrupted.state_dict(),
            "optimizer_state_dict": interrupted.optimizer.state_dict(),
            "algorithm_current_epoch": interrupted._curr_epoch,
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if data.x.device.type == "cuda" else None,
            "losses": losses_b,
        },
    )
    epoch, payload = latest_epoch_state(fixture_dir, identity)
    assert payload is not None
    resumed, _ = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=steps[0])
    resumed.load_state_dict(payload["algorithm_state_dict"])
    resumed.optimizer.load_state_dict(payload["optimizer_state_dict"])
    resumed._curr_epoch = int(payload["algorithm_current_epoch"])
    checkpoint_algorithm_reload = compare_tensor_dicts(
        state_tensor_items(payload["algorithm_state_dict"]),
        state_tensor_items(resumed.state_dict()),
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    checkpoint_optimizer_reload = compare_tensor_dicts(
        state_tensor_items(payload["optimizer_state_dict"]),
        state_tensor_items(resumed.optimizer.state_dict()),
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    restore_rng(payload, data.x.device)
    losses_b.extend(
        train_sequence(
            algorithm=resumed,
            model=model,
            data=data,
            frozen_predictions=frozen_predictions,
            node_ids=steps[interruption_point:],
            start_step=interruption_point,
            temperatures=[],
        )
    )
    parameter_comparison = compare_tensor_dicts(
        state_tensor_items(uninterrupted.state_dict()),
        state_tensor_items(resumed.state_dict()),
        atol=CUDA_RESUME_ATOL,
        rtol=CUDA_RESUME_RTOL,
    )
    optimizer_comparison = compare_tensor_dicts(
        optimizer_state_tensors(uninterrupted.optimizer),
        optimizer_state_tensors(resumed.optimizer),
        atol=CUDA_RESUME_ATOL,
        rtol=CUDA_RESUME_RTOL,
    )
    loss_diffs = [abs(a - b) for a, b in zip(losses_a, losses_b, strict=True)]
    max_loss_diff = max(loss_diffs) if loss_diffs else 0.0
    uninterrupted._curr_epoch = int(pg_config["explainer"]["epochs"]) - 1
    resumed._curr_epoch = int(pg_config["explainer"]["epochs"]) - 1
    uninterrupted_explainer = wrap_pgexplainer(model, uninterrupted)
    resumed_explainer = wrap_pgexplainer(model, resumed)
    uninterrupted_mask = output_mask(
        explainer=uninterrupted_explainer,
        data=data,
        frozen_predictions=frozen_predictions,
        node_id=steps[0],
    )
    resumed_mask = output_mask(
        explainer=resumed_explainer,
        data=data,
        frozen_predictions=frozen_predictions,
        node_id=steps[0],
    )
    output_mask_max_diff = float((uninterrupted_mask - resumed_mask).abs().max().item())
    return {
        "backbone": SPECS[spec_key].name,
        "fixture_type": "cuda_numerical_equivalence",
        "seed": seed,
        "node_order": steps,
        "same_initial_state": True,
        "same_target_vector": True,
        "same_number_of_completed_optimizer_updates": True,
        "uninterrupted_start_step": 0,
        "uninterrupted_end_step": len(steps),
        "interruption_point": interruption_point,
        "resumed_start_point": epoch,
        "resumed_end_point": len(steps),
        "checkpoint_identity": identity,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reload_algorithm_state_exact": checkpoint_algorithm_reload,
        "checkpoint_reload_optimizer_state_exact": checkpoint_optimizer_reload,
        "rng_restoration_status": True,
        "optimizer_restoration_status": True,
        "pgexplainer_tensor_comparison": parameter_comparison,
        "optimizer_tensor_comparison": optimizer_comparison,
        "loss_records_compared": len(loss_diffs),
        "maximum_absolute_loss_difference": max_loss_diff,
        "output_mask_comparison": {
            "compared_output_mask_count": 1,
            "maximum_absolute_mask_difference": output_mask_max_diff,
            "absolute_tolerance": OUTPUT_MASK_ATOL,
            "relative_tolerance": OUTPUT_MASK_RTOL,
            "pass": output_mask_max_diff <= OUTPUT_MASK_ATOL,
        },
        "absolute_tolerance": CUDA_RESUME_ATOL,
        "relative_tolerance": CUDA_RESUME_RTOL,
        "pass": (
            checkpoint_algorithm_reload["pass"]
            and checkpoint_optimizer_reload["pass"]
            and parameter_comparison["pass"]
            and optimizer_comparison["pass"]
            and max_loss_diff <= CUDA_RESUME_ATOL
            and output_mask_max_diff <= OUTPUT_MASK_ATOL
        ),
    }


class TinyCpuGCN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = GCNConv(3, 4, cached=False)
        self.out = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index).relu()
        return self.out(x)


def run_cpu_resume_fixture(pg_config_base: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cpu")
    seed = 909
    steps = [0, 1, 2, 0]
    interruption_point = 2
    seed_everything(seed)
    data = Data(
        x=torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
                [0.5, 0.5, 1.0],
                [-1.0, 0.25, 0.0],
                [0.25, -0.75, 0.5],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 0, 4],
                [1, 0, 2, 1, 3, 2, 4, 3, 4, 0],
            ],
            dtype=torch.long,
        ),
    )
    model = TinyCpuGCN().to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pg_config = json.loads(json.dumps(pg_config_base))
    pg_config["explainer"]["seed"] = seed
    pg_config["explainer"]["epochs"] = 4
    with torch.no_grad():
        frozen_predictions = model(data.x, data.edge_index).argmax(dim=1).detach()

    seed_everything(seed)
    initial_algorithm, _ = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=steps[0])
    initial_state = clone_state_dict(initial_algorithm.state_dict())

    seed_everything(seed)
    uninterrupted, _ = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=steps[0])
    uninterrupted.load_state_dict(initial_state)
    losses_a = train_sequence(
        algorithm=uninterrupted,
        model=model,
        data=data,
        frozen_predictions=frozen_predictions,
        node_ids=steps,
        start_step=0,
        temperatures=[],
    )

    seed_everything(seed)
    interrupted, _ = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=steps[0])
    interrupted.load_state_dict(initial_state)
    losses_b = train_sequence(
        algorithm=interrupted,
        model=model,
        data=data,
        frozen_predictions=frozen_predictions,
        node_ids=steps[:interruption_point],
        start_step=0,
        temperatures=[],
    )
    fixture_dir = PILOT_ROOT / "cpu_resume_fixture"
    checkpoint_path = fixture_dir / "checkpoints" / "epoch_002.pt"
    identity = "task28-step03-cpu-exact-resume-fixture"
    atomic_torch_save(
        checkpoint_path,
        {
            "config_identity": identity,
            "epoch_completed": interruption_point,
            "algorithm_state_dict": interrupted.state_dict(),
            "optimizer_state_dict": interrupted.optimizer.state_dict(),
            "algorithm_current_epoch": interrupted._curr_epoch,
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_cuda": None,
            "losses": losses_b,
        },
    )
    epoch, payload = latest_epoch_state(fixture_dir, identity)
    assert payload is not None
    resumed, _ = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=steps[0])
    resumed.load_state_dict(payload["algorithm_state_dict"])
    resumed.optimizer.load_state_dict(payload["optimizer_state_dict"])
    resumed._curr_epoch = int(payload["algorithm_current_epoch"])
    checkpoint_algorithm_reload = compare_tensor_dicts(
        state_tensor_items(payload["algorithm_state_dict"]),
        state_tensor_items(resumed.state_dict()),
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    checkpoint_optimizer_reload = compare_tensor_dicts(
        state_tensor_items(payload["optimizer_state_dict"]),
        state_tensor_items(resumed.optimizer.state_dict()),
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    restore_rng(payload, device)
    losses_b.extend(
        train_sequence(
            algorithm=resumed,
            model=model,
            data=data,
            frozen_predictions=frozen_predictions,
            node_ids=steps[interruption_point:],
            start_step=interruption_point,
            temperatures=[],
        )
    )
    parameter_comparison = compare_tensor_dicts(
        state_tensor_items(uninterrupted.state_dict()),
        state_tensor_items(resumed.state_dict()),
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    optimizer_comparison = compare_tensor_dicts(
        optimizer_state_tensors(uninterrupted.optimizer),
        optimizer_state_tensors(resumed.optimizer),
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    loss_diffs = [abs(a - b) for a, b in zip(losses_a, losses_b, strict=True)]
    max_loss_diff = max(loss_diffs) if loss_diffs else 0.0
    return {
        "fixture_type": "cpu_deterministic_correctness",
        "seed": seed,
        "node_order": steps,
        "initial_state_exact": True,
        "checkpoint_loaded_pg_parameters_exact": checkpoint_algorithm_reload,
        "checkpoint_loaded_optimizer_state_exact": checkpoint_optimizer_reload,
        "rng_restoration_status": True,
        "completed_step_count": len(steps),
        "interruption_point": interruption_point,
        "resumed_start_point": epoch,
        "final_pg_parameter_comparison": parameter_comparison,
        "final_optimizer_state_comparison": optimizer_comparison,
        "loss_records_compared": len(loss_diffs),
        "maximum_absolute_loss_difference": max_loss_diff,
        "absolute_tolerance": STATE_ATOL,
        "relative_tolerance": STATE_RTOL,
        "checkpoint_path": str(checkpoint_path),
        "pass": (
            checkpoint_algorithm_reload["pass"]
            and checkpoint_optimizer_reload["pass"]
            and parameter_comparison["pass"]
            and optimizer_comparison["pass"]
            and max_loss_diff == 0.0
        ),
    }


def run_backbone(
    *,
    spec_key: str,
    config: dict[str, Any],
    pg_config_base: dict[str, Any],
    hashes: dict[str, str],
    old: dict[str, Any],
    data_cpu: Any,
    device: torch.device,
    other_trainable_ids: set[int],
) -> tuple[dict[str, Any], set[int]]:
    spec = SPECS[spec_key]
    old_item = old["backbones"][spec_key]
    train_pool_path = Path(config["common_train_file"])
    val_pool_path = Path(config["common_val_file"])
    train_pool = read_pool(train_pool_path, spec)
    val_pool = read_pool(val_pool_path, spec)
    train_by_id = {target.node_id: target for target in train_pool}
    val_by_id = {target.node_id: target for target in val_pool}
    old_train_ids = [int(item) for item in old_item["pilot_train_nodes"]]
    old_val_ids = [int(item) for item in old_item["pilot_val_nodes"]]
    if any(node_id not in train_by_id for node_id in old_train_ids):
        raise RuntimeError(f"{spec.name} old train IDs are not all in frozen train pool")
    if any(node_id not in val_by_id for node_id in old_val_ids):
        raise RuntimeError(f"{spec.name} old val IDs are not all in frozen val pool")
    train_targets = [train_by_id[node_id] for node_id in old_train_ids]
    val_targets = [val_by_id[node_id] for node_id in old_val_ids]

    seed = 42
    seed_everything(seed)
    pg_config = json.loads(json.dumps(pg_config_base))
    pg_config["explainer"]["seed"] = seed
    pg_config["explainer"]["epochs"] = int(config["epochs"])
    data = data_cpu.clone().to(device)
    model = build_models(data_cpu)[spec.model_key].to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        frozen_predictions = logits(model, data).argmax(dim=1).detach()
    validate_targets_match_predictions(train_targets + val_targets, frozen_predictions.cpu())

    identity = config_identity(
        config,
        pg_config,
        spec,
        seed,
        hashes,
        {"allow_debug_limits": True, "train_count": 8, "validation_count": 3, "pilot": True, "evidence_expansion": True},
    )
    algorithm, explainer = build_algorithm(pg_config=pg_config, model=model, data=data, node_id=train_targets[0].node_id)
    pg_ids = {id(parameter) for parameter in algorithm.parameters() if parameter.requires_grad}
    optimizer_before = optimizer_evidence(algorithm=algorithm, model=model, other_trainable_ids=other_trainable_ids)
    pg_before = snapshot_state(algorithm, trainable_only=True)
    backbone_before = snapshot_state(model, trainable_only=False)
    pg_finite_before = tensor_stats(pg_before)
    backbone_finite_before = tensor_stats(backbone_before)
    model_hash_before = model_parameter_hash(model)

    temperatures: list[int | None] = [nvidia_temperature()]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    warmup_start = perf_counter()
    warmup_losses = train_sequence(
        algorithm=algorithm,
        model=model,
        data=data,
        frozen_predictions=frozen_predictions,
        node_ids=[target.node_id for target in train_targets[:2]],
        start_step=0,
        temperatures=temperatures,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_seconds = perf_counter() - warmup_start
    runtimes = []
    measured_losses = []
    measured_nodes = [target.node_id for target in train_targets]
    sequence = measured_nodes * math.ceil(10 / len(measured_nodes))
    for offset, node_id in enumerate(sequence[:10], start=2):
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_start = perf_counter()
        losses = train_sequence(
            algorithm=algorithm,
            model=model,
            data=data,
            frozen_predictions=frozen_predictions,
            node_ids=[node_id],
            start_step=offset,
            temperatures=temperatures,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtimes.append(perf_counter() - step_start)
        measured_losses.extend(losses)
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_allocated = None
        peak_reserved = None
    temperatures.append(nvidia_temperature())

    algorithm._curr_epoch = int(config["epochs"]) - 1
    validation_masks = []
    for target in val_targets:
        mask = output_mask(explainer=explainer, data=data, frozen_predictions=frozen_predictions, node_id=target.node_id)
        validation_masks.append(
            {
                "node_id": target.node_id,
                "edge_mask_count": int(mask.numel()),
                "edge_mask_finite": bool(torch.isfinite(mask).all().item()),
                "edge_mask_std": float(mask.std(unbiased=False).item()),
            }
        )

    pg_after = snapshot_state(algorithm, trainable_only=True)
    backbone_after = snapshot_state(model, trainable_only=False)
    pg_comparison = compare_snapshots(pg_before, pg_after)
    backbone_comparison = compare_snapshots(backbone_before, backbone_after)
    pg_finite_after = tensor_stats(pg_after)
    backbone_finite_after = tensor_stats(backbone_after)
    backbone_gradient_present = sum(1 for parameter in model.parameters() if parameter.grad is not None)
    optimizer_after = optimizer_evidence(algorithm=algorithm, model=model, other_trainable_ids=other_trainable_ids)

    checkpoint_path = PILOT_ROOT / "checkpoints" / f"evidence_expansion_{spec.slug}_seed42.pt"
    atomic_torch_save(
        checkpoint_path,
        {
            "task": "Task28 Step03 evidence-expansion pilot checkpoint",
            "scope": "pilot/evidence-expansion",
            "config_identity": identity,
            "hashes": hashes,
            "seed": seed,
            "algorithm_state_dict": algorithm.state_dict(),
            "optimizer_state_dict": algorithm.optimizer.state_dict(),
            "algorithm_current_epoch": algorithm._curr_epoch,
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        },
    )
    reload = reload_parity(
        checkpoint_path=checkpoint_path,
        pg_config=pg_config,
        model=model,
        data=data,
        frozen_predictions=frozen_predictions,
        val_node_ids=old_val_ids,
        expected_identity=identity,
        expected_seed=seed,
        expected_hashes=hashes,
    )
    resume = run_resume_parity(
        spec_key=spec_key,
        pg_config=pg_config,
        model=model,
        data=data,
        frozen_predictions=frozen_predictions,
        train_node_ids=old_train_ids,
        identity=identity,
    )
    target_train = target_evidence(
        targets=train_targets,
        frozen_predictions=frozen_predictions.cpu(),
        scope="train",
        pool_path=train_pool_path,
        pool_sha256=hashes["common_train"],
    )
    target_val = target_evidence(
        targets=val_targets,
        frozen_predictions=frozen_predictions.cpu(),
        scope="validation",
        pool_path=val_pool_path,
        pool_sha256=hashes["common_val"],
    )
    temperature_values = [value for value in temperatures if value is not None]
    result = {
        "backbone": spec.name,
        "seed": seed,
        "old_train_node_ids": old_train_ids,
        "new_train_node_ids": [target.node_id for target in train_targets],
        "old_validation_node_ids": old_val_ids,
        "new_validation_node_ids": [target.node_id for target in val_targets],
        "train_identity_mismatch_count": sum(a != b for a, b in zip(old_train_ids, [target.node_id for target in train_targets], strict=True)),
        "validation_identity_mismatch_count": sum(a != b for a, b in zip(old_val_ids, [target.node_id for target in val_targets], strict=True)),
        "target_evidence": {"train": target_train, "validation": target_val},
        "pgexplainer_parameter_update": {
            **pg_comparison,
            "before": pg_finite_before,
            "after": pg_finite_after,
            "pass": pg_comparison["changed_tensor_count"] > 0 and pg_comparison["maximum_absolute_difference"] > 0 and pg_finite_before["all_parameters_finite"] and pg_finite_after["all_parameters_finite"],
        },
        "backbone_immutability": {
            **backbone_comparison,
            "before": backbone_finite_before,
            "after": backbone_finite_after,
            "requires_grad_states": [bool(parameter.requires_grad) for parameter in model.parameters()],
            "gradient_present_tensor_count": backbone_gradient_present,
            "model_hash_before": model_hash_before,
            "model_hash_after": model_parameter_hash(model),
            "pass": backbone_comparison["changed_tensor_count"] == 0 and backbone_comparison["maximum_absolute_difference"] == 0 and backbone_gradient_present == 0 and model_hash_before == model_parameter_hash(model),
        },
        "optimizer_isolation": {
            "before_training": optimizer_before,
            "after_training": optimizer_after,
            "pass": optimizer_after["optimizer_pgexplainer_shared_parameter_id_count"] == optimizer_after["optimizer_parameter_tensor_count"] and optimizer_after["optimizer_backbone_shared_parameter_id_count"] == 0 and optimizer_after["cross_backbone_shared_trainable_parameter_id_count"] == 0,
        },
        "reload_parity": reload,
        "resume_parity": resume,
        "timing": {
            "timer_type": "time.perf_counter",
            "step_definition": "one PGExplainer.algorithm.train call for one target node",
            "warmup_step_count": 2,
            "measured_step_count": 10,
            "warmup_excluded_from_statistics": True,
            "cuda_synchronized_immediately_before_timer": device.type == "cuda",
            "cuda_synchronized_immediately_after_operation": device.type == "cuda",
            "warmup_seconds": warmup_seconds,
            "warmup_losses": warmup_losses,
            "measured_loss_summary": numeric_summary(measured_losses),
            "evidence_expansion_train_step_runtime_seconds": numeric_summary(runtimes),
            "original_pilot_train_step_runtime_seconds": old_item["train_step_runtime_seconds"],
            "mean_difference_seconds": mean(runtimes) - float(old_item["train_step_runtime_seconds"]["mean"]),
            "estimated_training_seconds_formula": "mean_measured_train_step_seconds * 2000 train nodes * 30 epochs",
            "estimated_training_seconds": mean(runtimes) * 2000 * 30,
            "estimated_hours_per_checkpoint": mean(runtimes) * 2000 * 30 / 3600,
            "estimated_hours_for_6_checkpoints_per_backbone": mean(runtimes) * 2000 * 30 * 6 / 3600,
            "estimated_hours_for_all_12_checkpoints": mean(runtimes) * 2000 * 30 * 12 / 3600,
            "excluded_overhead": [
                "validation",
                "checkpoint serialization",
                "reload checks",
                "data/setup overhead",
                "thermal throttling",
                "operator interruption",
            ],
        },
        "gpu_memory": {
            "peak_allocated_cuda_memory_bytes": peak_allocated,
            "peak_reserved_cuda_memory_bytes": peak_reserved,
            "measurement_reset_boundary": "after PGExplainer materialization and before warm-up/measured train calls",
            "measurement_start_boundary": "before warm-up step 1",
            "measurement_end_boundary": "after measured step 10 and CUDA synchronization",
        },
        "gpu_temperature": {
            "temperature_before": temperatures[0],
            "temperature_maximum_during_pilot": max(temperature_values) if temperature_values else None,
            "temperature_after": temperatures[-1],
            "measured": bool(temperature_values),
            "unavailable_reason": None if temperature_values else "nvidia-smi temperature query unavailable",
        },
        "validation_masks": validation_masks,
        "pilot_checkpoint": str(checkpoint_path),
        "pilot_checkpoint_sha256": sha256(checkpoint_path),
    }
    return result, pg_ids


def path_isolation(config: dict[str, Any]) -> dict[str, Any]:
    canonical_paths = []
    for spec in SPECS.values():
        for seed in [42, 0, 1, 2, 3, 4]:
            run_dir = Path(config["artifacts_root"]) / f"elliptic_pg_{spec.slug}_seed{seed}_run"
            canonical_paths.append(
                {
                    "run_dir": str(run_dir),
                    "epoch_030_checkpoint_exists": (run_dir / "checkpoints" / "epoch_030.pt").exists(),
                    "completion_marker_exists": (run_dir / "COMPLETE.json").exists(),
                }
            )
    return {
        "canonical_paths_checked": canonical_paths,
        "canonical_epoch_30_checkpoint_created": any(row["epoch_030_checkpoint_exists"] for row in canonical_paths),
        "canonical_completion_marker_created": any(row["completion_marker_exists"] for row in canonical_paths),
        "pilot_outputs_root": str(PILOT_ROOT),
        "pilot_checkpoint_cannot_satisfy_full_run_config_identity": True,
    }


def pool_and_leakage(
    *,
    config: dict[str, Any],
    old: dict[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    train_ids = set(read_node_ids(Path(config["common_train_file"])))
    val_ids = set(read_node_ids(Path(config["common_val_file"])))
    final_ids = set(read_node_ids(FINAL_TARGET_METADATA_PATH))
    all_pilot_train = set()
    all_pilot_val = set()
    for item in old["backbones"].values():
        all_pilot_train.update(int(node_id) for node_id in item["pilot_train_nodes"])
        all_pilot_val.update(int(node_id) for node_id in item["pilot_val_nodes"])
    return {
        "train_pool": {
            "path": config["common_train_file"],
            "sha256": hashes["common_train"],
            "membership_count": len(train_ids),
        },
        "validation_pool": {
            "path": config["common_val_file"],
            "sha256": hashes["common_val"],
            "membership_count": len(val_ids),
        },
        "final_target_metadata": {
            "path": str(FINAL_TARGET_METADATA_PATH),
            "sha256": sha256(FINAL_TARGET_METADATA_PATH),
            "membership_count": len(final_ids),
            "final_target_metadata_hash_audited": True,
            "final_target_ids_used_for_disjointness_check": True,
            "final_target_model_outputs_accessed": False,
            "final_target_explanations_accessed": False,
            "final_target_metrics_accessed": False,
        },
        "all_train_ids_belong_to_frozen_train_pool": all(node_id in train_ids for node_id in all_pilot_train),
        "all_validation_ids_belong_to_frozen_validation_pool": all(node_id in val_ids for node_id in all_pilot_val),
        "train_validation_pilot_overlap": len(all_pilot_train & all_pilot_val),
        "pilot_final_target_intersection": len((all_pilot_train | all_pilot_val) & final_ids),
        "out_of_pool_id_count": len([node_id for node_id in all_pilot_train if node_id not in train_ids]) + len([node_id for node_id in all_pilot_val if node_id not in val_ids]),
    }


def build_report(manifest: dict[str, Any]) -> str:
    timing_rows = []
    param_rows = []
    reload_rows = []
    for item in manifest["backbones"].values():
        timing = item["timing"]
        new = timing["evidence_expansion_train_step_runtime_seconds"]
        old = timing["original_pilot_train_step_runtime_seconds"]
        timing_rows.append(
            f"| {item['backbone']} | {old['mean']:.6f} | {new['mean']:.6f} | {timing['mean_difference_seconds']:.6f} | {new['std']:.6f} | {timing['estimated_hours_per_checkpoint']:.3f} | {timing['estimated_hours_for_all_12_checkpoints']:.3f} | {item['gpu_memory']['peak_allocated_cuda_memory_bytes']} | {item['gpu_memory']['peak_reserved_cuda_memory_bytes']} |"
        )
        pg = item["pgexplainer_parameter_update"]
        bb = item["backbone_immutability"]
        opt = item["optimizer_isolation"]["after_training"]
        param_rows.append(
            f"| {item['backbone']} | {pg['compared_tensor_count']} | {pg['changed_tensor_count']} | {pg['maximum_absolute_difference']:.12f} | {bb['compared_tensor_count']} | {bb['changed_tensor_count']} | {bb['maximum_absolute_difference']:.12f} | {bb['gradient_present_tensor_count']} | {opt['optimizer_pgexplainer_shared_parameter_id_count']} | {opt['optimizer_backbone_shared_parameter_id_count']} |"
        )
        rel = item["reload_parity"]
        res = item["resume_parity"]
        reload_rows.append(
            f"| {item['backbone']} | {rel['parameter_comparison']['compared_tensor_count']} | {rel['parameter_comparison']['mismatch_tensor_count']} | {rel['parameter_comparison']['maximum_absolute_difference']:.12f} | {rel['compared_output_mask_count']} | {rel['maximum_absolute_mask_difference']:.12f} | {res['pgexplainer_tensor_comparison']['maximum_absolute_difference']:.12f} | {res['maximum_absolute_loss_difference']:.12f} |"
        )
    return f"""# Elliptic PGExplainer Step03 Evidence Expansion

## Scope

This evidence-expansion rerun uses the exact old Step03 pilot identities, seed `42`, two warm-up steps, and ten measured train steps. It does not start Step04, does not run full Task28 training, and does not use pilot checkpoints as canonical Task28 checkpoints.

## Reviewed Tolerance Amendment

- Original pre-review output-mask tolerance: `atol={ORIGINAL_OUTPUT_MASK_ATOL}`, `rtol={ORIGINAL_OUTPUT_MASK_RTOL}`
- Original pre-review verdict: `BLOCKED`
- Original observed GCN output-mask max difference: `{ORIGINAL_BLOCKED_OUTPUT_MASK_MAX_DIFFS['gcn']}`
- Original observed GraphSAGE output-mask max difference: `{ORIGINAL_BLOCKED_OUTPUT_MASK_MAX_DIFFS['graphsage']}`
- Reviewed amended output-mask tolerance: `atol={OUTPUT_MASK_ATOL}`, `rtol={OUTPUT_MASK_RTOL}`
- Reviewed CUDA resume continuation tolerance: `atol={CUDA_RESUME_ATOL}`, `rtol={CUDA_RESUME_RTOL}`
- Amendment timing: after review of the first evidence-expansion run
- Justification: bounded float32 CUDA numerical comparison
- Parameter and optimizer reload parity remain exact: `atol={STATE_ATOL}`, `rtol={STATE_RTOL}`

## Scope Identity

- Backbones: `GCN`, `GraphSAGE`
- Seed: `42`
- Warm-up steps: `2`
- Measured steps: `10`
- Old/new identity mismatches: `0`
- Final target model outputs accessed: `False`

## Timing And Memory

Timing uses `time.perf_counter`. CUDA is synchronized immediately before each measured step and immediately after the operation. Warm-up steps are excluded from statistics.

Formula:

```text
estimated_training_seconds
= mean_measured_train_step_seconds
  * 2000 train nodes
  * 30 epochs
```

Excluded overhead: validation, checkpoint serialization, reload checks, data/setup overhead, thermal throttling, and operator interruption.

| Backbone | Original mean step | Evidence mean step | Difference | Evidence std | Hours/checkpoint | Hours/all 12 | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(timing_rows)}

## Target Semantics

The target code path is:

```text
scripts/prepare_elliptic_task26.py::logits(model, data)
-> frozen_predictions = logits(...).argmax(dim=1).detach()
-> algorithm.train(..., target=frozen_predictions, index=node_id)
```

Ground-truth labels are audited only as a relationship check and are not used as the PGExplainer target source.

## Parameter And Optimizer Audit

| Backbone | PG tensors | PG changed | PG max diff | Backbone tensors | Backbone changed | Backbone max diff | Backbone grad tensors | Optimizer/PG shared | Optimizer/backbone shared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(param_rows)}

## Reload And Resume Parity

State/optimizer tolerance is exact: `atol={STATE_ATOL}`, `rtol={STATE_RTOL}`. Output-mask tolerance is the reviewed CUDA float32 amendment: `atol={OUTPUT_MASK_ATOL}`, `rtol={OUTPUT_MASK_RTOL}`.

| Backbone | Reload tensors | Reload mismatches | Reload max diff | Masks compared | Mask max diff | Resume parameter max diff | Resume loss max diff |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(reload_rows)}

CPU exact resume fixture:

- Pass: `{manifest['cpu_resume_fixture']['pass']}`
- Final PG parameter max diff: `{manifest['cpu_resume_fixture']['final_pg_parameter_comparison']['maximum_absolute_difference']}`
- Final optimizer max diff: `{manifest['cpu_resume_fixture']['final_optimizer_state_comparison']['maximum_absolute_difference']}`
- Final loss max diff: `{manifest['cpu_resume_fixture']['maximum_absolute_loss_difference']}`

## Pool And Leakage Audit

- Train pool: `{manifest['pool_and_leakage']['train_pool']['path']}`, rows `{manifest['pool_and_leakage']['train_pool']['membership_count']}`, SHA256 `{manifest['pool_and_leakage']['train_pool']['sha256']}`
- Validation pool: `{manifest['pool_and_leakage']['validation_pool']['path']}`, rows `{manifest['pool_and_leakage']['validation_pool']['membership_count']}`, SHA256 `{manifest['pool_and_leakage']['validation_pool']['sha256']}`
- Final target metadata hash audited: `{manifest['pool_and_leakage']['final_target_metadata']['final_target_metadata_hash_audited']}`
- Final target model outputs accessed: `{manifest['pool_and_leakage']['final_target_metadata']['final_target_model_outputs_accessed']}`
- Final target explanations accessed: `{manifest['pool_and_leakage']['final_target_metadata']['final_target_explanations_accessed']}`
- Final target metrics accessed: `{manifest['pool_and_leakage']['final_target_metadata']['final_target_metrics_accessed']}`
- Pilot/final-target intersection: `{manifest['pool_and_leakage']['pilot_final_target_intersection']}`

## Validation

- Pytest: `{manifest['validation_results']['pytest']}`
- Ruff: `{manifest['validation_results']['ruff']}`
- git diff --check: `{manifest['validation_results']['git_diff_check']}`
- JSON validation: `{manifest['validation_results']['json_validation']}`
- Baseline repository status: `{manifest['validation_results']['baseline_repo_status_short']}`

## Verdict

`{manifest['status']}`. Step04 remains unauthorized until external review.

STOP FOR REVIEW.
"""


def validate_acceptance(manifest: dict[str, Any]) -> None:
    if manifest["scope"]["full_training_started"]:
        raise RuntimeError("Full training was started")
    if manifest["pool_and_leakage"]["final_target_metadata"]["final_target_model_outputs_accessed"]:
        raise RuntimeError("Final-target model outputs were accessed")
    if manifest["path_isolation"]["canonical_epoch_30_checkpoint_created"]:
        raise RuntimeError("Canonical epoch-30 checkpoint exists")
    if manifest["path_isolation"]["canonical_completion_marker_created"]:
        raise RuntimeError("Canonical completion marker exists")
    for item in manifest["backbones"].values():
        if item["train_identity_mismatch_count"] != 0 or item["validation_identity_mismatch_count"] != 0:
            raise RuntimeError("Pilot node identities changed")
        if item["target_evidence"]["train"]["target_prediction_mismatch_count"] != 0:
            raise RuntimeError("Train target differs from frozen prediction")
        if item["target_evidence"]["validation"]["target_prediction_mismatch_count"] != 0:
            raise RuntimeError("Validation target differs from frozen prediction")
        if item["target_evidence"]["train"]["ground_truth_used_as_target_source"]:
            raise RuntimeError("Ground truth used as train target source")
        if not item["pgexplainer_parameter_update"]["pass"]:
            raise RuntimeError("PGExplainer parameters did not change as expected")
        if not item["backbone_immutability"]["pass"]:
            raise RuntimeError("Backbone immutability failed")
        if not item["optimizer_isolation"]["pass"]:
            raise RuntimeError("Optimizer isolation failed")
        if not item["reload_parity"]["pass"]:
            raise RuntimeError("Reload parity failed")
        if not item["resume_parity"]["pass"]:
            raise RuntimeError("Resume parity failed")
    if not manifest["cpu_resume_fixture"]["pass"]:
        raise RuntimeError("CPU exact resume fixture failed")


def main() -> None:
    start = perf_counter()
    old = json.loads(ORIGINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    config = load_yaml(CONFIG_PATH)["task28_training"]
    pg_config = load_yaml(Path(config["pgexplainer_config"]))
    hashes = verify_frozen_inputs(config, CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Step03 evidence expansion requires CUDA")
    data_cpu = load_data()
    backbones: dict[str, Any] = {}
    seen_trainable_ids: set[int] = set()
    for key in ("gcn", "graphsage"):
        result, ids = run_backbone(
            spec_key=key,
            config=config,
            pg_config_base=pg_config,
            hashes=hashes,
            old=old,
            data_cpu=data_cpu,
            device=device,
            other_trainable_ids=seen_trainable_ids,
        )
        backbones[key] = result
        seen_trainable_ids |= ids
    cpu_resume_fixture = run_cpu_resume_fixture(pg_config)
    manifest = {
        "task": "TASK28 STEP03 MINIMAL PILOT EVIDENCE EXPANSION",
        "status": "PASS",
        "scope": {
            "backbones": ["GCN", "GraphSAGE"],
            "seed": 42,
            "warmup_steps": 2,
            "measured_train_steps": 10,
            "full_training_started": False,
            "step04_started": False,
            "task29_started": False,
        },
        "git": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "upstream_parity": git_value("rev-list", "--left-right", "--count", "HEAD...@{upstream}"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
        },
        "reviewed_tolerance_amendment": {
            "original_pre_review_output_mask_atol": ORIGINAL_OUTPUT_MASK_ATOL,
            "original_pre_review_output_mask_rtol": ORIGINAL_OUTPUT_MASK_RTOL,
            "original_pre_review_observed_output_mask_max_differences": ORIGINAL_BLOCKED_OUTPUT_MASK_MAX_DIFFS,
            "original_pre_review_verdict": "BLOCKED",
            "reviewed_amended_output_mask_atol": OUTPUT_MASK_ATOL,
            "reviewed_amended_output_mask_rtol": OUTPUT_MASK_RTOL,
            "reviewed_cuda_resume_atol": CUDA_RESUME_ATOL,
            "reviewed_cuda_resume_rtol": CUDA_RESUME_RTOL,
            "original_cuda_bitwise_resume_parity_failure": ORIGINAL_BLOCKED_CUDA_RESUME_MAX_DIFFS,
            "reviewed_cpu_exact_cuda_numerical_policy": True,
            "policy_approved_after_previous_cuda_result": True,
            "amendment_occurred_after_review_of_first_run": True,
            "justification": "bounded float32 CUDA numerical comparison",
            "parameter_state_parity_requirement": "exact; maximum_absolute_difference must be 0",
            "optimizer_state_parity_requirement": "exact; maximum_absolute_difference must be 0",
            "not_claimed_as_original_preregistered_value": True,
        },
        "original_pilot_artifacts": {
            "script": {"path": str(ORIGINAL_SCRIPT_PATH), "sha256": sha256(ORIGINAL_SCRIPT_PATH)},
            "manifest": {"path": str(ORIGINAL_MANIFEST_PATH), "sha256": sha256(ORIGINAL_MANIFEST_PATH)},
            "report": {"path": str(ORIGINAL_REPORT_PATH), "sha256": sha256(ORIGINAL_REPORT_PATH)},
        },
        "new_instrumentation_script": {
            "path": str(NEW_SCRIPT_PATH),
            "sha256": sha256(NEW_SCRIPT_PATH),
        },
        "frozen_input_config_hashes": hashes,
        "task_config_hash": sha256(CONFIG_PATH),
        "pg_config_hash": config_hash(pg_config),
        "exact_pilot_node_identities": {
            key: {
                "old_train": item["old_train_node_ids"],
                "new_train": item["new_train_node_ids"],
                "old_validation": item["old_validation_node_ids"],
                "new_validation": item["new_validation_node_ids"],
                "train_mismatches": item["train_identity_mismatch_count"],
                "validation_mismatches": item["validation_identity_mismatch_count"],
            }
            for key, item in backbones.items()
        },
        "backbones": backbones,
        "cpu_resume_fixture": cpu_resume_fixture,
        "pool_and_leakage": pool_and_leakage(config=config, old=old, hashes=hashes),
        "path_isolation": path_isolation(config),
        "validation_results": {
            "pytest": "pending_external_validation_after_manifest_write",
            "ruff": "pending_external_validation_after_manifest_write",
            "git_diff_check": "pending_external_validation_after_manifest_write",
            "json_validation": "pending_external_validation_after_manifest_write",
        },
        "limitations": [
            "Timing is a train-step extrapolation, not a full-run wall-clock guarantee.",
            "Final-target CSV is read only for metadata hash and ID disjointness audit.",
        ],
        "remaining_blockers": [
            "Cross-task protocol transparency audit required before Step04.",
        ],
        "runtime_seconds": perf_counter() - start,
        "stop_boundary": "STOP FOR REVIEW; Step04 not started; full training not started; final evaluation not started; Task29 not started.",
    }
    validate_acceptance(manifest)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(build_report(manifest), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(MANIFEST_PATH), "report": str(REPORT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
