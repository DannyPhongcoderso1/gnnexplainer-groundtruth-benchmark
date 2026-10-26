"""CPU state verification only: never construct a model or an optimizer."""

import csv
import json
import math
from pathlib import Path

from common import read_json, require, sha


def pool_ids(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        ids = [int(row["node_id"]) for row in csv.DictReader(stream)]
    require(ids == sorted(set(ids)), "Frozen pool order/uniqueness mismatch")
    return ids


def check_tensors(payload, backbone, updates, optimizer_group, torch):
    dimension = 90 if backbone == "gcn" else 103
    shapes = {
        "mlp.0.weight": (64, 3 * dimension),
        "mlp.0.bias": (64,),
        "mlp.2.weight": (1, 64),
        "mlp.2.bias": (1,),
    }
    state = payload["algorithm_state_dict"]
    require(set(state) == set(shapes), "PG MLP state keys mismatch")
    for key, shape in shapes.items():
        item = state[key]
        require(
            isinstance(item, torch.Tensor)
            and tuple(item.shape) == shape
            and item.dtype == torch.float32
            and bool(torch.isfinite(item).all()),
            f"PG MLP tensor invalid: {key}",
        )
    adam = payload["optimizer_state_dict"]
    require(
        set(adam) == {"state", "param_groups"}
        and json.loads(json.dumps(adam["param_groups"])) == [optimizer_group],
        "Historical Adam settings changed",
    )
    require(set(adam["state"]) == {0, 1, 2, 3}, "Adam parameter identities invalid")
    for pid, shape in enumerate(shapes.values()):
        row = adam["state"][pid]
        require(set(row) == {"step", "exp_avg", "exp_avg_sq"}, "Adam state keys invalid")
        require(
            isinstance(row["step"], torch.Tensor)
            and tuple(row["step"].shape) == ()
            and row["step"].item() == updates,
            "Incorrect committed Adam update count",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            item = row[key]
            require(
                isinstance(item, torch.Tensor)
                and tuple(item.shape) == shape
                and item.dtype == torch.float32
                and bool(torch.isfinite(item).all()),
                "Invalid Adam moment tensor",
            )
        require(bool((row["exp_avg_sq"] >= 0).all()), "Negative Adam second moment")


def verify_epoch(path, backbone, seed, config, protocol, runtime=None, targets=2000):
    import torch
    import numpy as np

    path = Path(path)
    before = sha(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    epoch = payload["epoch_completed"]
    require(type(epoch) is int and 1 <= epoch <= 30, "Invalid completed epoch")
    require(
        payload["backbone"] == ("GCN" if backbone == "gcn" else "GraphSAGE")
        and payload["seed"] == seed
        and payload["algorithm_current_epoch"] == epoch - 1,
        "Epoch backbone/seed/progress identity mismatch",
    )
    check_tensors(payload, backbone, epoch * targets, protocol["optimizer_group"], torch)
    require(len(payload["epoch_rows"]) == epoch, "Epoch history length mismatch")
    for number, row in enumerate(payload["epoch_rows"], 1):
        require(
            row["epoch"] == number
            and row["num_train_instances"] == targets
            and row["seed"] == seed
            and row["backbone"] == payload["backbone"],
            "Epoch update accounting mismatch",
        )
        require(
            all(
                math.isfinite(float(row[k]))
                for k in ("mean_loss", "median_loss", "min_loss", "max_loss", "runtime_seconds")
            ),
            "Nonfinite epoch records",
        )
    require(payload["hashes"]["task_config"] == sha(config), "Saved training config hash mismatch")
    for key, expected in protocol["runner_input_hashes"].items():
        require(payload["hashes"][key] == expected, f"Saved frozen input hash mismatch: {key}")
    for key in (
        "rng_state_cpu",
        "rng_state_cuda",
        "rng_state_numpy",
        "hard_compatibility",
        "config_identity",
        "runtime_ledger",
    ):
        require(key in payload, f"Missing recoverable state: {key}")
    require(
        isinstance(payload["rng_state_cpu"], torch.Tensor)
        and payload["rng_state_cpu"].dtype == torch.uint8
        and payload["rng_state_cpu"].numel() > 0,
        "Invalid CPU RNG state",
    )
    require(
        isinstance(payload["rng_state_cuda"], list)
        and len(payload["rng_state_cuda"]) == 1
        and all(
            isinstance(t, torch.Tensor) and t.dtype == torch.uint8 and t.numel() > 0
            for t in payload["rng_state_cuda"]
        ),
        "Invalid isolated CUDA RNG state",
    )
    numpy_rng = payload["rng_state_numpy"]
    require(
        isinstance(numpy_rng, tuple)
        and len(numpy_rng) == 5
        and numpy_rng[0] == "MT19937"
        and isinstance(numpy_rng[1], np.ndarray)
        and numpy_rng[1].shape == (624,),
        "Invalid NumPy RNG state",
    )
    if runtime:
        require(
            payload["hard_compatibility"] == runtime["hard_compatibility"],
            "State HARD runtime identity mismatch",
        )
    require(sha(path) == before, "Epoch checkpoint changed during read-back")
    return {
        "status": "EPOCH_VERIFIED",
        "path": str(path),
        "sha256": before,
        "epoch_completed": epoch,
        "durable_updates": epoch * targets,
        "config_identity": payload["config_identity"],
        "backbone": backbone,
        "seed": seed,
    }


def exact_serialized_state(path, expected):
    """Completed-boundary read-back, including exact tensor/RNG values, with no model calls."""
    import numpy as np
    import torch

    actual = torch.load(path, map_location="cpu", weights_only=False)

    def compare(left, right, label):
        if isinstance(right, torch.Tensor):
            require(
                isinstance(left, torch.Tensor)
                and left.shape == right.shape
                and left.dtype == right.dtype
                and torch.equal(left.cpu(), right.detach().cpu()),
                f"Exact serialized tensor mismatch:{label}",
            )
        elif isinstance(right, np.ndarray):
            require(
                isinstance(left, np.ndarray) and np.array_equal(left, right),
                f"Exact serialized NumPy RNG mismatch:{label}",
            )
        elif isinstance(right, dict):
            require(
                isinstance(left, dict) and left.keys() == right.keys(),
                f"State keys mismatch:{label}",
            )
            for key in right:
                compare(left[key], right[key], label + "/" + str(key))
        elif isinstance(right, (list, tuple)):
            require(
                type(left) is type(right) and len(left) == len(right),
                f"State sequence mismatch:{label}",
            )
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                compare(a, b, label + "/" + str(index))
        else:
            require(left == right, f"State scalar mismatch:{label}")

    compare(actual, expected, "epoch_payload")


def verify_final(directory, backbone, seed, config, protocol, runtime):
    import torch
    from common import state_paths

    paths = state_paths(directory, backbone, seed)
    published = torch.load(paths["final"], map_location="cpu", weights_only=False)
    manifest = read_json(paths["manifest"])
    marker = read_json(paths["run"] / "COMPLETE.json")
    latest = paths["run"] / "checkpoints/epoch_030.pt"
    epoch_receipt = verify_epoch(latest, backbone, seed, config, protocol, runtime)
    state = torch.load(latest, map_location="cpu", weights_only=False)
    require(
        published["seed"] == seed
        and published["epochs_completed"] == 30
        and published["algorithm_current_epoch"] == 29,
        "Final seed/horizon invalid",
    )
    require(
        published["training_config"] == read_json(config)["task28_training"], "Final config changed"
    )
    require(
        published["train_node_ids"] == pool_ids(published["training_config"]["common_train_file"])
        and len(published["train_node_ids"]) == 2000,
        "Final train identities mismatch",
    )
    require(
        published["validation_node_ids"]
        == pool_ids(published["training_config"]["common_val_file"])
        and len(published["validation_node_ids"]) == 500,
        "Final validation identities mismatch",
    )
    check_tensors(published, backbone, 60000, protocol["optimizer_group"], torch)
    require(
        published["hard_compatibility"]
        == state["hard_compatibility"]
        == runtime["hard_compatibility"],
        "Final/epoch/runtime mismatch",
    )
    require(
        published["config_identity"]
        == state["config_identity"]
        == marker["config_identity"]
        == manifest["config_identity"],
        "Final config_identity mismatch",
    )
    for key, item in published["algorithm_state_dict"].items():
        require(torch.equal(item, state["algorithm_state_dict"][key]), "Final/epoch MLP mismatch")
    for pid, row in published["optimizer_state_dict"]["state"].items():
        for key, item in row.items():
            require(
                torch.equal(item, state["optimizer_state_dict"]["state"][pid][key]),
                "Final/epoch Adam mismatch",
            )
    digest = sha(paths["final"])
    require(
        manifest["status"] == "success"
        and marker["status"] == "complete"
        and manifest["artifact_sha256"] == marker["artifact_sha256"] == digest,
        "Original completion hash/status mismatch",
    )
    require(
        manifest["model_weights_unchanged"] is True
        and manifest["model_hash_before"] == manifest["model_hash_after"]
        and manifest["post_test_tuning"] is False
        and manifest["final_targets_accessed"] is False,
        "Backbone freeze/leakage receipt invalid",
    )
    require(
        manifest["train_count"] == 2000 and manifest["validation_count"] == 500,
        "Canonical full-pool counts invalid",
    )
    return {
        "status": "FINAL_VERIFIED",
        "backbone": backbone,
        "seed": seed,
        "checkpoint_path": str(paths["final"]),
        "checkpoint_sha256": digest,
        "epochs": 30,
        "optimizer_updates": 60000,
        "epoch_state": epoch_receipt,
        "training_runtime_seconds": manifest["training_runtime_seconds"],
    }
