from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts.run_elliptic_pgexplainer_training import (
    ALLOWED_SEEDS,
    SPECS,
    atomic_torch_save,
    config_identity,
    latest_epoch_state,
    read_pool,
    restore_rng,
    run_dir,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/evaluation/elliptic_pgexplainer_training.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["task28_training"]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_elliptic_pgexplainer_training.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_task28_training_config_is_train_only_and_common_pool() -> None:
    data = config()

    assert "final_target_file" not in data
    assert data["forbid_final_target_input_during_training"] is True
    assert data["common_train_count"] == 2000
    assert data["common_val_count"] == 500
    assert data["epochs"] == 30
    assert data["primary_seed"] == 42
    assert data["stability_seeds"] == [0, 1, 2, 3, 4]
    assert data["selected_edge_ratio"] == 0.2
    assert data["checkpoint_selection"] == "fixed_final_epoch_30"
    assert data["post_test_tuning"] is False
    assert data["disabled_explanation_metrics"] == ["Edge AUROC", "Edge AP"]


def test_task28_common_train_and_val_pools_match_each_backbone_predictions() -> None:
    data = config()

    for spec in SPECS.values():
        train = read_pool(ROOT / data["common_train_file"], spec)
        val = read_pool(ROOT / data["common_val_file"], spec)
        assert len(train) == 2000
        assert len(val) == 500
        assert {target.node_id for target in train}.isdisjoint(
            {target.node_id for target in val}
        )
        assert all(target.candidate_mapping_valid for target in train)
        assert all(target.candidate_mapping_valid for target in val)


def test_task28_cli_requires_exactly_one_backbone_and_seed() -> None:
    result = run_cli("--dry-run")

    assert result.returncode != 0
    assert "--backbone" in result.stderr
    assert "--seed" in result.stderr


def test_task28_cli_dry_run_accepts_allowed_seed_set() -> None:
    assert ALLOWED_SEEDS == {0, 1, 2, 3, 4, 42}

    result = run_cli("--backbone", "gcn", "--seed", "42", "--dry-run")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run_pass"
    assert payload["backbone"] == "GCN"
    assert payload["seed"] == 42
    assert payload["train_count"] == 2000
    assert payload["validation_count"] == 500


def test_task28_cli_rejects_unknown_seed_and_final_target_argument() -> None:
    bad_seed = run_cli("--backbone", "gcn", "--seed", "99", "--dry-run")
    final_arg = run_cli(
        "--backbone",
        "gcn",
        "--seed",
        "42",
        "--final-target-file",
        "configs/targets/elliptic_xai_common_test_targets.csv",
    )

    assert bad_seed.returncode != 0
    assert final_arg.returncode != 0
    assert "final-target" in final_arg.stderr


def test_task28_debug_limits_require_explicit_flag() -> None:
    result = run_cli(
        "--backbone",
        "gcn",
        "--seed",
        "42",
        "--dry-run",
        "--debug-train-limit",
        "2",
    )

    assert result.returncode != 0
    assert "--debug-train-limit requires --allow-debug-limits" in result.stderr


def test_task28_debug_dry_run_can_limit_train_and_val_for_smoke_only() -> None:
    result = run_cli(
        "--backbone",
        "graphsage",
        "--seed",
        "0",
        "--dry-run",
        "--allow-debug-limits",
        "--debug-train-limit",
        "2",
        "--debug-val-limit",
        "1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run_pass"
    assert payload["train_count"] == 2
    assert payload["validation_count"] == 1


def test_task28_latest_epoch_state_rejects_corrupted_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_001.pt").write_text("not a torch checkpoint", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Corrupted checkpoint"):
        latest_epoch_state(tmp_path, "identity")


def test_task28_latest_epoch_state_rejects_incompatible_identity(tmp_path: Path) -> None:
    atomic_torch_save(
        tmp_path / "checkpoints" / "epoch_001.pt",
        {"config_identity": "other", "epoch_completed": 1},
    )

    with pytest.raises(RuntimeError, match="Incompatible resume checkpoint"):
        latest_epoch_state(tmp_path, "expected")


def test_task28_latest_epoch_state_resumes_highest_valid_epoch(tmp_path: Path) -> None:
    atomic_torch_save(
        tmp_path / "checkpoints" / "epoch_001.pt",
        {"config_identity": "same", "epoch_completed": 1},
    )
    atomic_torch_save(
        tmp_path / "checkpoints" / "epoch_002.pt",
        {"config_identity": "same", "epoch_completed": 2},
    )

    epoch, payload = latest_epoch_state(tmp_path, "same")

    assert epoch == 2
    assert payload is not None
    assert payload["epoch_completed"] == 2


def test_task28_config_identity_is_seed_isolated() -> None:
    data = config()
    pg_config = yaml.safe_load((ROOT / data["pgexplainer_config"]).read_text(encoding="utf-8"))
    hashes = {
        "dataset": data["dataset_sha256"],
        "common_train": data["common_train_sha256"],
        "common_val": data["common_val_sha256"],
        "gcn_checkpoint": data["gcn_checkpoint_sha256"],
        "graphsage_checkpoint": data["graphsage_checkpoint_sha256"],
    }

    seed42 = config_identity(data, pg_config, SPECS["gcn"], 42, hashes)
    seed0 = config_identity(data, pg_config, SPECS["gcn"], 0, hashes)
    sage42 = config_identity(data, pg_config, SPECS["graphsage"], 42, hashes)

    assert seed42 != seed0
    assert seed42 != sage42


def test_task28_interrupted_resume_matches_uninterrupted_tiny_fixture(
    tmp_path: Path,
) -> None:
    x = torch.tensor([[1.0, -1.0]])
    y = torch.tensor([[0.25]])

    def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        optimizer.step()

    torch.manual_seed(123)
    uninterrupted = torch.nn.Linear(2, 1)
    uninterrupted_optimizer = torch.optim.SGD(uninterrupted.parameters(), lr=0.1)
    step(uninterrupted, uninterrupted_optimizer)
    step(uninterrupted, uninterrupted_optimizer)

    torch.manual_seed(123)
    resumed = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.1)
    step(resumed, resumed_optimizer)
    atomic_torch_save(
        tmp_path / "checkpoints" / "epoch_001.pt",
        {
            "config_identity": "same",
            "epoch_completed": 1,
            "model_state_dict": resumed.state_dict(),
            "optimizer_state_dict": resumed_optimizer.state_dict(),
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_numpy": __import__("numpy").random.get_state(),
        },
    )
    _, payload = latest_epoch_state(tmp_path, "same")
    assert payload is not None
    resumed.load_state_dict(payload["model_state_dict"])
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_rng(payload, torch.device("cpu"))
    step(resumed, resumed_optimizer)

    for first, second in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(first, second)


def test_task28_completion_marker_path_is_backbone_seed_scoped() -> None:
    data = config()

    gcn_path = run_dir(data, SPECS["gcn"], 42)
    sage_path = run_dir(data, SPECS["graphsage"], 42)

    assert gcn_path != sage_path
    assert "elliptic_pg_gcn_seed42_run" in str(gcn_path)
    assert "elliptic_pg_graphsage_seed42_run" in str(sage_path)
    assert not (ROOT / gcn_path / "COMPLETE.json").exists()
