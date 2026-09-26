from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import scripts.run_elliptic_pgexplainer_training as runner
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
        "task_config": "task-config-bytes",
        "pgexplainer_config": "pg-config-bytes",
        "dataset": data["dataset_sha256"],
        "common_train": data["common_train_sha256"],
        "common_val": data["common_val_sha256"],
        "gcn_checkpoint": data["gcn_checkpoint_sha256"],
        "graphsage_checkpoint": data["graphsage_checkpoint_sha256"],
    }

    seed42 = config_identity(data, pg_config, SPECS["gcn"], 42, hashes)
    seed0 = config_identity(data, pg_config, SPECS["gcn"], 0, hashes)
    sage42 = config_identity(data, pg_config, SPECS["graphsage"], 42, hashes)
    debug42 = config_identity(
        data,
        pg_config,
        SPECS["gcn"],
        42,
        hashes,
        {"allow_debug_limits": True, "train_count": 2, "validation_count": 1},
    )

    assert seed42 != seed0
    assert seed42 != sage42
    assert seed42 != debug42


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


@pytest.fixture
def identity_inputs():
    data = config()
    pg = yaml.safe_load((ROOT / data["pgexplainer_config"]).read_text(encoding="utf-8"))
    hashes = {
        "task_config": "task",
        "pgexplainer_config": "pg",
        **{key: data[key + "_sha256"] for key in (
            "dataset", "common_train", "common_val", "gcn_checkpoint", "graphsage_checkpoint"
        )},
    }
    return data, pg, SPECS["gcn"], 42, hashes


@pytest.mark.parametrize("field", [
    "runner", "wrapper", "pg_training", "candidate_mapping", "seeding", "pyg_pgexplainer",
    "task_config", "pgexplainer_config", "dataset", "common_train", "common_val",
    "gcn_checkpoint", "git_commit", "python", "torch", "pyg", "cuda_runtime", "device_type",
])
def test_integrity_resume_binds_hard_fields(tmp_path, identity_inputs, field):
    args = deepcopy(identity_inputs)
    hard = runner.hard_compatibility(torch.device("cpu"))
    identity = config_identity(*args, compatibility=hard)
    atomic_torch_save(tmp_path / "checkpoints/epoch_001.pt", {
        "config_identity": identity, "epoch_completed": 1,
    })
    assert latest_epoch_state(tmp_path, identity)[0] == 1
    if field in hard["source_sha256"]:
        hard["source_sha256"][field] = "changed"
    elif field in hard:
        hard[field] = "changed"
    else:
        args[4][field] = "changed"
    changed = config_identity(*args, compatibility=hard)
    assert changed != identity
    with pytest.raises(RuntimeError, match="Incompatible resume"):
        latest_epoch_state(tmp_path, changed)


def test_integrity_full_config_and_gpu_provenance(identity_inputs, monkeypatch):
    hard = runner.hard_compatibility(torch.device("cpu"))
    original = config_identity(*identity_inputs, compatibility=hard)
    changed = deepcopy(identity_inputs)
    changed[0]["forbid_final_target_input_during_training"] = False
    assert config_identity(*changed, compatibility=hard) != original
    changed = deepcopy(identity_inputs)
    changed[1]["explainer"]["learning_rate"] *= 2
    assert config_identity(*changed, compatibility=hard) != original
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Different GPU")
    assert runner.hard_compatibility(torch.device("cpu")) == hard
    assert config_identity(*identity_inputs, compatibility=hard) == original


@pytest.fixture
def completed_fixture(tmp_path):
    data = config()
    data["artifacts_root"] = str(tmp_path / "artifacts")
    data["gcn_checkpoint"] = str(tmp_path / "backbone.pt")
    backbone = torch.nn.Linear(2, 2).requires_grad_(False)
    before = deepcopy(backbone.state_dict())
    atomic_torch_save(Path(data["gcn_checkpoint"]), {"out.weight": backbone.weight.detach()})
    pg = yaml.safe_load((ROOT / data["pgexplainer_config"]).read_text(encoding="utf-8"))
    hashes = {"gcn_checkpoint": runner.sha256(Path(data["gcn_checkpoint"]))}
    hard = runner.hard_compatibility(torch.device("cpu"))
    # One tiny CPU Adam update, not an Elliptic epoch or explanation run.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        algorithm = runner.build_pgexplainer_from_config(pg)
        loss = algorithm.mlp(torch.ones(1, 6)).square().mean()
        loss.backward()
        algorithm.optimizer.step()
    expected = runner.completion_metadata(data, pg, SPECS["gcn"], 42, "identity", hashes, [], [])
    payload = {
        **expected,
        "hard_compatibility": hard,
        "algorithm_state_dict": deepcopy(algorithm.state_dict()),
        "optimizer_state_dict": deepcopy(algorithm.optimizer.state_dict()),
    }
    artifact = runner.final_artifact_path(data, SPECS["gcn"], 42)
    atomic_torch_save(artifact, payload)
    digest = runner.verify_canonical_artifact(artifact, expected, SPECS["gcn"], pre_save=payload)
    manifest = {
        "status": "success", "artifact_path": str(artifact), "artifact_sha256": digest,
        **{key: expected[key] for key in (
            "config_identity", "backbone", "seed", "epochs_completed", "hashes"
        )},
    }
    runner.publish_completion(data, SPECS["gcn"], 42, expected, manifest)
    runner.exact_state(backbone.state_dict(), before, "backbone unchanged")
    return data, expected, payload, manifest, artifact


def test_integrity_completed_fast_path_and_exact_reload(completed_fixture):
    data, expected, payload, _, artifact = completed_fixture
    state = torch.get_rng_state().clone()
    result = runner.verify_completed_run(data, SPECS["gcn"], 42, expected)
    assert result["status"] == "already_complete"
    assert torch.equal(state, torch.get_rng_state())
    loaded = torch.load(artifact, weights_only=False)
    runner.exact_state(loaded, payload, "saved/reloaded")
    assert loaded["epochs_completed"] == 30
    assert loaded["algorithm_current_epoch"] == 29


@pytest.mark.parametrize("failure", [
    "missing", "hash", "corrupt", "keys", "shape", "nan", "inf", "backbone", "seed",
    "config", "epoch", "algorithm_epoch", "optimizer", "optimizer_shape",
    "marker_parse", "manifest_parse", "manifest_seed", "marker_identity", "checkpoint_hash",
])
def test_integrity_completed_rejects_invalid_artifact(completed_fixture, failure):
    data, expected, payload, manifest, artifact = completed_fixture
    marker_path = runner.run_dir(data, SPECS["gcn"], 42) / "COMPLETE.json"
    marker = runner.read_integrity_json(marker_path)
    if failure == "missing":
        artifact.unlink()
    elif failure in {"hash", "corrupt"}:
        artifact.write_bytes(b"not a torch artifact")
    elif failure == "marker_parse":
        marker_path.write_text("{broken", encoding="utf-8")
    elif failure == "manifest_parse":
        runner.manifest_path(data, SPECS["gcn"], 42).write_text("{broken", encoding="utf-8")
    elif failure == "manifest_seed":
        manifest["seed"] = 0
        runner.atomic_write_json(runner.manifest_path(data, SPECS["gcn"], 42), manifest)
    elif failure == "marker_identity":
        marker["config_identity"] = "other"
        runner.atomic_write_json(marker_path, marker)
    else:
        state = payload["algorithm_state_dict"]
        first = next(iter(state))
        if failure == "keys":
            state["unexpected"] = state.pop(first)
        elif failure == "shape":
            state[first] = torch.zeros(1)
        elif failure in {"nan", "inf"}:
            state[first].view(-1)[0] = float(failure)
        elif failure == "backbone":
            payload["backbone"] = "GraphSAGE"
        elif failure == "seed":
            payload["seed"] = 0
        elif failure == "config":
            payload["config_identity"] = "other"
        elif failure == "checkpoint_hash":
            payload["hashes"] = {"gcn_checkpoint": "other"}
        elif failure == "epoch":
            payload["epochs_completed"] = 29
        elif failure == "algorithm_epoch":
            payload["algorithm_current_epoch"] = 28
        elif failure == "optimizer":
            payload["optimizer_state_dict"]["state"] = {}
        elif failure == "optimizer_shape":
            payload["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.ones(1)
        atomic_torch_save(artifact, payload)
    if failure not in {
        "missing", "hash", "marker_parse", "manifest_parse", "manifest_seed", "marker_identity",
    }:
        # Re-sign the bytes to reach structural validation, not just the hash guard.
        marker["artifact_sha256"] = runner.sha256(artifact)
        manifest["artifact_sha256"] = marker["artifact_sha256"]
        runner.atomic_write_json(marker_path, marker)
        runner.atomic_write_json(runner.manifest_path(data, SPECS["gcn"], 42), manifest)
    with pytest.raises(RuntimeError, match="Integrity failure"):
        runner.verify_completed_run(data, SPECS["gcn"], 42, expected)


def test_integrity_marker_written_last_and_failed_reload_stays_incomplete(
    completed_fixture, monkeypatch,
):
    data, expected, payload, manifest, artifact = completed_fixture
    marker_path = runner.run_dir(data, SPECS["gcn"], 42) / "COMPLETE.json"
    marker_path.unlink()
    events = []
    original = runner.atomic_write_json

    def record(path, value):
        events.append(path.name)
        original(path, value)

    monkeypatch.setattr(runner, "atomic_write_json", record)
    runner.publish_completion(data, SPECS["gcn"], 42, expected, manifest)
    assert events == [runner.manifest_path(data, SPECS["gcn"], 42).name, "COMPLETE.json"]
    marker_path.unlink()
    artifact.write_bytes(b"broken")
    events.clear()
    with pytest.raises(RuntimeError, match="Integrity failure"):
        runner.publish_completion(data, SPECS["gcn"], 42, expected, manifest)
    assert not marker_path.exists()
    assert not events
    atomic_torch_save(artifact, payload)
    mutated = deepcopy(payload)
    next(iter(mutated["algorithm_state_dict"].values())).add_(1)
    atomic_torch_save(artifact, mutated)
    with pytest.raises(RuntimeError, match="unequal tensor"):
        runner.verify_canonical_artifact(artifact, expected, SPECS["gcn"], pre_save=payload)
    assert not marker_path.exists()


def test_integrity_latest_incompatible_never_falls_back(tmp_path):
    for epoch, identity in [(1, "matching"), (2, "incompatible")]:
        atomic_torch_save(tmp_path / "checkpoints" / f"epoch_{epoch:03d}.pt", {
            "epoch_completed": epoch, "config_identity": identity,
        })
    with pytest.raises(RuntimeError, match="Incompatible resume"):
        latest_epoch_state(tmp_path, "matching")


def test_integrity_manifest_write_failure_never_publishes_marker(completed_fixture, monkeypatch):
    data, expected, _, manifest, _ = completed_fixture
    marker_path = runner.run_dir(data, SPECS["gcn"], 42) / "COMPLETE.json"
    marker_path.unlink()

    def fail_write(path, value):
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(runner, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="simulated manifest"):
        runner.publish_completion(data, SPECS["gcn"], 42, expected, manifest)
    assert not marker_path.exists()
