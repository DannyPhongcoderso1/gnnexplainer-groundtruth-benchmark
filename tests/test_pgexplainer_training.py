from __future__ import annotations

import json
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    build_pgexplainer_from_config,
    load_yaml,
    materialize_pgexplainer,
    optimizer_excludes_gcn_parameters,
    read_final_test_node_ids,
    read_pgexplainer_targets,
    verify_pgexplainer_reload,
    wrap_pgexplainer,
)
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)


def test_pgexplainer_training_config_uses_frozen_paper_aligned_coefficients() -> None:
    config = load_yaml(Path("configs/shared/pgexplainer.yaml"))
    algorithm = build_pgexplainer_from_config(config)

    assert algorithm.epochs == 30
    assert algorithm.lr == 0.003
    assert algorithm.coeffs["edge_size"] == 0.05
    assert algorithm.coeffs["edge_ent"] == 1.0
    assert algorithm.coeffs["temp"] == [5.0, 2.0]
    assert algorithm.coeffs["bias"] == 0.01


def test_pgexplainer_training_targets_do_not_include_final_test_nodes() -> None:
    train_targets = read_pgexplainer_targets(Path("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv"))
    validation_targets = read_pgexplainer_targets(
        Path("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv")
    )
    final_test_ids = read_final_test_node_ids(Path("results/metrics/explainer_test_targets.csv"))

    train_ids = {target.node_id for target in train_targets}
    validation_ids = {target.node_id for target in validation_targets}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(final_test_ids)
    assert validation_ids.isdisjoint(final_test_ids)


def test_pgexplainer_training_optimizer_excludes_frozen_gcn_parameters() -> None:
    data = load_data_artifact(Path("data/ba_house/ba500/graph.pt"))
    checkpoint = load_checkpoint(Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"))
    model = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    config = load_yaml(Path("configs/shared/pgexplainer.yaml"))
    algorithm = build_pgexplainer_from_config(config)
    wrap_pgexplainer(model, algorithm)
    materialize_pgexplainer(algorithm, model, data.x, data.edge_index, node_id=300)

    assert optimizer_excludes_gcn_parameters(algorithm, model) is True
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert any(parameter.requires_grad for parameter in algorithm.parameters())
    assert all(torch.isfinite(parameter).all().item() for parameter in algorithm.parameters())


def test_task15_manifest_records_artifact_provenance_and_no_test_leakage() -> None:
    artifact_path = Path("artifacts/checkpoints/ba_house/ba500/pgexplainer/gcn/primary_seed42.pt")
    manifest_path = Path("artifacts/explainers/pgexplainer_gcn_ba_house_seed42_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_targets = read_pgexplainer_targets(Path(manifest["train_target_path"]))
    validation_targets = read_pgexplainer_targets(Path(manifest["validation_target_path"]))
    final_test_ids = read_final_test_node_ids(Path(manifest["final_test_target_path"]))
    train_ids = {target.node_id for target in train_targets}
    validation_ids = {target.node_id for target in validation_targets}

    assert manifest["run_status"] == "success"
    assert manifest["pgexplainer_checkpoint_sha256"] == sha256_file(artifact_path)
    assert manifest["train_target_count"] == 48
    assert manifest["validation_target_count"] == 16
    assert manifest["final_test_target_count"] == 12
    assert manifest["train_target_sha256"] == sha256_file(Path(manifest["train_target_path"]))
    assert manifest["validation_target_sha256"] == sha256_file(
        Path(manifest["validation_target_path"])
    )
    assert manifest["final_test_target_sha256"] == sha256_file(
        Path(manifest["final_test_target_path"])
    )
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(final_test_ids)
    assert validation_ids.isdisjoint(final_test_ids)
    assert manifest["final_test_targets_used_for_training"] == 0
    assert manifest["model_weights_unchanged"] is True
    assert manifest["optimizer_excludes_gcn_parameters"] is True
    assert manifest["checkpoint_reload_ok"] is True
    assert manifest["explainer_training_runtime_seconds"] == manifest["training_runtime_seconds"]
    assert manifest["inference_runtime_seconds"] is None
    assert manifest["evaluation_runtime_seconds"] is None
    assert manifest["environment"]["torch_version"] == "2.11.0+cu128"
    assert manifest["environment"]["torch_geometric_version"] == "2.8.0.post1"


def test_task15_pgexplainer_checkpoint_reload_path_is_supported() -> None:
    data = load_data_artifact(Path("data/ba_house/ba500/graph.pt"))
    checkpoint = load_checkpoint(Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"))
    model = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    model.eval()
    config = load_yaml(Path("configs/shared/pgexplainer.yaml"))

    assert verify_pgexplainer_reload(
        Path("artifacts/checkpoints/ba_house/ba500/pgexplainer/gcn/primary_seed42.pt"),
        config,
        model,
        data.x,
        data.edge_index,
        materialization_node_id=325,
    )
