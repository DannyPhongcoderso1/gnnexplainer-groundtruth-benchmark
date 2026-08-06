from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from gnnexplainer_benchmark.training.trainer import build_gcn_model, load_training_config


def test_checkpoint_reload_preserves_logits(tmp_path: Path) -> None:
    data = generate_ba_house_data(load_ba_house_config(Path("configs/dataset/ba_house.yaml")))
    config = load_training_config(Path("configs/model/gcn.yaml"))
    model = build_gcn_model(data, config)
    model.eval()
    with torch.no_grad():
        expected_logits = model(data.x, data.edge_index)

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        deepcopy(config),
        seed=42,
        best_epoch=1,
        best_validation_metric=0.5,
        dataset_checksum="test-checksum",
    )
    checkpoint = load_checkpoint(checkpoint_path)
    reloaded = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    reloaded.eval()
    with torch.no_grad():
        actual_logits = reloaded(data.x, data.edge_index)

    assert torch.allclose(actual_logits, expected_logits)
