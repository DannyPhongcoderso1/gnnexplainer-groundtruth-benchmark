from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from gnnexplainer_benchmark.models import GCN


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    seed: int,
    best_epoch: int,
    best_validation_metric: float,
    dataset_checksum: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "resolved_config": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_metric": best_validation_metric,
            "dataset_checksum": dataset_checksum,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def build_gcn_from_checkpoint(
    checkpoint: dict[str, Any],
    in_channels: int,
    out_channels: int,
) -> GCN:
    model_config = checkpoint["resolved_config"]["model"]
    model = GCN(
        in_channels=in_channels,
        hidden_channels=int(model_config["hidden_channels"]),
        out_channels=out_channels,
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        cached=bool(model_config["cached"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
