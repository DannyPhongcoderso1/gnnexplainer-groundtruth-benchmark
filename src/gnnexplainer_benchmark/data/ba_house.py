from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_geometric
import yaml
from torch_geometric.data import Data
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif

from gnnexplainer_benchmark.data.serialization import save_data_artifact, save_metadata
from gnnexplainer_benchmark.data.splitting import (
    SplitConfig,
    make_stratified_node_splits,
    split_counts,
)
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.utils.reproducibility import seed_everything


def load_ba_house_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "dataset" not in config:
        raise ValueError("Config must contain a top-level 'dataset' mapping")
    return config


def generate_ba_house_data(config: dict[str, Any]) -> Data:
    dataset_config = config["dataset"]
    seed = int(dataset_config["seed"])
    seed_everything(seed)

    dataset = ExplainerDataset(
        graph_generator=BAGraph(
            num_nodes=int(dataset_config["background_nodes"]),
            num_edges=int(dataset_config["ba_attachment_parameter"]),
        ),
        motif_generator=HouseMotif(),
        num_motifs=int(dataset_config["num_motifs"]),
    )
    data = dataset[0]

    if bool(dataset_config.get("constant_node_features", True)):
        feature_dim = int(dataset_config.get("feature_dim", 10))
        data.x = torch.ones((data.num_nodes, feature_dim), dtype=torch.float32)

    split_config = dataset_config.get("split", {})
    train_mask, val_mask, test_mask = make_stratified_node_splits(
        data.y,
        seed=seed,
        config=SplitConfig(
            train_ratio=float(split_config.get("train_ratio", 0.6)),
            val_ratio=float(split_config.get("val_ratio", 0.2)),
            test_ratio=float(split_config.get("test_ratio", 0.2)),
        ),
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    validate_ba_house_data(data)
    return data


def build_metadata(config: dict[str, Any], data: Data, artifact_sha256: str) -> dict[str, Any]:
    labels, counts = torch.unique(data.y, sorted=True, return_counts=True)
    label_distribution = {
        str(int(label.item())): int(count.item())
        for label, count in zip(labels, counts, strict=True)
    }
    return {
        "config": deepcopy(config),
        "seed": int(config["dataset"]["seed"]),
        "shapes": {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
            "num_node_features": int(data.x.size(1)),
            "x": list(data.x.shape),
            "edge_index": list(data.edge_index.shape),
            "y": list(data.y.shape),
            "node_mask": list(data.node_mask.shape),
            "edge_mask": list(data.edge_mask.shape),
        },
        "label_distribution": label_distribution,
        "ground_truth_counts": {
            "nodes": int(data.node_mask.sum().item()),
            "edges": int(data.edge_mask.sum().item()),
        },
        "split_counts": split_counts(data.train_mask, data.val_mask, data.test_mask),
        "versions": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
        "sha256": artifact_sha256,
    }


def generate_and_save_ba_house(config_path: Path, output_path: Path, metadata_path: Path) -> Data:
    config = load_ba_house_config(config_path)
    data = generate_ba_house_data(config)
    artifact_sha256 = save_data_artifact(data, output_path)
    metadata = build_metadata(config, data, artifact_sha256)
    save_metadata(metadata, metadata_path)
    return data
