from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch_geometric
import yaml
from torch_geometric.data import Data

from gnnexplainer_benchmark.data.serialization import save_data_artifact, save_metadata
from gnnexplainer_benchmark.data.splitting import (
    SplitConfig,
    make_stratified_node_splits,
    split_counts,
)
from gnnexplainer_benchmark.data.validation import validate_shapeggen_data
from gnnexplainer_benchmark.utils.reproducibility import seed_everything
from graphxai.datasets.shape_graph import ShapeGGen

GRAPHXAI_REPOSITORY = "https://github.com/mims-harvard/GraphXAI"
GRAPHXAI_COMMIT = "a11e65ffbc4df737f35522a8accf2283a8aeaa37"
GRAPHXAI_LICENSE = "MIT"
GRAPHXAI_SHAPEGGEN_SOURCE = "graphxai/datasets/shape_graph.py"
GRAPHXAI_IMPORT_PATH = "graphxai.datasets.shape_graph.ShapeGGen"

UPSTREAM_SHAPEGGEN_FIELDS = {
    "model_layers",
    "shape",
    "seed",
    "make_explanations",
    "variant",
    "num_subgraphs",
    "prob_connection",
    "subgraph_size",
    "verify",
    "max_tries_verification",
    "n_informative",
    "class_sep",
    "n_features",
    "n_clusters_per_class",
    "homophily_coef",
    "add_sensitive_feature",
    "attribute_sensitive_feature",
    "sens_attribution_noise",
}
LOCAL_ADAPTER_FIELDS = {
    "split",
    "edge_representation",
    "ground_truth_storage",
    "feature_explanations_in_core_metrics",
}
METADATA_FIELDS = {
    "dataset_name",
    "protocol_name",
    "upstream_project",
    "upstream_repository",
    "upstream_commit",
    "upstream_license",
    "upstream_shape_ggen_source",
    "upstream_import_path",
    "historical_local_task21_commit",
    "historical_local_task21_sha256",
}


def load_shapeggen_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "dataset" not in config:
        raise ValueError("Config must contain a top-level 'dataset' mapping")
    validate_shapeggen_config(config)
    return config


def validate_shapeggen_config(config: dict[str, Any]) -> None:
    dataset_config = config.get("dataset")
    if not isinstance(dataset_config, dict):
        raise ValueError("Config must contain a top-level 'dataset' mapping")
    required_sections = {"metadata", "upstream_generator", "local_adapter"}
    unknown_sections = set(dataset_config) - required_sections
    missing_sections = required_sections - set(dataset_config)
    if unknown_sections:
        raise ValueError(f"Unsupported ShapeGGen configuration section: {sorted(unknown_sections)}")
    if missing_sections:
        raise ValueError(f"Missing ShapeGGen configuration section: {sorted(missing_sections)}")

    _validate_fields(dataset_config["metadata"], METADATA_FIELDS, section="metadata")
    _validate_fields(
        dataset_config["upstream_generator"],
        UPSTREAM_SHAPEGGEN_FIELDS,
        section="upstream_generator",
    )
    _validate_fields(dataset_config["local_adapter"], LOCAL_ADAPTER_FIELDS, section="local_adapter")


def generate_shapeggen_data(config: dict[str, Any]) -> Data:
    validate_shapeggen_config(config)
    upstream_config = dict(config["dataset"]["upstream_generator"])
    local_config = config["dataset"]["local_adapter"]
    seed = int(upstream_config["seed"])
    seed_everything(seed)

    upstream_dataset = generate_upstream_shapeggen(upstream_config)
    data = convert_graphxai_to_pyg(upstream_dataset)
    attach_local_splits(data, local_config, fallback_seed=seed)
    data.graphxai_upstream_commit = GRAPHXAI_COMMIT
    data.graphxai_upstream_source = GRAPHXAI_SHAPEGGEN_SOURCE
    data.shape_ggen_protocol_name = config["dataset"]["metadata"]["protocol_name"]
    validate_shapeggen_data(data)
    return data


def generate_upstream_shapeggen(upstream_config: dict[str, Any]) -> ShapeGGen:
    _validate_fields(upstream_config, UPSTREAM_SHAPEGGEN_FIELDS, section="upstream_generator")
    constructor_args = {
        "model_layers": int(upstream_config["model_layers"]),
        "shape": upstream_config["shape"],
        "seed": int(upstream_config["seed"]),
        "make_explanations": bool(upstream_config["make_explanations"]),
    }
    kwargs = {
        key: value
        for key, value in upstream_config.items()
        if key not in constructor_args
    }
    dataset = ShapeGGen(**constructor_args, **kwargs)
    if dataset.explanations is None:
        raise ValueError("Official GraphXAI ShapeGGen did not return explanations")
    return dataset


def convert_graphxai_to_pyg(upstream_dataset: ShapeGGen) -> Data:
    upstream_graph = upstream_dataset.graph
    data = Data(
        x=upstream_graph.x.detach().cpu().clone(),
        y=upstream_graph.y.detach().cpu().clone(),
        edge_index=upstream_graph.edge_index.detach().cpu().clone(),
    )
    if hasattr(upstream_graph, "shape"):
        data.shape = upstream_graph.shape.detach().cpu().clone().long()
    data.shapes_in_khop = torch.tensor(
        [
            int(upstream_dataset.G.nodes[node]["shapes_in_khop"])
            for node in sorted(upstream_dataset.G.nodes)
        ],
        dtype=torch.long,
    )
    (
        data.node_mask,
        data.edge_mask,
        data.candidate_edge_mask,
        data.feature_mask,
    ) = convert_graphxai_ground_truth(upstream_dataset, data)
    data.model_layers = int(upstream_dataset.model_layers)
    data.ground_truth_semantics = (
        "Official GraphXAI ShapeGGen target-specific Explanation objects converted "
        "to dense local masks. edge_mask[target] and candidate_edge_mask[target] "
        "are aligned to the full directed PyG edge_index."
    )
    return data


def convert_graphxai_ground_truth(
    upstream_dataset: ShapeGGen,
    data: Data,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.size(1))
    node_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    edge_mask = torch.zeros((num_nodes, num_edges), dtype=torch.bool)
    candidate_edge_mask = torch.zeros((num_nodes, num_edges), dtype=torch.bool)
    feature_masks: list[torch.Tensor] = []

    for explanations in upstream_dataset.explanations:
        if len(explanations) != 1:
            raise ValueError("ShapeGGen is expected to provide exactly one explanation per node")
        explanation = explanations[0]
        target = int(explanation.node_idx)
        subgraph = explanation.enc_subgraph

        if subgraph.edge_mask.numel() != num_edges:
            raise ValueError("Upstream edge_mask is not aligned with full graph edge_index")
        if subgraph.edge_index.size(1) != explanation.edge_imp.numel():
            raise ValueError("Upstream edge_imp is not aligned with enclosing subgraph edge_index")
        if subgraph.nodes.numel() != explanation.node_imp.numel():
            raise ValueError("Upstream node_imp is not aligned with enclosing subgraph nodes")

        target_candidate = subgraph.edge_mask.detach().cpu().bool()
        candidate_edge_mask[target] = target_candidate
        global_edge_ids = torch.nonzero(target_candidate, as_tuple=False).view(-1)
        edge_mask[target, global_edge_ids] = explanation.edge_imp.detach().cpu().bool()
        node_ids = subgraph.nodes.detach().cpu().long()
        node_mask[target, node_ids] = explanation.node_imp.detach().cpu().bool()
        feature_masks.append(explanation.feature_imp.detach().cpu().bool())

    feature_mask = _common_feature_mask(feature_masks)
    return node_mask, edge_mask, candidate_edge_mask, feature_mask


def attach_local_splits(data: Data, local_config: dict[str, Any], *, fallback_seed: int) -> None:
    split_config = local_config["split"]
    method = str(split_config["method"])
    if method != "stratified_disjoint":
        raise ValueError(f"Unsupported ShapeGGen split method: {method}")
    split_seed = int(split_config.get("seed", fallback_seed))
    train_mask, val_mask, test_mask = make_stratified_node_splits(
        data.y,
        seed=split_seed,
        config=SplitConfig(
            train_ratio=float(split_config["train_ratio"]),
            val_ratio=float(split_config["val_ratio"]),
            test_ratio=float(split_config["test_ratio"]),
        ),
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.split_seed = split_seed
    data.split_method = method


def get_target_edge_ground_truth(data: Data, node_id: int) -> torch.Tensor:
    if data.edge_mask.dim() == 1:
        return data.edge_mask.detach().cpu().bool()
    if data.edge_mask.dim() == 2:
        return data.edge_mask[int(node_id)].detach().cpu().bool()
    raise ValueError("Unsupported edge_mask shape")


def get_target_candidate_edge_mask(data: Data, node_id: int) -> torch.Tensor:
    if hasattr(data, "candidate_edge_mask") and data.candidate_edge_mask.dim() == 2:
        return data.candidate_edge_mask[int(node_id)].detach().cpu().bool()
    raise ValueError("ShapeGGen requires target-specific candidate_edge_mask")


def build_metadata(config: dict[str, Any], data: Data, artifact_sha256: str) -> dict[str, Any]:
    validate_shapeggen_config(config)
    labels, counts = torch.unique(data.y, sorted=True, return_counts=True)
    label_distribution = {
        str(int(label.item())): int(count.item())
        for label, count in zip(labels, counts, strict=True)
    }
    positive_edge_counts = data.edge_mask.sum(dim=1)
    candidate_edge_counts = data.candidate_edge_mask.sum(dim=1)
    eligible_targets = (positive_edge_counts > 0) & (candidate_edge_counts > positive_edge_counts)
    metadata_config = deepcopy(config["dataset"]["metadata"])
    local_config = config["dataset"]["local_adapter"]
    return {
        "dataset_name": metadata_config["dataset_name"],
        "protocol_name": metadata_config["protocol_name"],
        "upstream_project": metadata_config["upstream_project"],
        "upstream_repository": metadata_config["upstream_repository"],
        "upstream_commit_or_version": metadata_config["upstream_commit"],
        "upstream_license": metadata_config["upstream_license"],
        "upstream_shape_ggen_source": metadata_config["upstream_shape_ggen_source"],
        "upstream_import_path": metadata_config["upstream_import_path"],
        "dependency_strategy": "vendored unmodified official GraphXAI source",
        "generator_config": deepcopy(config["dataset"]["upstream_generator"]),
        "generator_seed": int(config["dataset"]["upstream_generator"]["seed"]),
        "local_adapter_config": deepcopy(local_config),
        "local_config_path": "configs/dataset/shapeggen.yaml",
        "generation_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\generate_shapeggen.py "
            "--config configs\\dataset\\shapeggen.yaml "
            "--output artifacts\\data\\shapeggen.pt "
            "--metadata artifacts\\data\\shapeggen_manifest.json"
        ),
        "historical_local_task21_commit": metadata_config["historical_local_task21_commit"],
        "historical_local_task21_sha256": metadata_config["historical_local_task21_sha256"],
        "shapes": {
            "num_nodes": int(data.num_nodes),
            "num_undirected_edges": _unique_unordered_edge_count(data),
            "num_directed_edge_entries": int(data.edge_index.size(1)),
            "num_node_features": int(data.x.size(1)),
            "x": list(data.x.shape),
            "edge_index": list(data.edge_index.shape),
            "y": list(data.y.shape),
            "node_mask": list(data.node_mask.shape),
            "edge_mask": list(data.edge_mask.shape),
            "candidate_edge_mask": list(data.candidate_edge_mask.shape),
            "feature_mask": list(data.feature_mask.shape),
        },
        "label_distribution": label_distribution,
        "split": {
            "method": str(data.split_method),
            "seed": int(data.split_seed),
            "counts": split_counts(data.train_mask, data.val_mask, data.test_mask),
        },
        "ground_truth": {
            "representation": (
                "target-specific dense boolean masks derived from official GraphXAI "
                "Explanation objects"
            ),
            "edge_mask": "shape [num_nodes, num_directed_edge_entries]",
            "node_mask": "shape [num_nodes, num_nodes]",
            "feature_mask": "shape [num_features], global common upstream feature_imp",
            "positive_node_entries": int(data.node_mask.sum().item()),
            "positive_edge_entries": int(data.edge_mask.sum().item()),
            "candidate_edge_entries": int(data.candidate_edge_mask.sum().item()),
            "eligible_targets": int(eligible_targets.sum().item()),
            "min_positive_edges_per_target": int(positive_edge_counts.min().item()),
            "max_positive_edges_per_target": int(positive_edge_counts.max().item()),
            "min_candidate_edges_per_target": int(candidate_edge_counts.min().item()),
            "max_candidate_edges_per_target": int(candidate_edge_counts.max().item()),
        },
        "versions": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
        "sha256": artifact_sha256,
    }


def generate_and_save_shapeggen(config_path: Path, output_path: Path, metadata_path: Path) -> Data:
    config = load_shapeggen_config(config_path)
    data = generate_shapeggen_data(config)
    artifact_sha256 = save_data_artifact(data, output_path)
    metadata = build_metadata(config, data, artifact_sha256)
    save_metadata(metadata, metadata_path)
    return data


def _validate_fields(values: Any, allowed_fields: set[str], *, section: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"ShapeGGen config section must be a mapping: {section}")
    unknown = set(values) - allowed_fields
    if unknown:
        raise ValueError(
            f"Unsupported ShapeGGen configuration field in {section}: {sorted(unknown)}"
        )


def _common_feature_mask(feature_masks: list[torch.Tensor]) -> torch.Tensor:
    if not feature_masks:
        raise ValueError("No upstream feature explanations were returned")
    first = feature_masks[0]
    for mask in feature_masks[1:]:
        if not torch.equal(mask, first):
            raise ValueError("ShapeGGen feature explanations are not globally consistent")
    return first


def _unique_unordered_edge_count(data: Data) -> int:
    return len(
        {
            tuple(sorted((int(data.edge_index[0, idx]), int(data.edge_index[1, idx]))))
            for idx in range(data.edge_index.size(1))
        }
    )
