from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch_geometric
import yaml
from sklearn.utils.random import sample_without_replacement
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from gnnexplainer_benchmark.data.serialization import save_data_artifact, save_metadata
from gnnexplainer_benchmark.data.splitting import (
    SplitConfig,
    make_stratified_node_splits,
    split_counts,
)
from gnnexplainer_benchmark.data.validation import validate_shapeggen_data
from gnnexplainer_benchmark.utils.reproducibility import seed_everything


def load_shapeggen_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "dataset" not in config:
        raise ValueError("Config must contain a top-level 'dataset' mapping")
    return config


def generate_shapeggen_data(config: dict[str, Any]) -> Data:
    dataset_config = config["dataset"]
    seed = int(dataset_config["seed"])
    seed_everything(seed)

    graph = _build_bound_graph(
        num_subgraphs=int(dataset_config["num_subgraphs"]),
        prob_connection=float(dataset_config["prob_connection"]),
        subgraph_size=int(dataset_config["subgraph_size"]),
        seed=seed,
    )
    if bool(dataset_config.get("verify", True)):
        _verify_planted_house_motifs(graph)

    labels = torch.tensor(
        [int(graph.nodes[node]["shapes_in_khop"]) - 1 for node in sorted(graph.nodes)],
        dtype=torch.long,
    )
    features, feature_mask = _make_structured_feature(
        labels,
        n_features=int(dataset_config["n_features"]),
        n_informative=int(dataset_config["n_informative"]),
        n_clusters_per_class=int(dataset_config["n_clusters_per_class"]),
        class_sep=float(dataset_config["class_sep"]),
        flip_y=float(dataset_config.get("flip_y", 0.01)),
        seed=seed,
    )

    undirected_edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges)
    edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=graph.number_of_nodes())

    model_layers = int(dataset_config["model_layers"])
    node_mask, edge_mask, candidate_edge_mask = _build_target_ground_truth_masks(
        graph,
        edge_index,
        model_layers=model_layers,
    )

    split_config = dataset_config.get("split", {})
    train_mask, val_mask, test_mask = make_stratified_node_splits(
        labels,
        seed=seed,
        config=SplitConfig(
            train_ratio=float(split_config.get("train_ratio", 0.6)),
            val_ratio=float(split_config.get("val_ratio", 0.2)),
            test_ratio=float(split_config.get("test_ratio", 0.2)),
        ),
    )

    data = Data(x=features, y=labels, edge_index=edge_index)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.node_mask = node_mask
    data.edge_mask = edge_mask
    data.candidate_edge_mask = candidate_edge_mask
    data.shape = torch.tensor(
        [int(graph.nodes[node]["shape"]) for node in sorted(graph.nodes)],
        dtype=torch.long,
    )
    data.shapes_in_khop = torch.tensor(
        [int(graph.nodes[node]["shapes_in_khop"]) for node in sorted(graph.nodes)],
        dtype=torch.long,
    )
    data.feature_mask = feature_mask
    data.model_layers = model_layers
    data.ground_truth_semantics = (
        "Target-specific node masks mark motif nodes in the target 1-hop motif set "
        "within the model-layer computation subgraph. Edge masks mark directed "
        "candidate edges whose endpoints are both important motif nodes, plus "
        "source-to-motif edges when the target node itself is not in the motif."
    )
    validate_shapeggen_data(data)
    return data


def build_metadata(config: dict[str, Any], data: Data, artifact_sha256: str) -> dict[str, Any]:
    labels, counts = torch.unique(data.y, sorted=True, return_counts=True)
    label_distribution = {
        str(int(label.item())): int(count.item())
        for label, count in zip(labels, counts, strict=True)
    }
    positive_edge_counts = data.edge_mask.sum(dim=1)
    candidate_edge_counts = data.candidate_edge_mask.sum(dim=1)
    eligible_targets = (positive_edge_counts > 0) & (candidate_edge_counts > positive_edge_counts)
    return {
        "config": deepcopy(config),
        "seed": int(config["dataset"]["seed"]),
        "source": {
            "name": "GraphXAI ShapeGGen",
            "repository": config["dataset"]["source_repository"],
            "commit": config["dataset"]["source_commit"],
            "local_package_installed": False,
            "implementation": "local source-aligned implementation",
        },
        "protocol": {
            "name": config["dataset"]["protocol_name"],
            "split_policy": "stratified_disjoint",
            "feature_explanations_in_core_metrics": False,
            "candidate_edge_representation": "directed PyG edge_index with reverse-edge copies",
        },
        "shapes": {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
            "num_undirected_edges": int(data.edge_index.size(1) // 2),
            "num_node_features": int(data.x.size(1)),
            "x": list(data.x.shape),
            "edge_index": list(data.edge_index.shape),
            "y": list(data.y.shape),
            "node_mask": list(data.node_mask.shape),
            "edge_mask": list(data.edge_mask.shape),
            "candidate_edge_mask": list(data.candidate_edge_mask.shape),
        },
        "label_distribution": label_distribution,
        "ground_truth_counts": {
            "positive_node_entries": int(data.node_mask.sum().item()),
            "positive_edge_entries": int(data.edge_mask.sum().item()),
            "candidate_edge_entries": int(data.candidate_edge_mask.sum().item()),
            "eligible_targets": int(eligible_targets.sum().item()),
            "min_positive_edges_per_target": int(positive_edge_counts.min().item()),
            "max_positive_edges_per_target": int(positive_edge_counts.max().item()),
            "min_candidate_edges_per_target": int(candidate_edge_counts.min().item()),
            "max_candidate_edges_per_target": int(candidate_edge_counts.max().item()),
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


def generate_and_save_shapeggen(config_path: Path, output_path: Path, metadata_path: Path) -> Data:
    config = load_shapeggen_config(config_path)
    data = generate_shapeggen_data(config)
    artifact_sha256 = save_data_artifact(data, output_path)
    metadata = build_metadata(config, data, artifact_sha256)
    save_metadata(metadata, metadata_path)
    return data


def _house_graph() -> nx.Graph:
    return nx.house_graph()


def _build_bound_graph(
    *,
    num_subgraphs: int,
    prob_connection: float,
    subgraph_size: int,
    seed: int,
) -> nx.Graph:
    if not 0 <= prob_connection <= 1:
        raise ValueError("prob_connection must be in [0, 1]")

    rng = np.random.default_rng(seed)
    shape = _house_graph()
    nodes_in_shape = shape.number_of_nodes()
    subgraphs: list[nx.Graph] = []
    original_shapes: list[nx.Graph] = []
    next_node = 0

    for shape_number in range(1, num_subgraphs + 1):
        current_shape = nx.relabel_nodes(
            shape.copy(),
            {node: next_node + node for node in shape.nodes},
        )
        nx.set_node_attributes(current_shape, shape_number, "shape")
        original_shapes.append(current_shape.copy())

        add_size = int(rng.poisson(lam=subgraph_size - nodes_in_shape))
        grown = _ba_around_shape(current_shape, add_size=add_size, rng=rng)
        nx.set_node_attributes(grown, 1, "shapes_in_khop")
        subgraphs.append(grown.copy())
        next_node = max(grown.nodes) + 1

    graph = nx.Graph()
    for subgraph in subgraphs:
        graph.add_edges_from(subgraph.edges)
        graph.add_nodes_from(subgraph.nodes(data=True))

    subgraph_pairs = np.array(
        [(i, j) for i in range(num_subgraphs) for j in range(i + 1, num_subgraphs)]
    )
    sample_mask = rng.binomial(n=2, p=prob_connection, size=len(subgraph_pairs)).astype(bool)

    for i, j in subgraph_pairs[sample_mask]:
        _try_add_valid_inter_subgraph_edge(
            graph=graph,
            left=subgraphs[int(i)],
            right=subgraphs[int(j)],
            left_motif=original_shapes[int(i)],
            right_motif=original_shapes[int(j)],
            rng=rng,
        )

    largest_component = sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    graph = graph.subgraph(largest_component).copy()
    mapping = {node: idx for idx, node in enumerate(graph.nodes)}
    return nx.relabel_nodes(graph, mapping=mapping, copy=True)


def _ba_around_shape(shape: nx.Graph, *, add_size: int, rng: np.random.Generator) -> nx.Graph:
    original_nodes = list(shape.nodes)
    top_node = max(original_nodes)

    for offset in range(add_size):
        degrees = np.array([shape.degree[node] for node in original_nodes], dtype=float)
        probabilities = degrees / degrees.sum()
        connect_node = int(rng.choice(original_nodes, p=probabilities))
        new_node = top_node + offset + 1
        shape.add_node(new_node, shape=0)
        shape.add_edge(connect_node, new_node)
    return shape


def _try_add_valid_inter_subgraph_edge(
    *,
    graph: nx.Graph,
    left: nx.Graph,
    right: nx.Graph,
    left_motif: nx.Graph,
    right_motif: nx.Graph,
    rng: np.random.Generator,
) -> None:
    possible_edges = [(u, v) for u in left.nodes for v in right.nodes]
    weights = np.array([left.degree[u] + right.degree[v] for u, v in possible_edges], dtype=float)
    active = np.ones(len(possible_edges), dtype=bool)

    while active.any() and weights.sum() > 0:
        probabilities = weights / weights.sum()
        edge_idx = int(rng.choice(np.arange(len(possible_edges)), p=probabilities))
        candidate_edge = possible_edges[edge_idx]
        old_weight = weights[edge_idx]
        active[edge_idx] = False
        weights[edge_idx] = 0.0
        if active.any():
            weights[active] += old_weight / active.sum()

        graph.add_edge(*candidate_edge)
        nodes_to_search: set[int] = set()
        for motif_node in list(left_motif.nodes) + list(right_motif.nodes):
            nodes_to_search.update(_khop_nodes(graph, motif_node, num_hops=1))
        updates = _unique_shape_updates(graph, nodes_to_search, lower_bound=1, upper_bound=2)
        if updates is None:
            graph.remove_edge(*candidate_edge)
            continue
        for node, (num_unique, unique_shapes) in updates.items():
            graph.nodes[node]["shapes_in_khop"] = num_unique
            graph.nodes[node]["nearby_shapes"] = unique_shapes
        return


def _unique_shape_updates(
    graph: nx.Graph,
    nodes_to_search: set[int],
    *,
    lower_bound: int,
    upper_bound: int,
) -> dict[int, tuple[int, tuple[int, ...]]] | None:
    updates: dict[int, tuple[int, tuple[int, ...]]] = {}
    for node in nodes_to_search:
        unique_shapes = sorted(
            {
                int(graph.nodes[khop_node]["shape"])
                for khop_node in _khop_nodes(graph, node, num_hops=1)
                if int(graph.nodes[khop_node]["shape"]) != 0
            }
        )
        num_unique = len(unique_shapes)
        if num_unique < lower_bound or num_unique > upper_bound:
            return None
        updates[node] = (num_unique, tuple(unique_shapes))
    return updates


def _verify_planted_house_motifs(graph: nx.Graph) -> None:
    motif_node_groups: dict[int, list[int]] = {}
    for node, attrs in graph.nodes(data=True):
        shape_id = int(attrs["shape"])
        if shape_id != 0:
            motif_node_groups.setdefault(shape_id, []).append(int(node))

    for shape_id, nodes in motif_node_groups.items():
        if len(nodes) != 5:
            raise ValueError(f"Shape {shape_id} has {len(nodes)} nodes instead of 5")
        subgraph = graph.subgraph(nodes)
        if not nx.is_isomorphic(subgraph, _house_graph()):
            raise ValueError(f"Shape {shape_id} is not isomorphic to the house motif")

    labels = [int(attrs["shapes_in_khop"]) for _, attrs in graph.nodes(data=True)]
    if min(labels) < 1 or max(labels) > 2:
        raise ValueError("ShapeGGen labels must encode one or two nearby motifs")


def _build_target_ground_truth_masks(
    graph: nx.Graph,
    edge_index: torch.Tensor,
    *,
    model_layers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = graph.number_of_nodes()
    num_edges = edge_index.size(1)
    node_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    edge_mask = torch.zeros((num_nodes, num_edges), dtype=torch.bool)
    candidate_edge_mask = torch.zeros((num_nodes, num_edges), dtype=torch.bool)

    directed_edges = [
        (int(edge_index[0, idx]), int(edge_index[1, idx])) for idx in range(num_edges)
    ]
    for target in range(num_nodes):
        one_hop_shapes = {
            int(graph.nodes[node]["shape"])
            for node in _khop_nodes(graph, target, num_hops=1)
            if int(graph.nodes[node]["shape"]) != 0
        }
        khop = set(_khop_nodes(graph, target, num_hops=model_layers))
        important_nodes = {
            node for node in khop if int(graph.nodes[node]["shape"]) in one_hop_shapes
        }
        node_mask[target, list(important_nodes)] = True

        for edge_pos, (src, dst) in enumerate(directed_edges):
            if src not in khop or dst not in khop:
                continue
            candidate_edge_mask[target, edge_pos] = True
            endpoints_are_motif = src in important_nodes and dst in important_nodes
            source_to_motif = (
                target not in important_nodes
                and (
                    (src == target and dst in important_nodes)
                    or (dst == target and src in important_nodes)
                )
            )
            if endpoints_are_motif or source_to_motif:
                edge_mask[target, edge_pos] = True

    return node_mask, edge_mask, candidate_edge_mask


def _khop_nodes(graph: nx.Graph, node_idx: int, *, num_hops: int) -> list[int]:
    seen = {node_idx}
    queue: deque[tuple[int, int]] = deque([(node_idx, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth == num_hops:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor in seen:
                continue
            seen.add(int(neighbor))
            queue.append((int(neighbor), depth + 1))
    return sorted(seen)


def _generate_hypercube(samples: int, dimensions: int, rng: np.random.RandomState) -> np.ndarray:
    if dimensions > 30:
        return np.hstack(
            [
                rng.randint(2, size=(samples, dimensions - 30)),
                _generate_hypercube(samples, 30, rng),
            ]
        )
    out = sample_without_replacement(2**dimensions, samples, random_state=rng).astype(
        dtype=">u4",
        copy=False,
    )
    return np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]


def _make_structured_feature(
    labels: torch.Tensor,
    *,
    n_features: int,
    n_informative: int,
    n_clusters_per_class: int,
    class_sep: float,
    flip_y: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GraphXAI-style structured features adapted from its public implementation."""

    original_labels = labels.clone().numpy()
    generated_labels = labels.clone().numpy()
    n_samples = generated_labels.shape[0]
    unique_labels, n_samples_per_class = np.unique(generated_labels, return_counts=True)
    n_classes = len(unique_labels)
    rng = np.random.RandomState(seed)

    if n_informative > n_features:
        raise ValueError("n_informative must not exceed n_features")
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        raise ValueError("n_informative is too small for the number of classes and clusters")

    n_useless = n_features - n_informative
    n_clusters = n_classes * n_clusters_per_class
    n_samples_per_cluster = [
        int(n_samples_per_class[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]
    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    features = np.zeros((n_samples, n_features))
    centroids = _generate_hypercube(n_clusters, n_informative, rng).astype(float, copy=False)
    centroids *= 2 * class_sep
    centroids -= class_sep
    features[:, :n_informative] = rng.randn(n_samples, n_informative)

    stop = 0
    for cluster_idx, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[cluster_idx]
        generated_labels[start:stop] = cluster_idx % n_classes
        cluster_features = features[start:stop, :n_informative]
        covariance = 2 * rng.rand(n_informative, n_informative) - 1
        cluster_features[...] = np.dot(cluster_features, covariance)
        cluster_features += centroid

    if n_useless > 0:
        features[:, -n_useless:] = rng.randn(n_samples, n_useless)

    flip_mask = rng.rand(n_samples) < flip_y
    generated_labels[flip_mask] = rng.randint(n_classes, size=flip_mask.sum())
    feature_mask = np.zeros(n_features, dtype=bool)
    feature_mask[:n_informative] = True

    feature_indices = np.arange(n_features)
    rng.shuffle(feature_indices)
    features[:, :] = features[:, feature_indices]
    feature_mask[:] = feature_mask[feature_indices]

    reordered_features = np.zeros_like(features)
    for label in np.sort(np.unique(original_labels)):
        generated_idx = np.argwhere(generated_labels == label).flatten()
        original_idx = np.argwhere(original_labels == label).flatten()
        for generated_pos, original_pos in zip(generated_idx, original_idx, strict=False):
            reordered_features[original_pos, :] = features[generated_pos, :]

    return torch.from_numpy(reordered_features).float(), torch.from_numpy(feature_mask)
