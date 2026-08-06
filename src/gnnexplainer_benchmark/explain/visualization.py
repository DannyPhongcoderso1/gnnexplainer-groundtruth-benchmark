from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def write_explanation_figure(
    path: Path,
    data: Data,
    target_node: int,
    top_edges: list[dict[str, Any]],
) -> None:
    graph = to_networkx(data, to_undirected=True)
    highlighted_edges = {
        tuple(sorted((int(row["src"]), int(row["dst"])))) for row in top_edges[:10]
    }
    neighborhood_nodes = {target_node}
    for src, dst in highlighted_edges:
        neighborhood_nodes.add(src)
        neighborhood_nodes.add(dst)
    subgraph = graph.subgraph(neighborhood_nodes).copy()
    pos = nx.spring_layout(subgraph, seed=42)
    node_colors = ["#d62728" if node == target_node else "#4c78a8" for node in subgraph.nodes]
    edge_colors = [
        "#d62728" if tuple(sorted(edge)) in highlighted_edges else "#999999"
        for edge in subgraph.edges
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=260)
    nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, width=2.0)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
