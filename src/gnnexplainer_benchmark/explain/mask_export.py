from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from gnnexplainer_benchmark.explain.node_selection import TargetNode
from gnnexplainer_benchmark.explain.visualization import write_explanation_figure


def validate_explanation_masks(explanation: Any, data: Data) -> None:
    if not hasattr(explanation, "edge_mask") or explanation.edge_mask is None:
        raise ValueError("Explanation is missing edge_mask")
    if not hasattr(explanation, "node_mask") or explanation.node_mask is None:
        raise ValueError("Explanation is missing node_mask")
    if explanation.edge_mask.numel() != data.edge_index.size(1):
        raise ValueError("Explanation edge_mask must align with full graph edge_index")
    if not torch.isfinite(explanation.edge_mask).all():
        raise ValueError("edge_mask contains NaN or Inf")
    if not torch.isfinite(explanation.node_mask).all():
        raise ValueError("node_mask contains NaN or Inf")


def export_single_node_explanation(
    output_dir: Path,
    explanation: Any,
    data: Data,
    target: TargetNode,
    metadata: dict[str, Any],
    top_k_edges: int,
    subgraph_edge_mask: torch.Tensor,
) -> None:
    validate_explanation_masks(explanation, data)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(explanation.edge_mask.detach().cpu(), output_dir / "edge_mask.pt")
    torch.save(explanation.node_mask.detach().cpu(), output_dir / "node_mask.pt")
    top_edges = _write_top_edges(
        output_dir / "top_edges.csv",
        explanation.edge_mask.detach().cpu(),
        data,
        top_k_edges,
        subgraph_edge_mask.detach().cpu(),
    )
    write_explanation_figure(output_dir / "explanation.png", data.cpu(), target.node_id, top_edges)
    metadata_payload = {
        **metadata,
        "target": {
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "confidence": target.confidence,
        },
        "mask_shapes": {
            "edge_mask": list(explanation.edge_mask.shape),
            "node_mask": list(explanation.node_mask.shape),
        },
        "top_k_edges": top_k_edges,
        "raw_masks_saved": True,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_top_edges(
    path: Path,
    edge_mask: torch.Tensor,
    data: Data,
    top_k_edges: int,
    subgraph_edge_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    values, indices = torch.topk(edge_mask, k=min(top_k_edges, edge_mask.numel()))
    rows = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "global_edge_index",
                "src",
                "dst",
                "score",
                "is_ground_truth",
                "in_computation_subgraph",
            ],
        )
        writer.writeheader()
        for rank, (score, edge_idx) in enumerate(zip(values, indices, strict=True), start=1):
            global_edge_idx = int(edge_idx.item())
            row = {
                "rank": rank,
                "global_edge_index": global_edge_idx,
                "src": int(data.edge_index[0, global_edge_idx].item()),
                "dst": int(data.edge_index[1, global_edge_idx].item()),
                "score": float(score.item()),
                "is_ground_truth": bool(data.edge_mask[global_edge_idx].item()),
                "in_computation_subgraph": bool(subgraph_edge_mask[global_edge_idx].item()),
            }
            rows.append(row)
            writer.writerow(row)
    return rows
