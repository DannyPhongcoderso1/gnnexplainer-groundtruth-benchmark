# ruff: noqa: E402
from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from gnnexplainer_benchmark.data.serialization import (
    load_data_artifact,
    load_metadata,
    sha256_file,
)
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.evaluation.evaluator import evaluate_node_explanation
from gnnexplainer_benchmark.evaluation.runtime import measure_runtime
from gnnexplainer_benchmark.explainers.gnnexplainer import (
    load_explainer_config,
    run_gnnexplainer_for_node,
)
from gnnexplainer_benchmark.explain.mask_export import export_single_node_explanation
from gnnexplainer_benchmark.explain.node_selection import correctly_predicted_motif_test_nodes
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate core metrics for one node explanation.")
    parser.add_argument("--artifact", type=Path, default=Path("data/ba_house/ba500/graph.pt"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/data/ba_house_metadata.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/shared/gnnexplainer.yaml"),
    )
    parser.add_argument(
        "--metrics-config",
        type=Path,
        default=Path("configs/shared/core_metrics.yaml"),
    )
    parser.add_argument("--node-id", type=int, default=465)
    parser.add_argument("--output-root", type=Path, default=Path("results/explanations"))
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/metrics/node_465_metrics.json"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_metrics_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "evaluation" not in config:
        raise ValueError("Metrics config must contain a top-level evaluation mapping")
    return config


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    checkpoint = load_checkpoint(args.checkpoint)
    explainer_config = load_explainer_config(args.config)
    metrics_config = load_metrics_config(args.metrics_config)
    top_k = args.top_k or int(metrics_config["evaluation"]["top_k_edges"])
    device = resolve_device(args.device)

    model = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    ).to(device)
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    targets = correctly_predicted_motif_test_nodes(logits.cpu(), data.cpu())
    matches = [target for target in targets if target.node_id == args.node_id]
    if not matches:
        raise SystemExit(f"Requested node {args.node_id} is not a valid correct motif test target")
    target = matches[0]
    data = data.to(device)

    with measure_runtime() as timer:
        explanation = run_gnnexplainer_for_node(model, data, target, explainer_config)
    if timer.runtime_seconds is None:
        raise RuntimeError("Runtime measurement failed")

    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])
    data_cpu = data.cpu()
    candidate_edge_mask = computation_subgraph_edge_mask(data_cpu, target.node_id, num_hops)
    if args.metadata.exists():
        dataset_checksum = load_metadata(args.metadata)["sha256"]
    else:
        dataset_checksum = sha256_file(args.artifact)

    output_dir = args.output_root / f"node_{target.node_id}"
    export_single_node_explanation(
        output_dir=output_dir,
        explanation=explanation,
        data=data_cpu,
        target=target,
        metadata={
            "dataset_checksum": dataset_checksum,
            "checkpoint": {
                "path": str(args.checkpoint),
                "best_epoch": checkpoint["best_epoch"],
                "best_validation_metric": checkpoint["best_validation_metric"],
            },
            "explainer_config": explainer_config,
            "num_hops": num_hops,
            "subgraph_edge_count": int(candidate_edge_mask.sum().item()),
            "runtime_seconds": timer.runtime_seconds,
        },
        top_k_edges=top_k,
        subgraph_edge_mask=candidate_edge_mask,
    )

    edge_scores = torch.load(output_dir / "edge_mask.pt", map_location="cpu", weights_only=False)
    data_for_eval = data_cpu.to(device)
    metrics = evaluate_node_explanation(
        model=model,
        data=data_for_eval,
        node_id=target.node_id,
        edge_scores=edge_scores,
        candidate_edge_mask=candidate_edge_mask,
        top_k=top_k,
        runtime_seconds=timer.runtime_seconds,
        explainer=explainer_config["explainer"]["algorithm"],
        target={
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "confidence": target.confidence,
        },
        candidate_edge_scope=metrics_config["evaluation"]["candidate_edge_scope"],
    )
    metrics["edge_ground_truth"]["candidate_edge_scope_detail"] = (
        f"{num_hops}-hop computation subgraph around target node"
    )
    save_json(args.metrics_output, metrics)
    print(
        "Single explanation metrics | "
        f"node={target.node_id} | "
        f"edge_auroc={metrics['edge_ground_truth']['edge_auroc']} | "
        f"edge_ap={metrics['edge_ground_truth']['edge_ap']} | "
        f"remove_delta={metrics['fidelity']['remove_result']['remove_delta']:.6f} | "
        f"candidate_edges={metrics['edge_ground_truth']['candidate_edge_count']} | "
        f"selected_edges={metrics['sparsity']['selected_edge_count']} | "
        f"sparsity={metrics['sparsity']['sparsity']:.6f} | "
        f"runtime_seconds={metrics['runtime']['runtime_seconds']:.6f}"
    )
    print(f"metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
