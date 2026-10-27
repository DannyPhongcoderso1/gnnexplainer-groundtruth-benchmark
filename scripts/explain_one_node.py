from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.serialization import (
    load_data_artifact,
    load_metadata,
    sha256_file,
)
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.explainers.gnnexplainer import (
    load_explainer_config,
    run_gnnexplainer_for_node,
)
from gnnexplainer_benchmark.explain.mask_export import export_single_node_explanation
from gnnexplainer_benchmark.explain.node_selection import (
    correctly_predicted_motif_test_nodes,
    write_target_nodes,
)
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNNExplainer for one BA-House node.")
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
        "--targets-output",
        type=Path,
        default=Path("results/metrics/target_nodes.csv"),
    )
    parser.add_argument("--node-id", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("results/explanations"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    checkpoint = load_checkpoint(args.checkpoint)
    explainer_config = load_explainer_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    write_target_nodes(args.targets_output, targets)
    if not targets:
        raise SystemExit("No correctly predicted motif test nodes found")
    target = targets[0]
    if args.node_id is not None:
        matches = [item for item in targets if item.node_id == args.node_id]
        if not matches:
            raise SystemExit(f"Requested node {args.node_id} is not a valid target")
        target = matches[0]

    data = data.to(device)
    explanation = run_gnnexplainer_for_node(model, data, target, explainer_config)
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])
    data_cpu = data.cpu()
    subgraph_edge_mask = computation_subgraph_edge_mask(data_cpu, target.node_id, num_hops)
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
            "subgraph_edge_count": int(subgraph_edge_mask.sum().item()),
        },
        top_k_edges=int(explainer_config["explainer"].get("top_k_edges", 25)),
        subgraph_edge_mask=subgraph_edge_mask,
    )
    print(
        "Single-node explanation complete | "
        f"node={target.node_id} | "
        f"label={target.label} | "
        f"prediction={target.prediction} | "
        f"confidence={target.confidence:.6f} | "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
