from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.evaluation.calibration import (
    assert_gcnconv_cached_false,
    edge_and_fidelity_for_selected_mask,
    exact_ground_truth_candidate_edge_mask,
    oracle_edge_scores,
    random_top_k_calibration,
)
from gnnexplainer_benchmark.evaluation.sparsity import select_top_k_edges
from gnnexplainer_benchmark.explain.node_selection import correctly_predicted_motif_test_nodes
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate node 465 fidelity sanity checks.")
    parser.add_argument("--artifact", type=Path, default=Path("data/ba_house/ba500/graph.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"),
    )
    parser.add_argument(
        "--gnn-edge-mask",
        type=Path,
        default=Path("results/explanations/node_465/edge_mask.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/node_465_fidelity_calibration.json"),
    )
    parser.add_argument("--node-id", type=int, default=465)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--random-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def main() -> None:
    args = parse_args()
    data_cpu = load_data_artifact(args.artifact)
    validate_ba_house_data(data_cpu)
    checkpoint = load_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    model = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    assert_gcnconv_cached_false(model)

    data_device = data_cpu.clone().to(device)
    with torch.no_grad():
        logits = model(data_device.x, data_device.edge_index)
    targets = correctly_predicted_motif_test_nodes(logits.cpu(), data_cpu)
    matches = [target for target in targets if target.node_id == args.node_id]
    if not matches:
        raise SystemExit(f"Node {args.node_id} is not a correct motif test target")
    target = matches[0]

    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])
    candidate_edge_mask = computation_subgraph_edge_mask(data_cpu, target.node_id, num_hops)
    ground_truth_selected = exact_ground_truth_candidate_edge_mask(data_cpu, candidate_edge_mask)
    oracle_scores = oracle_edge_scores(data_cpu, candidate_edge_mask)
    oracle = edge_and_fidelity_for_selected_mask(
        model=model,
        data=data_device,
        node_id=target.node_id,
        candidate_edge_mask=candidate_edge_mask,
        edge_scores=oracle_scores,
        selected_edge_mask=ground_truth_selected,
    )

    gnn_scores = torch.load(args.gnn_edge_mask, map_location="cpu", weights_only=False)
    gnn_selected = select_top_k_edges(gnn_scores, candidate_edge_mask, args.top_k)
    gnnexplainer = edge_and_fidelity_for_selected_mask(
        model=model,
        data=data_device,
        node_id=target.node_id,
        candidate_edge_mask=candidate_edge_mask,
        edge_scores=gnn_scores,
        selected_edge_mask=gnn_selected,
    )

    _, random_summary = random_top_k_calibration(
        model=model,
        data=data_device,
        node_id=target.node_id,
        candidate_edge_mask=candidate_edge_mask,
        top_k=args.top_k,
        repeats=args.random_repeats,
        seed=args.seed,
    )

    candidate_count = int(candidate_edge_mask.sum().item())
    positive_count = int(ground_truth_selected.sum().item())
    report = {
        "node_id": target.node_id,
        "target_class": target.prediction,
        "checkpoint": {
            "path": str(args.checkpoint),
            "best_epoch": int(checkpoint["best_epoch"]),
            "seed": int(checkpoint["seed"]),
            "gcnconv_cached": bool(checkpoint["resolved_config"]["model"]["cached"]),
        },
        "verification": {
            "same_checkpoint": True,
            "same_target_class": True,
            "same_candidate_edge_mapping": True,
            "gcnconv_cached_false": True,
            "top_k_not_tuned": args.top_k == 25,
        },
        "candidate_edge_scope": f"{num_hops}-hop computation subgraph",
        "candidate_edge_count": candidate_count,
        "ground_truth_candidate_edge_count": positive_count,
        "ground_truth_edge_prevalence": positive_count / candidate_count,
        "top_k": args.top_k,
        "random_repeats": args.random_repeats,
        "random_seed": args.seed,
        "diagnostic": {
            "random_top_k": random_summary,
            "gnnexplainer_top_k": gnnexplainer,
            "ground_truth_oracle": oracle,
        },
    }
    save_json(args.output, report)
    print(
        "Node 465 fidelity calibration complete | "
        f"candidate_edges={candidate_count} | "
        f"gt_edges={positive_count} | "
        f"random_repeats={args.random_repeats} | "
        f"oracle_remove_delta={oracle['fidelity']['remove_delta']:.6f} | "
        f"gnn_remove_delta={gnnexplainer['fidelity']['remove_delta']:.6f}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
