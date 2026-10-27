# ruff: noqa: E402
from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.explain.node_selection import (
    correctly_predicted_motif_test_nodes,
    write_target_nodes,
)
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select correctly predicted motif test nodes.")
    parser.add_argument("--artifact", type=Path, default=Path("data/ba_house/ba500/graph.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/metrics/target_nodes.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    checkpoint = load_checkpoint(args.checkpoint)
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
    write_target_nodes(args.output, targets)
    if not targets:
        raise SystemExit("No correctly predicted motif test nodes found")
    best = targets[0]
    print(
        "Target nodes selected | "
        f"count={len(targets)} | "
        f"top_node={best.node_id} | "
        f"label={best.label} | "
        f"prediction={best.prediction} | "
        f"confidence={best.confidence:.6f}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
