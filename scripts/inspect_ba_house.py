from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.data.validation import validate_ba_house_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a generated BA-House artifact.")
    parser.add_argument("--artifact", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    labels, counts = torch.unique(data.y, sorted=True, return_counts=True)
    distribution = ", ".join(
        f"{int(label.item())}:{int(count.item())}"
        for label, count in zip(labels, counts, strict=True)
    )
    print(
        "BA-House artifact OK | "
        f"nodes={data.num_nodes} | "
        f"edges={data.edge_index.size(1)} | "
        f"features={data.x.size(1)} | "
        f"labels={{{distribution}}} | "
        f"gt_nodes={int(data.node_mask.sum().item())} | "
        f"gt_edges={int(data.edge_mask.sum().item())} | "
        f"train={int(data.train_mask.sum().item())} | "
        f"val={int(data.val_mask.sum().item())} | "
        f"test={int(data.test_mask.sum().item())}"
    )


if __name__ == "__main__":
    main()
