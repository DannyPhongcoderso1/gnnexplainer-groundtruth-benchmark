from __future__ import annotations

import argparse
from pathlib import Path

from gnnexplainer_benchmark.data.ba_house import generate_and_save_ba_house


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the BA-House ground-truth dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = generate_and_save_ba_house(args.config, args.output, args.metadata)
    print(
        "BA-House generated | "
        f"nodes={data.num_nodes} | "
        f"edges={data.edge_index.size(1)} | "
        f"features={data.x.size(1)} | "
        f"gt_nodes={int(data.node_mask.sum().item())} | "
        f"gt_edges={int(data.edge_mask.sum().item())}"
    )
    print(f"artifact={args.output}")
    print(f"metadata={args.metadata}")


if __name__ == "__main__":
    main()
