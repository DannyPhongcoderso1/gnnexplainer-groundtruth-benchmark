from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.serialization import load_data_artifact, load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the frozen ShapeGGen artifact.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.data)
    metadata = load_metadata(args.metadata)
    labels, counts = torch.unique(data.y, sorted=True, return_counts=True)
    label_distribution = dict(zip(labels.tolist(), counts.tolist(), strict=True))
    candidate_counts = data.candidate_edge_mask.sum(dim=1)
    positive_counts = data.edge_mask.sum(dim=1)
    print(f"sha256={metadata['sha256']}")
    print(f"nodes={data.num_nodes}")
    print(f"directed_edges={data.edge_index.size(1)}")
    print(f"features={data.x.size(1)}")
    print(f"labels={label_distribution}")
    print(f"splits={metadata['split_counts']}")
    print(
        "candidate_edges_per_target="
        f"min={int(candidate_counts.min())}, "
        f"median={float(candidate_counts.float().median()):.1f}, "
        f"max={int(candidate_counts.max())}"
    )
    print(
        "positive_edges_per_target="
        f"min={int(positive_counts.min())}, "
        f"median={float(positive_counts.float().median()):.1f}, "
        f"max={int(positive_counts.max())}"
    )


if __name__ == "__main__":
    main()
