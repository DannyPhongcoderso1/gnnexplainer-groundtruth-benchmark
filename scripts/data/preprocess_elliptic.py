# ruff: noqa: E402
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src")]

from gnnexplainer_benchmark.data.elliptic import BCDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw Elliptic CSV files into a PyG Data .pt file."
    )
    parser.add_argument(
        "--raw-dir", default="datasets/elliptic", help="Directory containing Elliptic CSV files."
    )
    parser.add_argument(
        "--out", default="datasets/elliptic/elliptic_data.pt", help="Output .pt path."
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Keep graph directed instead of adding reverse edges.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    required = [
        "elliptic_txs_features.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_classes.csv",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(args.raw_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing raw Elliptic files in {args.raw_dir}: {missing}")

    dataset = BCDataset(type="elliptic", path=args.raw_dir, directed=args.directed)
    data = dataset.to_torch_data()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(data, args.out)

    print(f"Saved: {args.out}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Features: {data.num_features}")
    print(
        "Masks: "
        f"train={int(data.train_mask.sum())}, "
        f"val={int(data.val_mask.sum())}, "
        f"test={int(data.test_mask.sum())}"
    )
