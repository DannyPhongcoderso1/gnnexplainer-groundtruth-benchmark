from __future__ import annotations

import argparse
from pathlib import Path

from gnnexplainer_benchmark.data.serialization import (
    load_data_artifact,
    load_metadata,
    sha256_file,
)
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.training.trainer import load_training_config, train_gcn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a benchmark GCN on BA-House.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/data/ba_house.pt"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/data/ba_house_metadata.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/model/gcn.yaml"))
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house_gcn_best.pt"),
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path("results/metrics/gcn_training_history.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/metrics/gcn_model_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    config = load_training_config(args.config)
    if args.metadata.exists():
        dataset_checksum = load_metadata(args.metadata)["sha256"]
    else:
        dataset_checksum = sha256_file(args.artifact)
    result = train_gcn(
        data=data,
        config=config,
        checkpoint_path=args.checkpoint_output,
        history_path=args.history_output,
        summary_path=args.summary_output,
        dataset_checksum=dataset_checksum,
    )
    print(
        "GCN training complete | "
        f"best_epoch={result.best_epoch} | "
        f"best_val_accuracy={result.best_validation_accuracy:.4f} | "
        f"test_accuracy={result.test_accuracy:.4f} | "
        f"majority_baseline={result.majority_baseline_accuracy:.4f} | "
        f"random_baseline={result.random_baseline_accuracy:.4f} | "
        f"motif_test_correct={result.motif_test_correct}/{result.motif_test_total}"
    )
    print(f"checkpoint={result.checkpoint_path}")
    print(f"history={result.history_path}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
