# ruff: noqa: E402
from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from gnnexplainer_benchmark.data.serialization import (
    load_data_artifact,
    load_metadata,
    sha256_file,
)
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.training.checkpointing import (
    build_graphsage_from_checkpoint,
    load_checkpoint,
)
from gnnexplainer_benchmark.training.trainer import load_training_config, train_graphsage

EXPECTED_BA_HOUSE_SHA256 = "4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and freeze GraphSAGE on BA-House.")
    parser.add_argument("--artifact", type=Path, default=Path("data/ba_house/ba500/graph.pt"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/data/ba_house_metadata.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ba_house/ba500/graphsage_ba_house.yaml"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/backbones/graphsage/seed42.pt"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house_graphsage_best_manifest.json"),
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path("results/metrics/graphsage_ba_house_training/history.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/metrics/graphsage_ba_house_training/summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    dataset_checksum = dataset_sha256(args.artifact, args.metadata)
    if dataset_checksum != EXPECTED_BA_HOUSE_SHA256:
        raise RuntimeError(
            f"BA-House dataset checksum mismatch: {dataset_checksum} != {EXPECTED_BA_HOUSE_SHA256}"
        )

    config = load_training_config(args.config)
    result = train_graphsage(
        data=data,
        config=config,
        checkpoint_path=args.checkpoint_output,
        history_path=args.history_output,
        summary_path=args.summary_output,
        dataset_checksum=dataset_checksum,
    )
    checkpoint = load_checkpoint(args.checkpoint_output)
    reload_parity = verify_reload_parity(data, checkpoint)
    checkpoint_hash = sha256_file(args.checkpoint_output)
    manifest = build_manifest(
        args=args,
        config=config,
        result=result.__dict__,
        dataset_checksum=dataset_checksum,
        checkpoint_hash=checkpoint_hash,
        reload_parity=reload_parity,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "GraphSAGE training complete | "
        f"best_epoch={result.best_epoch} | "
        f"best_val_accuracy={result.best_validation_accuracy:.4f} | "
        f"test_accuracy={result.test_accuracy:.4f} | "
        f"motif_test_correct={result.motif_test_correct}/{result.motif_test_total} | "
        f"reload_max_abs_diff={reload_parity['max_abs_diff']:.3e}"
    )
    print(f"checkpoint={args.checkpoint_output}")
    print(f"manifest={args.manifest_output}")
    print(f"history={args.history_output}")
    print(f"summary={args.summary_output}")


def dataset_sha256(artifact: Path, metadata: Path) -> str:
    if metadata.exists():
        return str(load_metadata(metadata)["sha256"])
    return sha256_file(artifact)


def verify_reload_parity(data: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cpu")
    model = build_graphsage_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    ).to(device)
    reloaded = build_graphsage_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    ).to(device)
    model.eval()
    reloaded.eval()
    cpu_data = data.cpu()
    with torch.no_grad():
        expected = model(cpu_data.x, cpu_data.edge_index)
        actual = reloaded(cpu_data.x, cpu_data.edge_index)
    max_abs_diff = float((expected - actual).abs().max().item())
    return {
        "passed": bool(torch.allclose(expected, actual, atol=0.0, rtol=0.0)),
        "max_abs_diff": max_abs_diff,
    }


def build_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    result: dict[str, Any],
    dataset_checksum: str,
    checkpoint_hash: str,
    reload_parity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "success",
        "task": "Task 18 GraphSAGE BA-House freeze",
        "dataset": {
            "path": str(args.artifact),
            "sha256": dataset_checksum,
            "expected_sha256": EXPECTED_BA_HOUSE_SHA256,
            "split_reused": True,
            "edge_ordering_reused": True,
        },
        "model_config": config,
        "model_config_path": str(args.config),
        "model_config_sha256": sha256_file(args.config),
        "architecture_comparison_to_gcn": {
            "same_num_message_passing_layers": True,
            "same_hidden_width": True,
            "same_dropout": True,
            "same_output_task": True,
            "difference": (
                "GraphSAGE uses PyG SAGEConv(sum) instead of GCNConv. "
                "The sum aggregator preserves neighborhood-size signal on "
                "BA-House, where mean aggregation was degenerate with the "
                "frozen synthetic features."
            ),
        },
        "training_protocol": {
            "train_split": "train_mask",
            "selection_split": "val_mask",
            "test_reported_after_checkpoint_freeze": True,
            "selection_metric": config["checkpoint"]["monitor"],
        },
        "training_result": result,
        "checkpoint": {
            "path": str(args.checkpoint_output),
            "sha256": checkpoint_hash,
            "reload_parity": reload_parity,
        },
        "outputs": {
            "history": str(args.history_output),
            "summary": str(args.summary_output),
            "manifest": str(args.manifest_output),
        },
        "environment": environment_metadata(),
        "git": git_metadata(),
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def git_metadata() -> dict[str, str]:
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "status_short": run_git("status", "--short"),
    }


def run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
        encoding="utf-8",
    ).strip()


if __name__ == "__main__":
    main()
