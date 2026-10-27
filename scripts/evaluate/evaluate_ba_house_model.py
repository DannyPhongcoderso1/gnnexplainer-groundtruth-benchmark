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

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.data.target_selection import (
    select_correct_motif_targets,
    split_explainer_targets,
    write_explainer_targets,
)
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.evaluation.model_quality import build_model_quality_report
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BA-House GCN quality gate.")
    parser.add_argument("--artifact", type=Path, default=Path("data/ba_house/ba500/graph.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("results/metrics/model_quality_report.json"),
    )
    parser.add_argument(
        "--validation-targets-output",
        type=Path,
        default=Path("results/metrics/explainer_validation_targets.csv"),
    )
    parser.add_argument(
        "--test-targets-output",
        type=Path,
        default=Path("results/metrics/explainer_test_targets.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    checkpoint = load_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    model = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    ).to(device)
    model.eval()

    with torch.no_grad():
        device_data = data.to(device)
        logits = model(device_data.x, device_data.edge_index).cpu()

    report = build_model_quality_report(logits, data)
    targets = select_correct_motif_targets(logits, data)
    validation_targets, test_targets = split_explainer_targets(
        targets,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )

    report.update(
        {
            "artifact": str(args.artifact),
            "artifact_sha256": sha256_file(args.artifact),
            "checkpoint": str(args.checkpoint),
            "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
            "checkpoint_best_validation_metric": float(checkpoint["best_validation_metric"]),
            "checkpoint_seed": int(checkpoint["seed"]),
            "device": str(device),
            "target_selection": {
                "criteria": "split == test AND node_mask == True AND prediction == label",
                "seed": args.seed,
                "validation_fraction": args.validation_fraction,
                "total_correct_motif_test_targets": len(targets),
                "explainer_validation_targets": len(validation_targets),
                "explainer_test_targets": len(test_targets),
            },
        }
    )

    save_report(args.report_output, report)
    write_explainer_targets(args.validation_targets_output, validation_targets)
    write_explainer_targets(args.test_targets_output, test_targets)

    val_accuracy = report["splits"]["validation"]["overall_accuracy"]["accuracy"]
    test_accuracy = report["splits"]["test"]["overall_accuracy"]["accuracy"]
    test_motif_correct = report["splits"]["test"]["correctly_predicted_motif_nodes"]
    test_motif_total = report["splits"]["test"]["motif_nodes"]
    print(
        "BA-House GCN quality gate | "
        f"device={device} | "
        f"best_epoch={checkpoint['best_epoch']} | "
        f"validation_accuracy={val_accuracy:.4f} | "
        f"test_accuracy={test_accuracy:.4f} | "
        f"motif_test_correct={test_motif_correct}/{test_motif_total} | "
        f"explainer_validation_targets={len(validation_targets)} | "
        f"explainer_test_targets={len(test_targets)}"
    )
    print(f"report={args.report_output}")
    print(f"validation_targets={args.validation_targets_output}")
    print(f"test_targets={args.test_targets_output}")


if __name__ == "__main__":
    main()
