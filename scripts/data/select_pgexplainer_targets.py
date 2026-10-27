# ruff: noqa: E402
from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
import csv
import shutil
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.pgexplainer_targets import (
    select_pgexplainer_targets,
    validate_pgexplainer_target_splits,
    write_pgexplainer_targets,
)
from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select PGExplainer train/validation targets.")
    parser.add_argument("--artifact", type=Path, default=Path("data/ba_house/ba500/graph.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"),
    )
    parser.add_argument(
        "--frozen-test-targets",
        type=Path,
        default=Path("results/metrics/explainer_test_targets.csv"),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv"),
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv"),
    )
    parser.add_argument(
        "--generated-train-copy",
        type=Path,
        default=Path("results/metrics/pgexplainer_train_targets.csv"),
    )
    parser.add_argument(
        "--generated-val-copy",
        type=Path,
        default=Path("results/metrics/pgexplainer_val_targets.csv"),
    )
    parser.add_argument("--write-canonical", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def read_node_ids(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["node_id"]) for row in csv.DictReader(handle)}


def assert_or_write_canonical(path: Path, targets: list[object], *, write_canonical: bool) -> None:
    if write_canonical or not path.exists():
        write_pgexplainer_targets(path, targets)
        return
    tmp_path = path.with_suffix(path.suffix + ".regenerated.tmp")
    write_pgexplainer_targets(tmp_path, targets)
    try:
        if not target_rows_match(path, tmp_path):
            raise ValueError(
                f"Regenerated targets differ from frozen canonical file: {path}. "
                "Do not silently overwrite canonical PGExplainer targets."
            )
    finally:
        tmp_path.unlink(missing_ok=True)


def target_rows_match(
    canonical_path: Path,
    regenerated_path: Path,
    confidence_tolerance: float = 1e-6,
) -> bool:
    with canonical_path.open(newline="", encoding="utf-8") as handle:
        canonical_rows = list(csv.DictReader(handle))
    with regenerated_path.open(newline="", encoding="utf-8") as handle:
        regenerated_rows = list(csv.DictReader(handle))
    if len(canonical_rows) != len(regenerated_rows):
        return False
    stable_fields = [
        "node_id",
        "ground_truth_label",
        "frozen_model_prediction",
        "target_for_pgexplainer",
        "base_model_split",
        "is_ground_truth_motif",
        "gcn_prediction_correct",
        "candidate_edge_count",
        "candidate_edge_mapping_valid",
    ]
    for canonical, regenerated in zip(canonical_rows, regenerated_rows, strict=True):
        for field in stable_fields:
            if canonical[field] != regenerated[field]:
                return False
        confidence_delta = abs(float(canonical["confidence"]) - float(regenerated["confidence"]))
        if confidence_delta > confidence_tolerance:
            return False
    return True


def main() -> None:
    args = parse_args()
    data_cpu = load_data_artifact(args.artifact)
    validate_ba_house_data(data_cpu)
    checkpoint = load_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    data_device = data_cpu.clone().to(device)
    with torch.no_grad():
        logits = model(data_device.x, data_device.edge_index).cpu()

    test_node_ids = read_node_ids(args.frozen_test_targets)
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])
    train_targets = select_pgexplainer_targets(
        logits=logits,
        data=data_cpu,
        split="train",
        num_hops=num_hops,
        excluded_node_ids=test_node_ids,
    )
    validation_targets = select_pgexplainer_targets(
        logits=logits,
        data=data_cpu,
        split="validation",
        num_hops=num_hops,
        excluded_node_ids=test_node_ids,
    )
    validate_pgexplainer_target_splits(train_targets, validation_targets, test_node_ids)
    assert_or_write_canonical(
        args.train_output,
        train_targets,
        write_canonical=args.write_canonical,
    )
    assert_or_write_canonical(
        args.val_output,
        validation_targets,
        write_canonical=args.write_canonical,
    )
    args.generated_train_copy.parent.mkdir(parents=True, exist_ok=True)
    args.generated_val_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.train_output, args.generated_train_copy)
    shutil.copy2(args.val_output, args.generated_val_copy)
    print(
        "PGExplainer targets selected | "
        f"train={len(train_targets)} | validation={len(validation_targets)} | "
        f"frozen_test_excluded={len(test_node_ids)} | device={device}"
    )
    print(f"train_output={args.train_output}")
    print(f"val_output={args.val_output}")


if __name__ == "__main__":
    main()
