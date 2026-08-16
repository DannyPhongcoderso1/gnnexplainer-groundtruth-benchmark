from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import yaml

from gnnexplainer_benchmark.data.serialization import (
    load_data_artifact,
    load_metadata,
    sha256_file,
)
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import (
    BatchConfig,
    failure_row,
    read_explainer_targets,
    run_batch_target,
    write_per_node_metrics,
    write_summary_metrics,
)
from gnnexplainer_benchmark.explain.gnnexplainer_runner import load_explainer_config
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch GNNExplainer on frozen targets.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/data/ba_house.pt"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/data/ba_house_metadata.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house_gcn_best.pt"),
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=Path("results/metrics/explainer_validation_targets.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/explainer/gnnexplainer.yaml"))
    parser.add_argument(
        "--metrics-config",
        type=Path,
        default=Path("configs/evaluation/core_metrics.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explainer-epochs", type=int, default=None)
    parser.add_argument("--selected-edge-ratio", type=float, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/explanations/phase4c_pilot"),
    )
    parser.add_argument(
        "--metrics-output-dir",
        type=Path,
        default=Path("results/metrics/phase4c_pilot"),
    )
    parser.add_argument("--raw-reuse-root", type=Path, default=Path("results/explanations"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-targets", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def load_metrics_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "evaluation" not in config:
        raise ValueError("Metrics config must contain a top-level evaluation mapping")
    return config


def write_failures(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [row for row in rows if row["status"] == "failed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "status", "error"])
        writer.writeheader()
        for row in failures:
            writer.writerow(
                {
                    "node_id": row["node_id"],
                    "status": row["status"],
                    "error": row["error"],
                }
            )


def main() -> None:
    args = parse_args()
    started_at = perf_counter()
    data_cpu = load_data_artifact(args.artifact)
    validate_ba_house_data(data_cpu)
    checkpoint = load_checkpoint(args.checkpoint)
    explainer_config = load_explainer_config(args.config)
    metrics_config = load_metrics_config(args.metrics_config)
    if args.explainer_epochs is not None:
        explainer_config["explainer"]["epochs"] = args.explainer_epochs
    explainer_config["explainer"]["seed"] = args.seed
    selected_edge_ratio = args.selected_edge_ratio
    if selected_edge_ratio is None:
        selected_edge_ratio = float(metrics_config["evaluation"]["selected_edge_ratio"])
    device = resolve_device(args.device)
    targets = read_explainer_targets(args.target_file)
    if args.max_targets is not None:
        targets = targets[: args.max_targets]

    if args.metadata.exists():
        dataset_checksum = load_metadata(args.metadata)["sha256"]
    else:
        dataset_checksum = sha256_file(args.artifact)

    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    before_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    data_device = data_cpu.clone().to(device)
    batch_config = BatchConfig(
        checkpoint_path=args.checkpoint,
        target_file=args.target_file,
        explainer_config=explainer_config,
        metrics_config=metrics_config,
        seed=args.seed,
        selected_edge_ratio=selected_edge_ratio,
        output_root=args.output_root,
        metrics_output_dir=args.metrics_output_dir,
        device=str(device),
        dataset_checksum=dataset_checksum,
        num_hops=int(checkpoint["resolved_config"]["model"]["num_layers"]),
        raw_reuse_root=args.raw_reuse_root,
    )

    rows: list[dict[str, Any]] = []
    for target in targets:
        try:
            rows.append(
                run_batch_target(
                    model=model,
                    data_cpu=data_cpu,
                    data_device=data_device,
                    target=target,
                    config=batch_config,
                )
            )
        except Exception as exc:
            rows.append(failure_row(target, args.seed, exc))

    per_node_path = args.metrics_output_dir / "per_node_metrics.csv"
    summary_path = args.metrics_output_dir / "summary_metrics.csv"
    failures_path = args.metrics_output_dir / "failed_targets.csv"
    manifest_path = args.metrics_output_dir / "batch_gnnexplainer_manifest.json"
    write_per_node_metrics(per_node_path, rows)
    summary = write_summary_metrics(summary_path, rows)
    write_failures(failures_path, rows)

    total_runtime_seconds = perf_counter() - started_at
    success_count = sum(row["status"] == "success" for row in rows)
    fail_count = sum(row["status"] == "failed" for row in rows)
    model_weights_unchanged = all(
        torch.equal(before_state[name], value.detach().cpu())
        for name, value in model.state_dict().items()
    )
    manifest = {
        "target_file": str(args.target_file),
        "target_count": len(targets),
        "success_count": success_count,
        "fail_count": fail_count,
        "seed": args.seed,
        "budget_protocol": metrics_config["evaluation"]["budget_protocol"],
        "selected_edge_ratio": selected_edge_ratio,
        "rounding_rule": "ceil(candidate_edge_count * selected_edge_ratio), minimum 1",
        "edge_ranking_metric_scope": metrics_config["evaluation"]["edge_ranking_metric_scope"],
        "model_weights_unchanged": model_weights_unchanged,
        "device": str(device),
        "total_runtime_seconds": total_runtime_seconds,
        **artifact_source_counts(args.output_root, targets),
        "per_node_metrics": str(per_node_path),
        "summary_metrics": str(summary_path),
        "failed_targets": str(failures_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Batch GNNExplainer complete | "
        f"targets={len(targets)} | "
        f"success={success_count} | "
        f"failed={fail_count} | "
        f"selected_edge_ratio={selected_edge_ratio} | "
        f"runtime_seconds={total_runtime_seconds:.3f}"
    )
    print(f"per_node_metrics={per_node_path}")
    print(f"summary_metrics={summary_path}")
    print(f"summary={summary}")


def artifact_source_counts(output_root: Path, targets: list[Any]) -> dict[str, Any]:
    reused = 0
    generated = 0
    source_generation_runtime = 0.0
    for target in targets:
        metadata_path = output_root / f"node_{target.node_id}" / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("raw_explanation_reused"):
            reused += 1
            source_generation_runtime += float(metadata.get("source_runtime_seconds") or 0.0)
        else:
            generated += 1
            source_generation_runtime += float(metadata.get("runtime_seconds") or 0.0)
    return {
        "raw_explanations_reused": reused,
        "raw_explanations_generated": generated,
        "raw_generation_runtime_seconds": source_generation_runtime,
    }


if __name__ == "__main__":
    main()
