# ruff: noqa: E402
from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


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
)
from gnnexplainer_benchmark.evaluation.budget import select_top_ratio_edges
from gnnexplainer_benchmark.evaluation.stability import (
    summarize_node_stability,
    summarize_stability_rows,
)
from gnnexplainer_benchmark.explain.gnnexplainer_runner import load_explainer_config
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4D GNNExplainer stability.")
    parser.add_argument("--artifact", type=Path, default=Path("data/ba_house/ba500/graph.pt"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/data/ba_house_metadata.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"),
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=Path("results/metrics/explainer_test_targets.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/shared/gnnexplainer.yaml"))
    parser.add_argument(
        "--metrics-config",
        type=Path,
        default=Path("configs/shared/core_metrics.yaml"),
    )
    parser.add_argument(
        "--phase4d-config",
        type=Path,
        default=Path("configs/ba_house/ba500/phase4d.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/explanations/phase4d_stability"),
    )
    parser.add_argument(
        "--metrics-output-dir",
        type=Path,
        default=Path("results/metrics/phase4d_stability"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-targets", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def load_yaml(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def selected_edge_set_for_node(
    *,
    edge_mask_path: Path,
    data: Any,
    node_id: int,
    num_hops: int,
    selected_edge_ratio: float,
) -> tuple[set[int], int]:
    edge_scores = torch.load(edge_mask_path, map_location="cpu", weights_only=False)
    candidate_edge_mask = computation_subgraph_edge_mask(data, node_id, num_hops)
    selected_edge_mask, budget = select_top_ratio_edges(
        edge_scores=edge_scores.detach().cpu(),
        candidate_edge_mask=candidate_edge_mask.detach().cpu(),
        ratio=selected_edge_ratio,
    )
    selected_indices = torch.nonzero(selected_edge_mask, as_tuple=False).view(-1).tolist()
    return set(int(edge_idx) for edge_idx in selected_indices), budget.selected_edge_count


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    started_at = perf_counter()
    data_cpu = load_data_artifact(args.artifact)
    validate_ba_house_data(data_cpu)
    checkpoint = load_checkpoint(args.checkpoint)
    metrics_config = load_yaml(args.metrics_config)
    phase4d_config = load_yaml(args.phase4d_config)["phase4d"]
    selected_edge_ratio = float(phase4d_config["selected_edge_ratio"])
    seeds = [int(seed) for seed in phase4d_config["stability_seeds"]]
    targets = read_explainer_targets(args.target_file)
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    if args.metadata.exists():
        dataset_checksum = load_metadata(args.metadata)["sha256"]
    else:
        dataset_checksum = sha256_file(args.artifact)

    device = resolve_device(args.device)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    before_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    data_device = data_cpu.clone().to(device)
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])

    per_seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_config = load_explainer_config(args.config)
        seed_config["explainer"]["seed"] = seed
        batch_config = BatchConfig(
            checkpoint_path=args.checkpoint,
            target_file=args.target_file,
            explainer_config=seed_config,
            metrics_config=metrics_config,
            seed=seed,
            selected_edge_ratio=selected_edge_ratio,
            output_root=args.output_root / f"seed_{seed}",
            metrics_output_dir=args.metrics_output_dir,
            device=str(device),
            dataset_checksum=dataset_checksum,
            num_hops=num_hops,
            raw_reuse_root=None,
        )
        for target in targets:
            try:
                per_seed_rows.append(
                    run_batch_target(
                        model=model,
                        data_cpu=data_cpu,
                        data_device=data_device,
                        target=target,
                        config=batch_config,
                    )
                )
            except Exception as exc:
                per_seed_rows.append(failure_row(target, seed, exc))

    per_seed_path = args.metrics_output_dir / "gnnexplainer_stability_seed_metrics.csv"
    write_per_node_metrics(per_seed_path, per_seed_rows)

    stability_rows: list[dict[str, Any]] = []
    for target in targets:
        edge_sets: list[set[int]] = []
        selected_counts: list[int] = []
        errors = [
            row["error"]
            for row in per_seed_rows
            if row["node_id"] == target.node_id and row["status"] == "failed"
        ]
        for seed in seeds:
            edge_mask_path = (
                args.output_root / f"seed_{seed}" / f"node_{target.node_id}" / "edge_mask.pt"
            )
            if not edge_mask_path.exists():
                continue
            selected_edges, selected_count = selected_edge_set_for_node(
                edge_mask_path=edge_mask_path,
                data=data_cpu,
                node_id=target.node_id,
                num_hops=num_hops,
                selected_edge_ratio=selected_edge_ratio,
            )
            edge_sets.append(selected_edges)
            selected_counts.append(selected_count)
        if len(edge_sets) >= 2:
            row = summarize_node_stability(target.node_id, edge_sets)
            row.update(
                {
                    "selected_edge_count": selected_counts[0] if selected_counts else None,
                    "status": "success",
                    "error": "",
                }
            )
        else:
            row = {
                "node_id": target.node_id,
                "mean_pairwise_jaccard": None,
                "std_pairwise_jaccard": None,
                "median_pairwise_jaccard": None,
                "min_pairwise_jaccard": None,
                "max_pairwise_jaccard": None,
                "num_seeds": len(edge_sets),
                "num_pairs": 0,
                "selected_edge_count": selected_counts[0] if selected_counts else None,
                "status": "failed",
                "error": "not enough successful seeds" if not errors else " | ".join(errors),
            }
        stability_rows.append(row)

    stability_path = args.metrics_output_dir / "gnnexplainer_stability.csv"
    stability_fields = [
        "node_id",
        "mean_pairwise_jaccard",
        "std_pairwise_jaccard",
        "median_pairwise_jaccard",
        "min_pairwise_jaccard",
        "max_pairwise_jaccard",
        "num_seeds",
        "num_pairs",
        "selected_edge_count",
        "status",
        "error",
    ]
    write_rows(stability_path, stability_rows, stability_fields)

    summary = summarize_stability_rows(stability_rows)
    summary_path = args.metrics_output_dir / "gnnexplainer_stability_summary.csv"
    write_rows(
        summary_path,
        summary,
        ["metric", "n", "mean", "std", "median", "min", "max"],
    )

    model_weights_unchanged = all(
        torch.equal(before_state[name], value.detach().cpu())
        for name, value in model.state_dict().items()
    )
    manifest = {
        "target_file": str(args.target_file),
        "target_count": len(targets),
        "seeds": seeds,
        "selected_edge_ratio": selected_edge_ratio,
        "rounding_rule": phase4d_config["rounding_rule"],
        "edge_ranking_metric_scope": phase4d_config["edge_ranking_metric_scope"],
        "success_nodes": sum(row["status"] == "success" for row in stability_rows),
        "failed_nodes": sum(row["status"] == "failed" for row in stability_rows),
        "model_weights_unchanged": model_weights_unchanged,
        "device": str(device),
        "total_runtime_seconds": perf_counter() - started_at,
        "per_seed_metrics": str(per_seed_path),
        "stability_metrics": str(stability_path),
        "stability_summary": str(summary_path),
    }
    manifest_path = args.metrics_output_dir / "gnnexplainer_stability_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "GNNExplainer stability complete | "
        f"targets={len(targets)} | seeds={len(seeds)} | "
        f"success_nodes={manifest['success_nodes']} | failed_nodes={manifest['failed_nodes']} | "
        f"runtime_seconds={manifest['total_runtime_seconds']:.3f}"
    )
    print(f"stability_metrics={stability_path}")
    print(f"stability_summary={summary_path}")


if __name__ == "__main__":
    main()
