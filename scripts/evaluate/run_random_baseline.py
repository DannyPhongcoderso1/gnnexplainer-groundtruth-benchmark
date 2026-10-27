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
import sys
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
    compute_ratio_metrics,
    failure_row,
    flatten_metrics_row,
    read_explainer_targets,
    write_per_node_metrics,
    write_selected_edges_csv,
    write_summary_metrics,
)
from gnnexplainer_benchmark.evaluation.runtime import measure_runtime
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.explainers.random_baseline import random_candidate_edge_scores
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    build_graphsage_from_checkpoint,
    load_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4D Random explanation baseline.")
    parser.add_argument("--backbone", choices=["gcn", "graphsage"], default="gcn")
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
        default=Path("results/explanations/phase4d_random_baseline"),
    )
    parser.add_argument(
        "--metrics-output-dir",
        type=Path,
        default=Path("results/metrics/phase4d_random_baseline"),
    )
    parser.add_argument(
        "--gnnexplainer-summary",
        type=Path,
        default=Path("results/metrics/frozen_phase4c_v2/summary_metrics.csv"),
    )
    parser.add_argument(
        "--stability-summary",
        type=Path,
        default=Path("results/metrics/phase4d_stability/gnnexplainer_stability_summary.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("results/metrics/phase4d_comparison.csv"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-targets", type=int, default=None)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.backbone == "graphsage":
        # Keep historical GCN defaults intact and isolate GraphSAGE outputs/references.
        defaults = {
            "checkpoint": "artifacts/checkpoints/ba_house/ba500/backbones/graphsage/seed42.pt",
            "target_file": "configs/ba_house/ba500/targets/ba_house_2x2_targets.csv",
            "output_root": "results/explanations/graphsage_random_baseline",
            "metrics_output_dir": "results/metrics/graphsage_random_baseline",
            "gnnexplainer_summary": (
                "results/metrics/graphsage_gnnexplainer/seed42/summary_metrics.csv"
            ),
            "stability_summary": "results/metrics/graphsage_gnnexplainer/stability/"
            "gnnexplainer_stability_summary.csv",
            "comparison_output": "results/metrics/graphsage_random_baseline/comparison.csv",
        }
        for key, value in defaults.items():
            option = "--" + key.replace("_", "-")
            if not any(arg == option or arg.startswith(option + "=") for arg in argv):
                setattr(args, key, Path(value))
        for key in ("output_root", "metrics_output_dir", "comparison_output"):
            path = getattr(args, key).resolve()
            legacy = Path(parser.get_default(key)).resolve()
            if path == legacy or legacy in path.parents or path in legacy.parents:
                raise ValueError("GraphSAGE output must not overlap historical GCN outputs")
    return args


def build_backbone_from_checkpoint(
    checkpoint: dict[str, Any], *, backbone: str, in_channels: int, out_channels: int
) -> torch.nn.Module:
    constructors = {"gcn": build_gcn_from_checkpoint, "graphsage": build_graphsage_from_checkpoint}
    if backbone not in constructors:
        raise ValueError(f"Unsupported backbone: {backbone}")
    model_type = str(checkpoint["resolved_config"]["model"].get("type", "GCN")).lower()
    if model_type != backbone:
        raise ValueError(f"Checkpoint type {model_type} does not match backbone {backbone}")
    model = constructors[backbone](checkpoint, in_channels, out_channels)
    if backbone == "graphsage":
        model.eval().requires_grad_(False)
    return model


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


def seed_for_node(seed: int, node_id: int) -> int:
    return seed * 1_000_003 + node_id


def read_summary_means(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["metric"]: float(row["mean"])
            for row in csv.DictReader(handle)
            if row.get("mean") not in (None, "")
        }


def write_comparison_table(
    *,
    path: Path,
    gnn_summary_path: Path,
    random_summary_path: Path,
    stability_summary_path: Path,
) -> None:
    gnn = read_summary_means(gnn_summary_path)
    random = read_summary_means(random_summary_path)
    stability = read_summary_means(stability_summary_path)
    rows = [
        {
            "method": "Random",
            "edge_auroc": random.get("edge_auroc"),
            "edge_ap": random.get("edge_ap"),
            "fidelity": random.get("fidelity_legacy"),
            "fid_plus": random.get("fid_plus"),
            "fid_minus": random.get("fid_minus"),
            "one_minus_fid_minus": random.get("one_minus_fid_minus"),
            "sparsity": random.get("sparsity"),
            "stability": "",
            "runtime_seconds": random.get("runtime_seconds"),
        },
        {
            "method": "GNNExplainer",
            "edge_auroc": gnn.get("edge_auroc"),
            "edge_ap": gnn.get("edge_ap"),
            "fidelity": gnn.get("fidelity_legacy"),
            "fid_plus": gnn.get("fid_plus"),
            "fid_minus": gnn.get("fid_minus"),
            "one_minus_fid_minus": gnn.get("one_minus_fid_minus"),
            "sparsity": gnn.get("sparsity"),
            "stability": stability.get("mean_pairwise_jaccard"),
            "runtime_seconds": gnn.get("generation_runtime_seconds") or gnn.get("runtime_seconds"),
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
    seeds = [int(seed) for seed in phase4d_config["random_baseline_seeds"]]
    targets = read_explainer_targets(args.target_file)
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    if args.metadata.exists():
        dataset_checksum = load_metadata(args.metadata)["sha256"]
    else:
        dataset_checksum = sha256_file(args.artifact)

    device = resolve_device(args.device)
    model = build_backbone_from_checkpoint(
        checkpoint,
        backbone=args.backbone,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    before_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    data_device = data_cpu.clone().to(device)
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        for target in targets:
            output_dir = args.output_root / f"seed_{seed}" / f"node_{target.node_id}"
            try:
                candidate_edge_mask = computation_subgraph_edge_mask(
                    data_cpu,
                    target.node_id,
                    num_hops,
                )
                edge_scores = random_candidate_edge_scores(
                    edge_count=int(data_cpu.edge_index.size(1)),
                    candidate_edge_mask=candidate_edge_mask,
                    seed=seed_for_node(seed, target.node_id),
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(edge_scores, output_dir / "edge_mask.pt")
                torch.save(torch.empty(0), output_dir / "node_mask.pt")
                metadata = {
                    "dataset_checksum": dataset_checksum,
                    "checkpoint": {"path": str(args.checkpoint)},
                    "metrics_config": metrics_config,
                    "num_hops": num_hops,
                    "explainer": "Random",
                    "random_seed": seed,
                    "node_random_seed": seed_for_node(seed, target.node_id),
                    "budget_protocol": "fixed_candidate_edge_ratio",
                    "selected_edge_ratio": selected_edge_ratio,
                    "rounding_rule": phase4d_config["rounding_rule"],
                    "raw_masks_saved": True,
                    "target": {
                        "node_id": target.node_id,
                        "label": target.label,
                        "prediction": target.prediction,
                        "confidence": target.confidence,
                    },
                }
                (output_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with measure_runtime() as timer:
                    metrics = compute_ratio_metrics(
                        model=model,
                        data=data_device,
                        node_id=target.node_id,
                        target=target,
                        edge_scores=edge_scores,
                        candidate_edge_mask=candidate_edge_mask,
                        selected_edge_ratio=selected_edge_ratio,
                        runtime_seconds=0.0,
                        explainer="Random",
                        candidate_edge_scope=metrics_config["evaluation"]["candidate_edge_scope"],
                    )
                if timer.runtime_seconds is None:
                    raise RuntimeError("Runtime measurement failed")
                metrics["runtime"]["runtime_seconds"] = timer.runtime_seconds
                selected_edge_mask = metrics.pop("_selected_edge_mask")
                write_selected_edges_csv(
                    output_dir / "selected_edges.csv",
                    edge_scores=edge_scores,
                    data=data_cpu,
                    selected_edge_mask=selected_edge_mask,
                    candidate_edge_mask=candidate_edge_mask,
                )
                write_selected_edges_csv(
                    output_dir / "top_edges.csv",
                    edge_scores=edge_scores,
                    data=data_cpu,
                    selected_edge_mask=selected_edge_mask,
                    candidate_edge_mask=candidate_edge_mask,
                )
                (output_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                rows.append(
                    flatten_metrics_row(
                        metrics,
                        target,
                        seed,
                        status="success",
                        generation_runtime_seconds=None,
                        evaluation_runtime_seconds=timer.runtime_seconds,
                        explanation_source="random_candidate_scores",
                    )
                )
            except Exception as exc:
                rows.append(failure_row(target, seed, exc))

    per_node_path = args.metrics_output_dir / "random_baseline_per_node.csv"
    summary_path = args.metrics_output_dir / "random_baseline_summary.csv"
    write_per_node_metrics(per_node_path, rows)
    summary = write_summary_metrics(summary_path, rows)
    write_comparison_table(
        path=args.comparison_output,
        gnn_summary_path=args.gnnexplainer_summary,
        random_summary_path=summary_path,
        stability_summary_path=args.stability_summary,
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
        "success_count": sum(row["status"] == "success" for row in rows),
        "fail_count": sum(row["status"] == "failed" for row in rows),
        "model_weights_unchanged": model_weights_unchanged,
        "device": str(device),
        "dataset_checksum": dataset_checksum,
        "total_runtime_seconds": perf_counter() - started_at,
        "per_node_metrics": str(per_node_path),
        "summary_metrics": str(summary_path),
        "comparison_table": str(args.comparison_output),
        "summary": summary,
    }
    manifest_path = args.metrics_output_dir / "random_baseline_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Random baseline complete | "
        f"targets={len(targets)} | seeds={len(seeds)} | "
        f"success={manifest['success_count']} | failed={manifest['fail_count']} | "
        f"runtime_seconds={manifest['total_runtime_seconds']:.3f}"
    )
    print(f"per_node_metrics={per_node_path}")
    print(f"summary_metrics={summary_path}")
    print(f"comparison_table={args.comparison_output}")


if __name__ == "__main__":
    main()
