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
from statistics import mean, median, stdev
from typing import Any

import torch
import yaml

from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.evaluation.budget import (
    choose_ratio_by_validation_remove_delta,
    select_top_ratio_edges,
)
from gnnexplainer_benchmark.evaluation.edge_ground_truth import (
    candidate_edge_scores_and_labels,
    compute_edge_ground_truth_metrics,
)
from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity
from gnnexplainer_benchmark.explainers.gnnexplainer import (
    load_explainer_config,
    run_gnnexplainer_for_node,
)
from gnnexplainer_benchmark.explain.node_selection import TargetNode
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)

PER_RATIO_FIELDS = [
    "node_id",
    "ratio",
    "candidate_edge_count",
    "selected_edge_count",
    "sparsity",
    "edge_auroc",
    "edge_ap",
    "original_probability",
    "remove_probability",
    "remove_delta",
    "retain_probability",
    "retain_delta",
    "retain_matches_original_prediction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation-only explanation budgets.")
    parser.add_argument("--artifact", type=Path, default=Path("data/ba_house/ba500/graph.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"),
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=Path("results/metrics/explainer_validation_targets.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/shared/gnnexplainer.yaml"))
    parser.add_argument(
        "--metrics-config",
        type=Path,
        default=Path("configs/shared/core_metrics.yaml"),
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument(
        "--per-ratio-output",
        type=Path,
        default=Path("results/metrics/validation_budget_per_ratio.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/metrics/validation_budget_summary.json"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_arg)


def read_targets(path: Path) -> list[TargetNode]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        TargetNode(
            node_id=int(row["node_id"]),
            label=int(row["label"]),
            prediction=int(row["prediction"]),
            confidence=float(row["confidence"]),
        )
        for row in rows
    ]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_RATIO_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], ratios: list[float]) -> dict[str, Any]:
    summary = {}
    for ratio in ratios:
        ratio_rows = [row for row in rows if float(row["ratio"]) == ratio]
        summary[str(ratio)] = {
            "n": len(ratio_rows),
            "selected_edge_count_by_node": {
                str(row["node_id"]): int(row["selected_edge_count"]) for row in ratio_rows
            },
            "mean_remove_delta": mean(float(row["remove_delta"]) for row in ratio_rows),
            "median_remove_delta": median(float(row["remove_delta"]) for row in ratio_rows),
            "std_remove_delta": stdev(float(row["remove_delta"]) for row in ratio_rows)
            if len(ratio_rows) > 1
            else 0.0,
            "mean_retain_probability": mean(float(row["retain_probability"]) for row in ratio_rows),
            "retain_match_rate": mean(
                1.0 if str(row["retain_matches_original_prediction"]) == "True" else 0.0
                for row in ratio_rows
            ),
            "mean_sparsity": mean(float(row["sparsity"]) for row in ratio_rows),
        }
    return summary


def freeze_ratio_protocol(path: Path, selected_ratio: float, ratios: list[float]) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    evaluation = config.setdefault("evaluation", {})
    evaluation["budget_protocol"] = "fixed_candidate_edge_ratio"
    evaluation["selected_edge_ratio"] = selected_ratio
    evaluation["candidate_edge_ratios_evaluated_on_validation"] = ratios
    evaluation["ratio_selection_criterion"] = (
        "validation_only_max_mean_remove_delta_tie_lowest_ratio"
    )
    evaluation["edge_ranking_metric_scope"] = "continuous_scores_over_full_candidate_edge_set"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_cpu = load_data_artifact(args.artifact)
    validate_ba_house_data(data_cpu)
    checkpoint = load_checkpoint(args.checkpoint)
    explainer_config = load_explainer_config(args.config)
    device = resolve_device(args.device)
    model = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    data_device = data_cpu.clone().to(device)
    targets = read_targets(args.target_file)
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])

    rows: list[dict[str, Any]] = []
    for target in targets:
        explanation = run_gnnexplainer_for_node(model, data_device, target, explainer_config)
        edge_scores = explanation.edge_mask.detach().cpu()
        candidate_edge_mask = computation_subgraph_edge_mask(data_cpu, target.node_id, num_hops)
        candidate_scores, candidate_ground_truth = candidate_edge_scores_and_labels(
            edge_scores,
            data_cpu.edge_mask,
            candidate_edge_mask,
        )
        edge_metrics = compute_edge_ground_truth_metrics(candidate_scores, candidate_ground_truth)
        for ratio in args.ratios:
            selected_mask, budget = select_top_ratio_edges(edge_scores, candidate_edge_mask, ratio)
            fidelity = compute_fidelity(
                model=model,
                data=data_device,
                node_id=target.node_id,
                selected_edge_mask=selected_mask.to(device),
            )
            rows.append(
                {
                    "node_id": target.node_id,
                    "ratio": ratio,
                    "candidate_edge_count": budget.candidate_edge_count,
                    "selected_edge_count": budget.selected_edge_count,
                    "sparsity": budget.sparsity,
                    "edge_auroc": edge_metrics["edge_auroc"].value,
                    "edge_ap": edge_metrics["edge_ap"].value,
                    "original_probability": fidelity.original_probability,
                    "remove_probability": fidelity.remove_probability,
                    "remove_delta": fidelity.remove_delta,
                    "retain_probability": fidelity.retain_probability,
                    "retain_delta": fidelity.retain_delta,
                    "retain_matches_original_prediction": (
                        fidelity.retain_matches_original_prediction
                    ),
                }
            )

    selected_ratio = choose_ratio_by_validation_remove_delta(rows, args.ratios)
    summary = {
        "target_file": str(args.target_file),
        "targets": [target.node_id for target in targets],
        "ratios": args.ratios,
        "selection_criterion": "validation_only_max_mean_remove_delta_tie_lowest_ratio",
        "selected_ratio": selected_ratio,
        "edge_ranking_metric_scope": "continuous_scores_over_full_candidate_edge_set",
        "by_ratio": summarize(rows, args.ratios),
    }
    write_rows(args.per_ratio_output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze_ratio_protocol(args.metrics_config, selected_ratio, args.ratios)
    print(
        "Validation budget calibration complete | "
        f"targets={len(targets)} | "
        f"ratios={args.ratios} | "
        f"selected_ratio={selected_ratio}"
    )
    print(f"per_ratio={args.per_ratio_output}")
    print(f"summary={args.summary_output}")


if __name__ == "__main__":
    main()
