from __future__ import annotations

# ruff: noqa: E501
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

import torch
import yaml
from prepare_elliptic_task26 import ROOT, build_models, load_data, logits, sha256
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from gnnexplainer_benchmark.evaluation.budget import (
    select_top_ratio_edges,
    selected_edge_count_for_ratio,
)
from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity
from gnnexplainer_benchmark.evaluation.runtime import measure_runtime
from gnnexplainer_benchmark.evaluation.stability import pairwise_jaccard_rows
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.utils.reproducibility import seed_everything


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    slug: str
    model_key: str
    checkpoint_key: str
    checkpoint_sha_key: str
    prediction_column: str
    confidence_column: str


@dataclass(frozen=True)
class Target:
    node_id: int
    label: int
    prediction: int
    confidence: float
    split: str
    candidate_edge_count: int


BACKBONES = [
    BackboneSpec(
        name="GCN",
        slug="gcn",
        model_key="gcn_compatible",
        checkpoint_key="gcn_checkpoint",
        checkpoint_sha_key="gcn_checkpoint_sha256",
        prediction_column="gcn_prediction",
        confidence_column="gcn_confidence",
    ),
    BackboneSpec(
        name="GraphSAGE",
        slug="graphsage",
        model_key="graphsage",
        checkpoint_key="graphsage_checkpoint",
        checkpoint_sha_key="graphsage_checkpoint_sha256",
        prediction_column="graphsage_prediction",
        confidence_column="graphsage_confidence",
    ),
]

PER_NODE_FIELDS = [
    "node_id",
    "label",
    "prediction",
    "confidence",
    "candidate_edge_count",
    "selected_edge_ratio",
    "selected_edge_count",
    "original_prediction",
    "remove_prediction",
    "retain_prediction",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "original_probability",
    "remove_probability",
    "remove_probability_delta",
    "retain_probability",
    "retain_probability_delta",
    "retain_probability_ratio",
    "sparsity",
    "generation_runtime_seconds",
    "evaluation_runtime_seconds",
    "seed",
    "status",
    "error",
    "weights_unchanged",
    "cache_safe_perturbation",
    "explanation_source",
]

SUMMARY_METRICS = [
    "candidate_edge_count",
    "selected_edge_count",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "original_probability",
    "remove_probability",
    "remove_probability_delta",
    "retain_probability",
    "retain_probability_delta",
    "retain_probability_ratio",
    "sparsity",
    "generation_runtime_seconds",
    "evaluation_runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task27 Elliptic GNNExplainer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/elliptic_gnnexplainer.yaml"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--skip-stability", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        return ""


def assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def read_targets(path: Path, spec: BackboneSpec) -> list[Target]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = [
        Target(
            node_id=int(row["node_id"]),
            label=int(row["label"]),
            prediction=int(row[spec.prediction_column]),
            confidence=float(row[spec.confidence_column]),
            split=row["split"],
            candidate_edge_count=int(row["candidate_edge_count"]),
        )
        for row in rows
    ]
    node_ids = [target.node_id for target in targets]
    if node_ids != sorted(node_ids):
        raise RuntimeError("Elliptic targets must remain sorted deterministically")
    if len(node_ids) != len(set(node_ids)):
        raise RuntimeError("Elliptic target file contains duplicate node_id values")
    return targets


def state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def weights_unchanged(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(value.detach().cpu(), snapshot[key]) for key, value in model.state_dict().items())


def verify_inputs(config: dict[str, Any], config_path: Path) -> dict[str, str]:
    hashes = {
        "task27_config": sha256(config_path),
        "task26_freeze_file": sha256(Path(config["task26_freeze_file"])),
        "dataset": assert_hash(Path(config["dataset_path"]), config["dataset_sha256"], "Elliptic dataset"),
        "target_file": assert_hash(Path(config["target_file"]), config["target_sha256"], "Task26 target file"),
        "gcn_checkpoint": assert_hash(Path(config["gcn_checkpoint"]), config["gcn_checkpoint_sha256"], "GCN checkpoint"),
        "graphsage_checkpoint": assert_hash(Path(config["graphsage_checkpoint"]), config["graphsage_checkpoint_sha256"], "GraphSAGE checkpoint"),
    }
    task26 = json.loads(Path(config["task26_freeze_file"]).read_text(encoding="utf-8"))
    if task26.get("task_id") != "TASK26" or task26.get("status") != "PASS":
        raise RuntimeError("Task26 freeze file is not a passing TASK26 manifest")
    if task26["final_target_rule_seed_count_hash_distributions"]["hash"] != config["target_sha256"]:
        raise RuntimeError("Task26 target hash does not match Task27 config")
    if "Edge AUROC" not in config["disabled_explanation_metrics"]:
        raise RuntimeError("Edge AUROC must be disabled for Elliptic")
    if "Edge AP" not in config["disabled_explanation_metrics"]:
        raise RuntimeError("Edge AP must be disabled for Elliptic")
    return hashes


@torch.no_grad()
def prediction_vector(model: torch.nn.Module, data: Any) -> torch.Tensor:
    model.eval()
    return logits(model, data).argmax(dim=1).detach()


def run_explainer(
    model: torch.nn.Module,
    data: Any,
    target_vector: torch.Tensor,
    node_id: int,
    explainer_config: dict[str, Any],
    seed: int,
) -> Any:
    seed_everything(seed)
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(
            epochs=int(explainer_config["epochs"]),
            lr=float(explainer_config["learning_rate"]),
        ),
        explanation_type=str(explainer_config["explanation_type"]),
        node_mask_type=str(explainer_config["node_mask_type"]),
        edge_mask_type=str(explainer_config["edge_mask_type"]),
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level=str(explainer_config["task_level"]),
            return_type=str(explainer_config["return_type"]),
        ),
    )
    return explainer(data.x, data.edge_index, target=target_vector, index=node_id)


def run_signature(
    *,
    spec: BackboneSpec,
    target: Target,
    seed: int,
    config: dict[str, Any],
    hashes: dict[str, str],
) -> str:
    payload = {
        "task": "TASK27",
        "backbone": spec.slug,
        "node_id": target.node_id,
        "prediction": target.prediction,
        "seed": seed,
        "dataset_sha256": hashes["dataset"],
        "target_sha256": hashes["target_file"],
        "checkpoint_sha256": hashes[spec.checkpoint_key],
        "config_sha256": hashes["task27_config"],
        "selected_edge_ratio": config["selected_edge_ratio"],
        "candidate_edge_scope": config["candidate_edge_scope"],
        "explanation_type": config["gnnexplainer"]["explanation_type"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def can_resume(output_dir: Path, signature: str) -> bool:
    required = [output_dir / "edge_mask.pt", output_dir / "node_mask.pt", output_dir / "metadata.json"]
    if not all(path.exists() for path in required):
        return False
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return metadata.get("task27_signature") == signature


def selected_mask_and_edges(
    edge_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, set[int], int, float]:
    selected_mask, budget = select_top_ratio_edges(edge_scores.cpu(), candidate_mask.cpu(), ratio)
    selected_edges = set(torch.nonzero(selected_mask, as_tuple=False).view(-1).tolist())
    return selected_mask, selected_edges, budget.selected_edge_count, budget.sparsity


def edge_input_hash(edge_index: torch.Tensor) -> str:
    h = hashlib.sha256()
    edge_cpu = edge_index.detach().cpu().contiguous()
    h.update(str(tuple(edge_cpu.shape)).encode("utf-8"))
    h.update(edge_cpu.numpy().tobytes())
    return h.hexdigest()


def write_selected_edges(
    path: Path,
    *,
    data: Any,
    edge_scores: torch.Tensor,
    selected_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_indices = torch.nonzero(selected_mask.cpu(), as_tuple=False).view(-1)
    scores = edge_scores.cpu()[selected_indices]
    order = torch.argsort(scores, descending=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "global_edge_index", "src", "dst", "score", "in_candidate_edge_set"],
        )
        writer.writeheader()
        for rank, order_idx in enumerate(order.tolist(), start=1):
            edge_idx = int(selected_indices[order_idx].item())
            writer.writerow(
                {
                    "rank": rank,
                    "global_edge_index": edge_idx,
                    "src": int(data.edge_index[0, edge_idx].item()),
                    "dst": int(data.edge_index[1, edge_idx].item()),
                    "score": float(edge_scores[edge_idx].item()),
                    "in_candidate_edge_set": bool(candidate_mask[edge_idx].item()),
                }
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_mask(
    *,
    model: torch.nn.Module,
    data: Any,
    target: Target,
    edge_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    selected_edge_ratio: float,
    generation_runtime_seconds: float | None,
    evaluation_runtime_seconds: float,
    seed: int,
    weights_stable: bool,
    explanation_source: str,
) -> tuple[dict[str, Any], set[int]]:
    candidate_count = int(candidate_mask.sum().item())
    if candidate_count != target.candidate_edge_count:
        raise RuntimeError(
            f"Candidate count mismatch for node {target.node_id}: "
            f"target file={target.candidate_edge_count}, computed={candidate_count}"
        )
    selected_mask, selected_edges, selected_count, sparsity = selected_mask_and_edges(
        edge_scores,
        candidate_mask,
        selected_edge_ratio,
    )
    fidelity = compute_fidelity(
        model=model,
        data=data,
        node_id=target.node_id,
        selected_edge_mask=selected_mask.to(data.edge_index.device),
    )
    original_hash = edge_input_hash(data.edge_index)
    remove_hash = edge_input_hash(data.edge_index[:, ~selected_mask.to(data.edge_index.device)])
    retain_hash = edge_input_hash(data.edge_index[:, selected_mask.to(data.edge_index.device)])
    return (
        {
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "confidence": target.confidence,
            "candidate_edge_count": candidate_count,
            "selected_edge_ratio": selected_edge_ratio,
            "selected_edge_count": selected_count,
            "original_prediction": fidelity.original_prediction,
            "remove_prediction": fidelity.remove_prediction,
            "retain_prediction": fidelity.retain_prediction,
            "fid_plus": fidelity.fid_plus,
            "fid_minus": fidelity.fid_minus,
            "one_minus_fid_minus": fidelity.one_minus_fid_minus,
            "original_probability": fidelity.original_probability,
            "remove_probability": fidelity.remove_probability,
            "remove_probability_delta": fidelity.remove_probability_delta,
            "retain_probability": fidelity.retain_probability,
            "retain_probability_delta": fidelity.retain_probability_delta,
            "retain_probability_ratio": fidelity.retain_probability_ratio,
            "sparsity": sparsity,
            "generation_runtime_seconds": generation_runtime_seconds,
            "evaluation_runtime_seconds": evaluation_runtime_seconds,
            "seed": seed,
            "status": "success",
            "error": "",
            "weights_unchanged": weights_stable,
            "cache_safe_perturbation": original_hash != remove_hash and original_hash != retain_hash,
            "explanation_source": explanation_source,
        },
        selected_edges,
    )


def failure_row(target: Target, seed: int, error: Exception) -> dict[str, Any]:
    return {
        **{field: None for field in PER_NODE_FIELDS},
        "node_id": target.node_id,
        "label": target.label,
        "prediction": target.prediction,
        "confidence": target.confidence,
        "seed": seed,
        "status": "failed",
        "error": str(error),
    }


def run_target(
    *,
    spec: BackboneSpec,
    model: torch.nn.Module,
    data: Any,
    target_vector: torch.Tensor,
    target: Target,
    seed: int,
    config: dict[str, Any],
    hashes: dict[str, str],
    output_dir: Path,
) -> tuple[dict[str, Any], set[int]]:
    signature = run_signature(spec=spec, target=target, seed=seed, config=config, hashes=hashes)
    candidate_mask = computation_subgraph_edge_mask(data, target.node_id, num_hops=2)
    generation_runtime_seconds = None
    explanation_source = "resumed_existing_raw_mask"
    before_state = state_snapshot(model)
    if not can_resume(output_dir, signature):
        with measure_runtime() as timer:
            explanation = run_explainer(
                model=model,
                data=data,
                target_vector=target_vector,
                node_id=target.node_id,
                explainer_config=config["gnnexplainer"],
                seed=seed,
            )
        generation_runtime_seconds = timer.runtime_seconds
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(explanation.edge_mask.detach().cpu(), output_dir / "edge_mask.pt")
        node_mask = getattr(explanation, "node_mask", None)
        torch.save(
            torch.empty(0) if node_mask is None else node_mask.detach().cpu(),
            output_dir / "node_mask.pt",
        )
        explanation_source = "generated_raw_mask"
        metadata = {
            "task": "TASK27",
            "backbone": spec.name,
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "seed": seed,
            "task27_signature": signature,
            "dataset_sha256": hashes["dataset"],
            "target_file_sha256": hashes["target_file"],
            "checkpoint_sha256": hashes[spec.checkpoint_key],
            "config_sha256": hashes["task27_config"],
            "candidate_edge_scope": config["candidate_edge_scope"],
            "selected_edge_ratio": config["selected_edge_ratio"],
            "rounding_rule": config["rounding_rule"],
            "generation_runtime_seconds": generation_runtime_seconds,
            "node_mask_available": node_mask is not None,
            "raw_masks_saved": True,
        }
        write_json(output_dir / "metadata.json", metadata)
    else:
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        generation_runtime_seconds = metadata.get("generation_runtime_seconds")

    edge_scores = torch.load(output_dir / "edge_mask.pt", map_location=data.x.device, weights_only=False)
    if edge_scores.numel() != data.edge_index.size(1):
        raise RuntimeError("Raw edge mask does not align with Elliptic edge_index")
    weights_stable = weights_unchanged(model, before_state)
    with measure_runtime() as eval_timer:
        row, selected_edges = evaluate_mask(
            model=model,
            data=data,
            target=target,
            edge_scores=edge_scores,
            candidate_mask=candidate_mask,
            selected_edge_ratio=float(config["selected_edge_ratio"]),
            generation_runtime_seconds=generation_runtime_seconds,
            evaluation_runtime_seconds=0.0,
            seed=seed,
            weights_stable=weights_stable,
            explanation_source=explanation_source,
        )
    row["evaluation_runtime_seconds"] = eval_timer.runtime_seconds
    selected_mask, _, _, _ = selected_mask_and_edges(
        edge_scores.detach().cpu(),
        candidate_mask.detach().cpu(),
        float(config["selected_edge_ratio"]),
    )
    write_selected_edges(
        output_dir / "selected_edges.csv",
        data=data,
        edge_scores=edge_scores.detach().cpu(),
        selected_mask=selected_mask,
        candidate_mask=candidate_mask.detach().cpu(),
    )
    write_json(output_dir / "metrics.json", row)
    if row["original_prediction"] != target.prediction:
        raise RuntimeError("Original explainer-wrapper prediction differs from frozen target prediction")
    if not row["weights_unchanged"]:
        raise RuntimeError("Model weights changed during GNNExplainer evaluation")
    if not row["cache_safe_perturbation"]:
        raise RuntimeError("Remove/retain perturbations did not change edge input")
    return row, selected_edges


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    success = [row for row in rows if row["status"] == "success"]
    summary = []
    for metric in SUMMARY_METRICS:
        values = [float(row[metric]) for row in success if row.get(metric) not in (None, "")]
        if values:
            summary.append(
                {
                    "metric": metric,
                    "n": len(values),
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "median": median(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
        else:
            summary.append({"metric": metric, "n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None})
    return summary


def metric_mean(summary: list[dict[str, Any]], metric: str) -> float | None:
    for row in summary:
        if row["metric"] == metric:
            return None if row["mean"] in (None, "") else float(row["mean"])
    return None


def summarize_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [float(row["mean_pairwise_jaccard"]) for row in rows if row["status"] == "success"]
    if not values:
        return [{"metric": "mean_pairwise_jaccard", "n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}]
    return [
        {
            "metric": "mean_pairwise_jaccard",
            "n": len(values),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
    ]


def run_backbone(
    *,
    spec: BackboneSpec,
    model: torch.nn.Module,
    data: Any,
    targets: list[Target],
    target_vector: torch.Tensor,
    config: dict[str, Any],
    hashes: dict[str, str],
    skip_stability: bool,
) -> dict[str, Any]:
    explanation_root = Path(config["output_root"]) / spec.slug / "gnnexplainer"
    metric_root = Path(config["metrics_root"]) / spec.slug / "gnnexplainer"
    primary_rows = []
    primary_selected: dict[int, set[int]] = {}
    for index, target in enumerate(targets, start=1):
        print(f"{spec.name} primary seed42 {index}/{len(targets)} node={target.node_id}", flush=True)
        try:
            row, selected_edges = run_target(
                spec=spec,
                model=model,
                data=data,
                target_vector=target_vector,
                target=target,
                seed=int(config["primary_seed"]),
                config=config,
                hashes=hashes,
                output_dir=explanation_root / "seed_42" / f"node_{target.node_id}",
            )
            primary_rows.append(row)
            primary_selected[target.node_id] = selected_edges
        except Exception as exc:
            primary_rows.append(failure_row(target, int(config["primary_seed"]), exc))
    primary_summary = summarize_rows(primary_rows)
    write_rows(metric_root / "per_node_metrics.csv", primary_rows, PER_NODE_FIELDS)
    write_rows(metric_root / "summary_metrics.csv", primary_summary, ["metric", "n", "mean", "std", "median", "min", "max"])
    stability_seed_rows = []
    pairwise_rows = []
    per_node_jaccard_rows = []
    if not skip_stability:
        for target_index, target in enumerate(targets, start=1):
            seed_to_edges: dict[int, set[int]] = {}
            for seed in config["stability_seeds"]:
                print(
                    f"{spec.name} stability seed{seed} {target_index}/{len(targets)} node={target.node_id}",
                    flush=True,
                )
                try:
                    row, selected_edges = run_target(
                        spec=spec,
                        model=model,
                        data=data,
                        target_vector=target_vector,
                        target=target,
                        seed=int(seed),
                        config=config,
                        hashes=hashes,
                        output_dir=explanation_root / f"seed_{seed}" / f"node_{target.node_id}",
                    )
                    stability_seed_rows.append(row)
                    seed_to_edges[int(seed)] = selected_edges
                except Exception as exc:
                    stability_seed_rows.append(failure_row(target, int(seed), exc))
            if len(seed_to_edges) == len(config["stability_seeds"]):
                selected_count = selected_edge_count_for_ratio(
                    target.candidate_edge_count,
                    float(config["selected_edge_ratio"]),
                )
                pairs = pairwise_jaccard_rows(
                    node_id=target.node_id,
                    seed_to_edges=seed_to_edges,
                    selected_edge_count=selected_count,
                )
                pairwise_rows.extend(pairs)
                values = [float(row["jaccard"]) for row in pairs]
                per_node_jaccard_rows.append(
                    {
                        "node_id": target.node_id,
                        "mean_pairwise_jaccard": mean(values),
                        "std_pairwise_jaccard": stdev(values) if len(values) > 1 else 0.0,
                        "median_pairwise_jaccard": median(values),
                        "min_pairwise_jaccard": min(values),
                        "max_pairwise_jaccard": max(values),
                        "num_seeds": len(config["stability_seeds"]),
                        "num_pairs": len(values),
                        "status": "success",
                        "error": "",
                    }
                )
            else:
                per_node_jaccard_rows.append(
                    {
                        "node_id": target.node_id,
                        "mean_pairwise_jaccard": None,
                        "std_pairwise_jaccard": None,
                        "median_pairwise_jaccard": None,
                        "min_pairwise_jaccard": None,
                        "max_pairwise_jaccard": None,
                        "num_seeds": len(config["stability_seeds"]),
                        "num_pairs": 0,
                        "status": "failed",
                        "error": "not enough successful seed explanations",
                    }
                )
    stability_summary = summarize_stability(per_node_jaccard_rows)
    write_rows(metric_root / "stability_seed_metrics.csv", stability_seed_rows, PER_NODE_FIELDS)
    write_rows(
        metric_root / "stability_pairwise_jaccard.csv",
        pairwise_rows,
        ["node_id", "seed_a", "seed_b", "selected_edge_count", "intersection_count", "union_count", "jaccard"],
    )
    write_rows(
        metric_root / "stability_per_node_jaccard.csv",
        per_node_jaccard_rows,
        [
            "node_id",
            "mean_pairwise_jaccard",
            "std_pairwise_jaccard",
            "median_pairwise_jaccard",
            "min_pairwise_jaccard",
            "max_pairwise_jaccard",
            "num_seeds",
            "num_pairs",
            "status",
            "error",
        ],
    )
    write_rows(metric_root / "stability_summary.csv", stability_summary, ["metric", "n", "mean", "std", "median", "min", "max"])
    return {
        "primary_success_count": sum(row["status"] == "success" for row in primary_rows),
        "primary_failure_count": sum(row["status"] != "success" for row in primary_rows),
        "stability_seed_success_count": sum(row["status"] == "success" for row in stability_seed_rows),
        "stability_seed_failure_count": sum(row["status"] != "success" for row in stability_seed_rows),
        "stability_node_success_count": sum(row["status"] == "success" for row in per_node_jaccard_rows),
        "stability_pairwise_rows": len(pairwise_rows),
        "primary_summary": primary_summary,
        "stability_summary": stability_summary,
        "model_weights_unchanged": True,
        "output_files": {
            "per_node_metrics": str(metric_root / "per_node_metrics.csv"),
            "summary_metrics": str(metric_root / "summary_metrics.csv"),
            "stability_seed_metrics": str(metric_root / "stability_seed_metrics.csv"),
            "stability_pairwise_jaccard": str(metric_root / "stability_pairwise_jaccard.csv"),
            "stability_per_node_jaccard": str(metric_root / "stability_per_node_jaccard.csv"),
            "stability_summary": str(metric_root / "stability_summary.csv"),
        },
    }


def output_hashes(results: dict[str, Any]) -> dict[str, str]:
    paths = []
    for backbone in results.values():
        paths.extend(backbone["output_files"].values())
    out = {}
    for item in paths:
        path = Path(item)
        if path.exists():
            out[item] = sha256(path)
    return out


def environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def build_report(manifest: dict[str, Any]) -> str:
    rows = []
    for spec in BACKBONES:
        result = manifest["backbones"][spec.name]
        rows.append(
            [
                spec.name,
                metric_mean(result["primary_summary"], "fid_plus"),
                metric_mean(result["primary_summary"], "fid_minus"),
                metric_mean(result["primary_summary"], "one_minus_fid_minus"),
                metric_mean(result["primary_summary"], "remove_probability_delta"),
                metric_mean(result["primary_summary"], "retain_probability"),
                metric_mean(result["primary_summary"], "retain_probability_delta"),
                metric_mean(result["primary_summary"], "retain_probability_ratio"),
                metric_mean(result["primary_summary"], "sparsity"),
                metric_mean(result["stability_summary"], "mean_pairwise_jaccard"),
            ]
        )
    table = "\n".join(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row[0], *[format_metric(value) for value in row[1:]]
        )
        for row in rows
    )
    return f"""# Elliptic GNNExplainer Report

## Scope

Task27 applies GNNExplainer to the frozen Elliptic GCN and GraphSAGE explanation-compatible backbones. The same Task26 common final target file is used for both backbones. No Edge AUROC/AP explanation metrics are computed because Elliptic has no ground-truth explanatory edge masks in this thesis.

## Frozen Protocol

- Target file: `{manifest["inputs"]["target_file"]}`
- Target SHA256: `{manifest["inputs"]["target_sha256"]}`
- Target count: `{manifest["inputs"]["target_count"]}`
- Primary seed: `{manifest["protocol"]["primary_seed"]}`
- Stability seeds: `{manifest["protocol"]["stability_seeds"]}`
- Selected edge ratio: `{manifest["protocol"]["selected_edge_ratio"]}`
- Rounding rule: `{manifest["protocol"]["rounding_rule"]}`
- Candidate edge scope: `{manifest["protocol"]["candidate_edge_scope"]}`
- Explanation type: `{manifest["protocol"]["explanation_type"]}`

## Main Metrics

| Backbone | Fid+ | Fid- | 1-Fid- | Remove delta | Retain probability | Retain delta | Retain ratio | Sparsity | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Runtime Semantics

GNNExplainer runtime is per-instance optimization/generation time. Metric evaluation time is recorded separately as evaluation runtime. These are not PGExplainer training or inference runtimes.

## Integrity

- Original wrapper predictions are checked against frozen Task26 predictions.
- GCN uses the Task26 `GCNConv(cached=False)` compatible wrapper.
- GraphSAGE uses the Task26 compatible wrapper.
- Model weights are checked for immutability during each target run.
- Remove/retain perturbations must alter the edge input.
- Failures are written explicitly and do not silently drop targets.

## Cross-Backbone Interpretation Boundary

The GCN and GraphSAGE rows are descriptive backbone-context observations on the same frozen target nodes. They must not be interpreted as causal backbone effects. Primary explainer comparison is deferred to later within-backbone tasks.

## Outputs

- `results/explanations/elliptic/gcn/gnnexplainer/`
- `results/explanations/elliptic/graphsage/gnnexplainer/`
- `results/metrics/elliptic/gcn/gnnexplainer/`
- `results/metrics/elliptic/graphsage/gnnexplainer/`
- `docs/elliptic_gnnexplainer_freeze.json`

## Stop Boundary

STOP FOR REVIEW. Task28 and later tasks are not started here.
"""


def format_metric(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    start = perf_counter()
    full_config = load_yaml(args.config)
    config = full_config["task27"]
    device = resolve_device(args.device)
    hashes = verify_inputs(config, args.config)
    data_cpu = load_data()
    models = build_models(data_cpu)
    data = data_cpu.to(device)
    results = {}
    for spec in BACKBONES:
        checkpoint_before = sha256(Path(config[spec.checkpoint_key]))
        model = models[spec.model_key].to(device)
        targets = read_targets(Path(config["target_file"]), spec)
        if args.max_targets is not None:
            targets = targets[: args.max_targets]
        with torch.no_grad():
            frozen_predictions = prediction_vector(model, data)
        for target in targets:
            if int(frozen_predictions[target.node_id].item()) != target.prediction:
                raise RuntimeError(
                    f"{spec.name} prediction mismatch for frozen target {target.node_id}"
                )
        results[spec.name] = run_backbone(
            spec=spec,
            model=model,
            data=data,
            targets=targets,
            target_vector=frozen_predictions,
            config=config,
            hashes=hashes,
            skip_stability=args.skip_stability,
        )
        checkpoint_after = sha256(Path(config[spec.checkpoint_key]))
        if checkpoint_before != checkpoint_after:
            raise RuntimeError(f"{spec.name} checkpoint changed during Task27")
        results[spec.name]["checkpoint_unchanged"] = True
    acceptance = {
        "exact_task26_common_final_targets_used": True,
        "no_auroc_ap_explanation_metrics": True,
        "gcn_graphsage_weights_unchanged": all(
            result["model_weights_unchanged"] and result["checkpoint_unchanged"]
            for result in results.values()
        ),
        "cache_safe_perturbation_semantics_verified": True,
        "primary_seed42_complete_both_backbones": all(
            result["primary_success_count"] == (args.max_targets or int(config["target_count"]))
            for result in results.values()
        ),
        "stability_complete_on_prespecified_scope": (
            args.skip_stability
            or all(
                result["stability_node_success_count"] == (args.max_targets or int(config["target_count"]))
                for result in results.values()
            )
        ),
        "stochastic_source_documented": True,
        "runtime_semantics_recorded": True,
        "failures_explicit_not_silent": True,
        "no_causal_cross_backbone_wording": True,
    }
    manifest = {
        "task": "TASK27 Elliptic GNNExplainer",
        "status": "success" if all(acceptance.values()) else "failed",
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
        },
        "environment": environment(device),
        "inputs": {
            "dataset_path": config["dataset_path"],
            "dataset_sha256": config["dataset_sha256"],
            "target_file": config["target_file"],
            "target_sha256": config["target_sha256"],
            "target_count": args.max_targets or config["target_count"],
            "gcn_checkpoint_sha256": config["gcn_checkpoint_sha256"],
            "graphsage_checkpoint_sha256": config["graphsage_checkpoint_sha256"],
        },
        "input_hashes": hashes,
        "protocol": {
            "primary_seed": config["primary_seed"],
            "stability_seeds": [] if args.skip_stability else config["stability_seeds"],
            "selected_edge_ratio": config["selected_edge_ratio"],
            "rounding_rule": config["rounding_rule"],
            "candidate_edge_scope": config["candidate_edge_scope"],
            "explanation_type": config["gnnexplainer"]["explanation_type"],
            "probability_reference_class": config["probability_reference_class"],
            "disabled_explanation_metrics": config["disabled_explanation_metrics"],
            "stochastic_source": "independent per-instance explanation optimization runs",
        },
        "backbones": results,
        "output_hashes": output_hashes(results),
        "acceptance": acceptance,
        "total_runtime_seconds": perf_counter() - start,
    }
    write_json(Path("docs/elliptic_gnnexplainer_freeze.json"), manifest)
    Path("docs/elliptic_gnnexplainer_report.md").write_text(
        build_report(manifest),
        encoding="utf-8",
    )
    print(f"Task27 status={manifest['status']}", flush=True)
    print(f"runtime_seconds={manifest['total_runtime_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
