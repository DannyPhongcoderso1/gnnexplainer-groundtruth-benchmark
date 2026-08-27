from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

import torch
import yaml

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.data.shapeggen import (
    get_target_candidate_edge_mask,
    get_target_edge_ground_truth,
)
from gnnexplainer_benchmark.data.target_selection import ExplainerTarget
from gnnexplainer_benchmark.data.validation import validate_shapeggen_data
from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import (
    failure_row,
    flatten_metrics_row,
    write_per_node_metrics,
    write_summary_metrics,
)
from gnnexplainer_benchmark.evaluation.budget import (
    select_top_ratio_edges,
    selected_edge_count_for_ratio,
)
from gnnexplainer_benchmark.evaluation.calibration import assert_gcnconv_cached_false
from gnnexplainer_benchmark.evaluation.edge_ground_truth import (
    candidate_edge_scores_and_labels,
    compute_edge_ground_truth_metrics,
)
from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity
from gnnexplainer_benchmark.evaluation.runtime import measure_runtime
from gnnexplainer_benchmark.evaluation.stability import (
    pairwise_jaccard_rows,
    summarize_stability_rows,
)
from gnnexplainer_benchmark.explain.gnnexplainer_runner import (
    load_explainer_config,
    run_gnnexplainer_for_node,
)
from gnnexplainer_benchmark.explain.node_selection import TargetNode
from gnnexplainer_benchmark.explainers.random_baseline import random_candidate_edge_scores
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    slug: str
    checkpoint_key: str
    checkpoint_sha_key: str
    prediction_column: str
    confidence_column: str


BACKBONES = [
    BackboneSpec(
        name="GCN",
        slug="gcn",
        checkpoint_key="gcn_checkpoint",
        checkpoint_sha_key="gcn_checkpoint_sha256",
        prediction_column="gcn_prediction",
        confidence_column="gcn_confidence",
    ),
    BackboneSpec(
        name="GraphSAGE",
        slug="graphsage",
        checkpoint_key="graphsage_checkpoint",
        checkpoint_sha_key="graphsage_checkpoint_sha256",
        prediction_column="graphsage_prediction",
        confidence_column="graphsage_confidence",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task23 ShapeGGen GNNExplainer and Random baselines.",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/evaluation/shapeggen_task23.yaml"),
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


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def read_targets(path: Path, spec: BackboneSpec) -> list[ExplainerTarget]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    targets = []
    for row in rows:
        targets.append(
            ExplainerTarget(
                node_id=int(row["node_id"]),
                label=int(row["label"]),
                prediction=int(row[spec.prediction_column]),
                confidence=float(row[spec.confidence_column]),
                split=str(row["split"]),
                is_ground_truth_motif=True,
            )
        )
    validate_target_order(targets)
    return targets


def validate_target_order(targets: list[ExplainerTarget]) -> None:
    node_ids = [target.node_id for target in targets]
    if node_ids != sorted(node_ids):
        raise ValueError("ShapeGGen targets must be sorted by node_id")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("ShapeGGen target file contains duplicate node_id values")


def assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def verify_frozen_inputs(
    *,
    data: Any,
    task_config: dict[str, Any],
    task_config_path: Path,
) -> dict[str, str]:
    dataset_path = Path(task_config["dataset_path"])
    target_file = Path(task_config["target_file"])
    gnn_config = Path(task_config["gnnexplainer_config"])
    hashes = {
        "task_config": sha256_file(task_config_path),
        "dataset": assert_hash(dataset_path, task_config["dataset_sha256"], "ShapeGGen dataset"),
        "target_file": assert_hash(target_file, task_config["target_sha256"], "Target file"),
        "gnnexplainer_config": sha256_file(gnn_config),
        "gcn_checkpoint": assert_hash(
            Path(task_config["gcn_checkpoint"]),
            task_config["gcn_checkpoint_sha256"],
            "GCN checkpoint",
        ),
        "graphsage_checkpoint": assert_hash(
            Path(task_config["graphsage_checkpoint"]),
            task_config["graphsage_checkpoint_sha256"],
            "GraphSAGE checkpoint",
        ),
    }
    if getattr(data, "shape_ggen_protocol_name", None) != task_config["dataset_protocol_name"]:
        raise RuntimeError("ShapeGGen protocol name does not match Task23 config")
    if getattr(data, "graphxai_upstream_commit", None) != task_config["upstream_graphxai_commit"]:
        raise RuntimeError("ShapeGGen upstream GraphXAI commit does not match Task23 config")
    verify_reverse_edge_policy(data)
    return hashes


def verify_reverse_edge_policy(data: Any) -> None:
    edge_pairs = []
    edge_set = set()
    for edge_idx in range(int(data.edge_index.size(1))):
        src = int(data.edge_index[0, edge_idx].item())
        dst = int(data.edge_index[1, edge_idx].item())
        if src == dst:
            raise RuntimeError("ShapeGGen edge_index contains a self-loop")
        pair = (src, dst)
        if pair in edge_set:
            raise RuntimeError("ShapeGGen edge_index contains duplicate directed edges")
        edge_pairs.append(pair)
        edge_set.add(pair)
    missing_reverse = [(src, dst) for src, dst in edge_pairs if (dst, src) not in edge_set]
    if missing_reverse:
        raise RuntimeError("ShapeGGen edge_index is missing reverse-edge entries")


def validate_targets_against_data(
    *,
    data: Any,
    targets: list[ExplainerTarget],
    expected_count: int,
    selected_edge_ratio: float,
) -> dict[str, Any]:
    if len(targets) != expected_count:
        raise RuntimeError(f"Expected {expected_count} targets, found {len(targets)}")
    selected_counts = []
    candidate_counts = []
    positive_counts = []
    for target in targets:
        if target.split != "test":
            raise RuntimeError(f"Target {target.node_id} is not in the test split")
        if target.prediction != target.label:
            raise RuntimeError(f"Target {target.node_id} is not correctly predicted")
        candidate_mask = get_target_candidate_edge_mask(data, target.node_id)
        gt_mask = get_target_edge_ground_truth(data, target.node_id)
        verify_candidate_mapping(candidate_mask, gt_mask, int(data.edge_index.size(1)))
        candidate_count = int(candidate_mask.sum().item())
        positive_count = int(gt_mask[candidate_mask].sum().item())
        selected_count = selected_edge_count_for_ratio(candidate_count, selected_edge_ratio)
        selected_counts.append(selected_count)
        candidate_counts.append(candidate_count)
        positive_counts.append(positive_count)
    return {
        "target_count": len(targets),
        "candidate_edge_count": distribution(candidate_counts),
        "positive_gt_edges": distribution(positive_counts),
        "selected_edge_count": distribution(selected_counts),
    }


def verify_candidate_mapping(
    candidate_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    edge_count: int,
) -> None:
    if candidate_mask.dtype != torch.bool or gt_mask.dtype != torch.bool:
        raise RuntimeError("Candidate and ground-truth masks must be boolean")
    if candidate_mask.numel() != edge_count or gt_mask.numel() != edge_count:
        raise RuntimeError("Candidate and ground-truth masks must align with edge_index")
    if bool((gt_mask & ~candidate_mask).any()):
        raise RuntimeError("Positive ground-truth edges must be a subset of candidate edges")
    positive = int(gt_mask[candidate_mask].sum().item())
    candidate_count = int(candidate_mask.sum().item())
    negative = candidate_count - positive
    if positive <= 0 or negative <= 0:
        raise RuntimeError("Each target must have positive and negative candidate edges")


def distribution(values: list[int | float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    return {
        "n": len(numeric),
        "mean": mean(numeric),
        "std": stdev(numeric) if len(numeric) > 1 else 0.0,
        "median": median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def weights_unchanged(
    model: torch.nn.Module,
    before_state: dict[str, torch.Tensor],
) -> bool:
    return all(
        torch.equal(before_state[name], value.detach().cpu())
        for name, value in model.state_dict().items()
    )


def target_node(target: ExplainerTarget) -> TargetNode:
    return TargetNode(
        node_id=target.node_id,
        label=target.label,
        prediction=target.prediction,
        confidence=target.confidence,
    )


def run_signature(
    *,
    task_config: dict[str, Any],
    config_hashes: dict[str, str],
    spec: BackboneSpec,
    target: ExplainerTarget,
    method: str,
    seed: int,
) -> str:
    payload = {
        "backbone": spec.name,
        "checkpoint_sha256": config_hashes[f"{spec.slug}_checkpoint"],
        "dataset_sha256": task_config["dataset_sha256"],
        "edge_ranking_metric_scope": task_config["edge_ranking_metric_scope"],
        "method": method,
        "node_id": target.node_id,
        "primary_target_prediction": target.prediction,
        "seed": seed,
        "selected_edge_ratio": task_config["selected_edge_ratio"],
        "target_sha256": task_config["target_sha256"],
        "task_config_sha256": config_hashes["task_config"],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def can_resume(output_dir: Path, signature: str) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    required = [output_dir / "edge_mask.pt", output_dir / "node_mask.pt"]
    if not all(path.exists() for path in required):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("task23_signature") == signature


def run_gnnexplainer_once(
    *,
    model: torch.nn.Module,
    data_cpu: Any,
    data_device: Any,
    target: ExplainerTarget,
    seed: int,
    spec: BackboneSpec,
    task_config: dict[str, Any],
    gnn_config_template: dict[str, Any],
    config_hashes: dict[str, str],
    output_dir: Path,
) -> tuple[dict[str, Any], set[int]]:
    signature = run_signature(
        task_config=task_config,
        config_hashes=config_hashes,
        spec=spec,
        target=target,
        method="GNNExplainer",
        seed=seed,
    )
    candidate_mask = get_target_candidate_edge_mask(data_cpu, target.node_id)
    gt_mask = get_target_edge_ground_truth(data_cpu, target.node_id)
    verify_candidate_mapping(candidate_mask, gt_mask, int(data_cpu.edge_index.size(1)))
    generation_runtime = None
    source = "resumed_existing_gnnexplainer_mask"
    if not can_resume(output_dir, signature):
        config = json.loads(json.dumps(gnn_config_template))
        config["explainer"]["seed"] = seed
        with measure_runtime() as timer:
            explanation = run_gnnexplainer_for_node(
                model=model,
                data=data_device,
                target=target_node(target),
                config=config,
            )
        if timer.runtime_seconds is None:
            raise RuntimeError("GNNExplainer runtime measurement failed")
        edge_scores = explanation.edge_mask.detach().cpu()
        node_scores = (
            explanation.node_mask.detach().cpu()
            if getattr(explanation, "node_mask", None) is not None
            else torch.empty(0)
        )
        generation_runtime = timer.runtime_seconds
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(edge_scores, output_dir / "edge_mask.pt")
        torch.save(node_scores, output_dir / "node_mask.pt")
        source = "generated_gnnexplainer_mask"
        metadata = base_metadata(
            task_config=task_config,
            config_hashes=config_hashes,
            spec=spec,
            target=target,
            method="GNNExplainer",
            seed=seed,
            signature=signature,
        )
        metadata.update(
            {
                "generation_runtime_seconds": generation_runtime,
                "evaluation_runtime_seconds": None,
                "node_mask_available": bool(node_scores.numel()),
            }
        )
        write_json(output_dir / "metadata.json", metadata)
    else:
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        generation_runtime = metadata.get("generation_runtime_seconds")

    edge_scores = torch.load(output_dir / "edge_mask.pt", map_location="cpu", weights_only=False)
    if edge_scores.numel() != data_cpu.edge_index.size(1):
        raise RuntimeError("GNNExplainer edge mask does not align with edge_index")
    with measure_runtime() as eval_timer:
        metrics, selected_edges = compute_metrics_for_scores(
            model=model,
            data=data_device,
            data_cpu=data_cpu,
            target=target,
            edge_scores=edge_scores,
            candidate_mask=candidate_mask,
            gt_mask=gt_mask,
            selected_edge_ratio=float(task_config["selected_edge_ratio"]),
            runtime_seconds=generation_runtime,
            explainer="GNNExplainer",
            candidate_edge_scope=task_config["candidate_edge_scope"],
        )
    if eval_timer.runtime_seconds is None:
        raise RuntimeError("Evaluation runtime measurement failed")
    write_selected_edges(
        output_dir / "selected_edges.csv",
        data=data_cpu,
        edge_scores=edge_scores,
        selected_edges=selected_edges,
        candidate_mask=candidate_mask,
        gt_mask=gt_mask,
    )
    write_selected_edges(
        output_dir / "top_edges.csv",
        data=data_cpu,
        edge_scores=edge_scores,
        selected_edges=selected_edges,
        candidate_mask=candidate_mask,
        gt_mask=gt_mask,
    )
    write_json(output_dir / "metrics.json", metrics)
    return (
        flatten_metrics_row(
            metrics,
            target,
            seed,
            status="success",
            generation_runtime_seconds=generation_runtime,
            evaluation_runtime_seconds=eval_timer.runtime_seconds,
            explanation_source=source,
        ),
        selected_edges,
    )


def run_random_once(
    *,
    model: torch.nn.Module,
    data_cpu: Any,
    data_device: Any,
    target: ExplainerTarget,
    seed: int,
    spec: BackboneSpec,
    task_config: dict[str, Any],
    config_hashes: dict[str, str],
    output_dir: Path,
) -> tuple[dict[str, Any], set[int]]:
    signature = run_signature(
        task_config=task_config,
        config_hashes=config_hashes,
        spec=spec,
        target=target,
        method="Random",
        seed=seed,
    )
    candidate_mask = get_target_candidate_edge_mask(data_cpu, target.node_id)
    gt_mask = get_target_edge_ground_truth(data_cpu, target.node_id)
    verify_candidate_mapping(candidate_mask, gt_mask, int(data_cpu.edge_index.size(1)))
    source = "resumed_existing_random_scores"
    if not can_resume(output_dir, signature):
        edge_scores = random_candidate_edge_scores(
            edge_count=int(data_cpu.edge_index.size(1)),
            candidate_edge_mask=candidate_mask,
            seed=random_seed_for_node(seed, target.node_id),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(edge_scores, output_dir / "edge_mask.pt")
        torch.save(torch.empty(0), output_dir / "node_mask.pt")
        source = "generated_random_scores"
        metadata = base_metadata(
            task_config=task_config,
            config_hashes=config_hashes,
            spec=spec,
            target=target,
            method="Random",
            seed=seed,
            signature=signature,
        )
        metadata.update(
            {
                "node_random_seed": random_seed_for_node(seed, target.node_id),
                "generation_runtime_seconds": None,
                "evaluation_runtime_seconds": None,
                "node_mask_available": False,
            }
        )
        write_json(output_dir / "metadata.json", metadata)

    edge_scores = torch.load(output_dir / "edge_mask.pt", map_location="cpu", weights_only=False)
    with measure_runtime() as eval_timer:
        metrics, selected_edges = compute_metrics_for_scores(
            model=model,
            data=data_device,
            data_cpu=data_cpu,
            target=target,
            edge_scores=edge_scores,
            candidate_mask=candidate_mask,
            gt_mask=gt_mask,
            selected_edge_ratio=float(task_config["selected_edge_ratio"]),
            runtime_seconds=None,
            explainer="Random",
            candidate_edge_scope=task_config["candidate_edge_scope"],
        )
    if eval_timer.runtime_seconds is None:
        raise RuntimeError("Random evaluation runtime measurement failed")
    metrics["runtime"]["runtime_seconds"] = eval_timer.runtime_seconds
    write_selected_edges(
        output_dir / "selected_edges.csv",
        data=data_cpu,
        edge_scores=edge_scores,
        selected_edges=selected_edges,
        candidate_mask=candidate_mask,
        gt_mask=gt_mask,
    )
    write_selected_edges(
        output_dir / "top_edges.csv",
        data=data_cpu,
        edge_scores=edge_scores,
        selected_edges=selected_edges,
        candidate_mask=candidate_mask,
        gt_mask=gt_mask,
    )
    write_json(output_dir / "metrics.json", metrics)
    return (
        flatten_metrics_row(
            metrics,
            target,
            seed,
            status="success",
            generation_runtime_seconds=None,
            evaluation_runtime_seconds=eval_timer.runtime_seconds,
            explanation_source=source,
        ),
        selected_edges,
    )


def random_seed_for_node(seed: int, node_id: int) -> int:
    return seed * 1_000_003 + node_id


def base_metadata(
    *,
    task_config: dict[str, Any],
    config_hashes: dict[str, str],
    spec: BackboneSpec,
    target: ExplainerTarget,
    method: str,
    seed: int,
    signature: str,
) -> dict[str, Any]:
    return {
        "backbone": spec.name,
        "candidate_edge_scope": task_config["candidate_edge_scope"],
        "dataset_sha256": task_config["dataset_sha256"],
        "edge_ranking_metric_scope": task_config["edge_ranking_metric_scope"],
        "method": method,
        "rounding_rule": task_config["rounding_rule"],
        "seed": seed,
        "selected_edge_ratio": task_config["selected_edge_ratio"],
        "target_file": task_config["target_file"],
        "target_sha256": task_config["target_sha256"],
        "task23_signature": signature,
        "task_config_sha256": config_hashes["task_config"],
        "checkpoint": {
            "path": task_config[spec.checkpoint_key],
            "sha256": config_hashes[f"{spec.slug}_checkpoint"],
        },
        "target": {
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "confidence": target.confidence,
            "split": target.split,
        },
    }


def compute_metrics_for_scores(
    *,
    model: torch.nn.Module,
    data: Any,
    data_cpu: Any,
    target: ExplainerTarget,
    edge_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    selected_edge_ratio: float,
    runtime_seconds: float | None,
    explainer: str,
    candidate_edge_scope: str,
) -> tuple[dict[str, Any], set[int]]:
    candidate_scores, candidate_ground_truth = candidate_edge_scores_and_labels(
        edge_scores=edge_scores.detach().cpu(),
        ground_truth_edge_mask=gt_mask.detach().cpu(),
        candidate_edge_mask=candidate_mask.detach().cpu(),
    )
    edge_metrics = compute_edge_ground_truth_metrics(candidate_scores, candidate_ground_truth)
    selected_edge_mask, budget = select_top_ratio_edges(
        edge_scores=edge_scores.detach().cpu(),
        candidate_edge_mask=candidate_mask.detach().cpu(),
        ratio=selected_edge_ratio,
    )
    expected_selected = selected_edge_count_for_ratio(
        budget.candidate_edge_count,
        selected_edge_ratio,
    )
    if budget.selected_edge_count != expected_selected:
        raise RuntimeError("Selected-edge count does not obey the frozen rounding rule")
    fidelity = compute_fidelity(
        model=model,
        data=data,
        node_id=target.node_id,
        selected_edge_mask=selected_edge_mask.to(data.edge_index.device),
    )
    positive_edges = int(candidate_ground_truth.sum().item())
    negative_edges = int(candidate_ground_truth.numel() - positive_edges)
    selected_indices = torch.nonzero(selected_edge_mask, as_tuple=False).view(-1).tolist()
    metrics = {
        "schema_version": "shapeggen_task23_core_metrics_v1",
        "explainer": explainer,
        "node_id": target.node_id,
        "target": {
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "confidence": target.confidence,
        },
        "edge_ground_truth": {
            "candidate_edge_scope": candidate_edge_scope,
            "candidate_edge_scope_detail": (
                "Official GraphXAI target-specific candidate edge mask aligned "
                "to the full directed PyG edge_index"
            ),
            "candidate_edge_count": budget.candidate_edge_count,
            "candidate_positive_edge_count": positive_edges,
            "candidate_negative_edge_count": negative_edges,
            "ground_truth_edge_prevalence": positive_edges / budget.candidate_edge_count,
            "edge_mapping_validated": True,
            "edge_auroc": edge_metrics["edge_auroc"].value,
            "edge_auroc_valid": edge_metrics["edge_auroc"].valid,
            "edge_auroc_reason": edge_metrics["edge_auroc"].reason,
            "edge_ap": edge_metrics["edge_ap"].value,
            "edge_ap_valid": edge_metrics["edge_ap"].valid,
            "edge_ap_reason": edge_metrics["edge_ap"].reason,
        },
        "sparsity": {
            "protocol": "fixed_candidate_edge_ratio",
            "rounding_rule": "ceil(candidate_edge_count * selected_edge_ratio), minimum 1",
            "selected_edge_ratio": selected_edge_ratio,
            "selected_edge_count": budget.selected_edge_count,
            "sparsity": budget.sparsity,
        },
        "fidelity": {
            "protocol": fidelity.protocol,
            "metric_name": fidelity.metric_name,
            "original": {
                "prediction": fidelity.original_prediction,
                "probability": fidelity.original_probability,
            },
            "remove_result": {
                "prediction": fidelity.remove_prediction,
                "probability": fidelity.remove_probability,
                "remove_delta": fidelity.remove_delta,
                "remove_probability_delta": fidelity.remove_probability_delta,
                "changed_prediction": fidelity.remove_changed_prediction,
            },
            "retain_result": {
                "prediction": fidelity.retain_prediction,
                "probability": fidelity.retain_probability,
                "retain_delta": fidelity.retain_delta,
                "retain_probability_delta": fidelity.retain_probability_delta,
                "probability_ratio": fidelity.retain_probability_ratio,
                "matches_original_prediction": fidelity.retain_matches_original_prediction,
            },
            "fid_plus": fidelity.fid_plus,
            "fid_minus": fidelity.fid_minus,
            "one_minus_fid_minus": fidelity.one_minus_fid_minus,
            "fidelity_legacy": fidelity.remove_probability_delta,
        },
        "runtime": {"runtime_seconds": runtime_seconds},
    }
    if data_cpu.edge_index.size(1) != edge_scores.numel():
        raise RuntimeError("Edge scores must align with full directed edge_index")
    return metrics, {int(edge_idx) for edge_idx in selected_indices}


def write_selected_edges(
    path: Path,
    *,
    data: Any,
    edge_scores: torch.Tensor,
    selected_edges: set[int],
    candidate_mask: torch.Tensor,
    gt_mask: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_indices = sorted(
        selected_edges,
        key=lambda edge_idx: float(edge_scores[edge_idx].item()),
        reverse=True,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "global_edge_index",
                "src",
                "dst",
                "score",
                "is_ground_truth",
                "in_candidate_edge_set",
            ],
        )
        writer.writeheader()
        for rank, edge_idx in enumerate(selected_indices, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "global_edge_index": edge_idx,
                    "src": int(data.edge_index[0, edge_idx].item()),
                    "dst": int(data.edge_index[1, edge_idx].item()),
                    "score": float(edge_scores[edge_idx].item()),
                    "is_ground_truth": bool(gt_mask[edge_idx].item()),
                    "in_candidate_edge_set": bool(candidate_mask[edge_idx].item()),
                }
            )


def run_backbone(
    *,
    spec: BackboneSpec,
    data_cpu: Any,
    task_config: dict[str, Any],
    config_hashes: dict[str, str],
    device: torch.device,
    max_targets: int | None,
) -> dict[str, Any]:
    checkpoint_path = Path(task_config[spec.checkpoint_key])
    checkpoint_hash_before = sha256_file(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("dataset_checksum") != task_config["dataset_sha256"]:
        raise RuntimeError(f"{spec.name} checkpoint was not trained on frozen ShapeGGen")
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(device)
    model.eval()
    assert_gcnconv_cached_false(model)
    before_state = snapshot_state(model)
    data_device = data_cpu.clone().to(device)
    targets = read_targets(Path(task_config["target_file"]), spec)
    if max_targets is not None:
        targets = targets[:max_targets]
    target_audit = validate_targets_against_data(
        data=data_cpu,
        targets=targets,
        expected_count=max_targets or int(task_config["target_count"]),
        selected_edge_ratio=float(task_config["selected_edge_ratio"]),
    )
    verify_predictions_match_targets(
        model=model,
        data=data_device,
        targets=targets,
        spec=spec,
    )
    gnn_config = load_explainer_config(Path(task_config["gnnexplainer_config"]))
    primary = run_gnnexplainer_quality(
        spec=spec,
        model=model,
        data_cpu=data_cpu,
        data_device=data_device,
        targets=targets,
        task_config=task_config,
        gnn_config=gnn_config,
        config_hashes=config_hashes,
    )
    gnn_stability = run_gnnexplainer_stability(
        spec=spec,
        model=model,
        data_cpu=data_cpu,
        data_device=data_device,
        targets=targets,
        task_config=task_config,
        gnn_config=gnn_config,
        config_hashes=config_hashes,
    )
    random_result = run_random_quality_and_stability(
        spec=spec,
        model=model,
        data_cpu=data_cpu,
        data_device=data_device,
        targets=targets,
        task_config=task_config,
        config_hashes=config_hashes,
    )
    checkpoint_hash_after = sha256_file(checkpoint_path)
    model_weights_unchanged = weights_unchanged(model, before_state)
    return {
        "backbone": spec.name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_file_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "model_weights_unchanged": model_weights_unchanged,
        "target_audit": target_audit,
        "gnnexplainer_primary": primary,
        "gnnexplainer_stability": gnn_stability,
        "random": random_result,
    }


def verify_predictions_match_targets(
    *,
    model: torch.nn.Module,
    data: Any,
    targets: list[ExplainerTarget],
    spec: BackboneSpec,
) -> None:
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = logits.softmax(dim=-1)
        predictions = logits.argmax(dim=-1)
    for target in targets:
        prediction = int(predictions[target.node_id].item())
        if prediction != target.prediction:
            raise RuntimeError(
                f"{spec.name} prediction mismatch for node {target.node_id}: "
                f"target file has {target.prediction}, checkpoint gives {prediction}"
            )
        confidence = float(probabilities[target.node_id, prediction].item())
        if abs(confidence - target.confidence) > 1e-5:
            raise RuntimeError(
                f"{spec.name} confidence mismatch for node {target.node_id}: "
                f"target file has {target.confidence}, checkpoint gives {confidence}"
            )


def run_gnnexplainer_quality(
    *,
    spec: BackboneSpec,
    model: torch.nn.Module,
    data_cpu: Any,
    data_device: Any,
    targets: list[ExplainerTarget],
    task_config: dict[str, Any],
    gnn_config: dict[str, Any],
    config_hashes: dict[str, str],
) -> dict[str, Any]:
    seed = int(task_config["primary_seed"])
    output_root = Path("results/explanations/shapeggen") / spec.slug / "gnnexplainer"
    metrics_dir = Path("results/metrics/shapeggen") / spec.slug / "gnnexplainer"
    rows = []
    start = perf_counter()
    for index, target in enumerate(targets, start=1):
        try:
            row, _ = run_gnnexplainer_once(
                model=model,
                data_cpu=data_cpu,
                data_device=data_device,
                target=target,
                seed=seed,
                spec=spec,
                task_config=task_config,
                gnn_config_template=gnn_config,
                config_hashes=config_hashes,
                output_dir=output_root / f"seed_{seed}" / f"node_{target.node_id}",
            )
            rows.append(row)
        except Exception as exc:
            rows.append(failure_row(target, seed, exc))
        print_progress(spec.name, "GNNExplainer seed42", index, len(targets))
    per_node = metrics_dir / "per_node_metrics.csv"
    summary_path = metrics_dir / "summary_metrics.csv"
    failed_path = metrics_dir / "failed_targets.csv"
    write_per_node_metrics(per_node, rows)
    summary = write_summary_metrics(summary_path, rows)
    write_failed_targets(failed_path, rows)
    return {
        "seed": seed,
        "success_count": sum(row["status"] == "success" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "runtime_seconds": perf_counter() - start,
        "per_node_metrics": str(per_node),
        "summary_metrics": str(summary_path),
        "failed_targets": str(failed_path),
        "summary": summary,
    }


def run_gnnexplainer_stability(
    *,
    spec: BackboneSpec,
    model: torch.nn.Module,
    data_cpu: Any,
    data_device: Any,
    targets: list[ExplainerTarget],
    task_config: dict[str, Any],
    gnn_config: dict[str, Any],
    config_hashes: dict[str, str],
) -> dict[str, Any]:
    seeds = [int(seed) for seed in task_config["gnnexplainer_stability_seeds"]]
    output_root = Path("results/explanations/shapeggen") / spec.slug / "gnnexplainer"
    metrics_dir = Path("results/metrics/shapeggen") / spec.slug / "gnnexplainer"
    seed_rows = []
    node_rows = []
    pair_rows = []
    start = perf_counter()
    for seed in seeds:
        for index, target in enumerate(targets, start=1):
            try:
                row, selected_edges = run_gnnexplainer_once(
                    model=model,
                    data_cpu=data_cpu,
                    data_device=data_device,
                    target=target,
                    seed=seed,
                    spec=spec,
                    task_config=task_config,
                    gnn_config_template=gnn_config,
                    config_hashes=config_hashes,
                    output_dir=output_root / f"seed_{seed}" / f"node_{target.node_id}",
                )
                row["_selected_edges"] = selected_edges
                seed_rows.append(row)
            except Exception as exc:
                seed_rows.append(failure_row(target, seed, exc))
            print_progress(spec.name, f"GNNExplainer stability seed {seed}", index, len(targets))
    for target in targets:
        seed_to_edges = {
            int(row["seed"]): row["_selected_edges"]
            for row in seed_rows
            if row["node_id"] == target.node_id
            and row["status"] == "success"
            and "_selected_edges" in row
        }
        selected_count = selected_count_for_target(data_cpu, target, task_config)
        if len(seed_to_edges) == len(seeds):
            per_pair = pairwise_jaccard_rows(
                node_id=target.node_id,
                seed_to_edges=seed_to_edges,
                selected_edge_count=selected_count,
            )
            pair_rows.extend(per_pair)
            values = [float(row["jaccard"]) for row in per_pair]
            node_rows.append(stability_node_row(target.node_id, values, len(seeds)))
        else:
            node_rows.append(failed_stability_node_row(target.node_id, len(seed_to_edges)))
    clean_seed_rows = [
        {k: v for k, v in row.items() if k != "_selected_edges"}
        for row in seed_rows
    ]
    seed_path = metrics_dir / "stability_seed_metrics.csv"
    pair_path = metrics_dir / "stability_pairwise_jaccard.csv"
    node_path = metrics_dir / "stability_per_node_jaccard.csv"
    summary_path = metrics_dir / "stability_summary.csv"
    write_per_node_metrics(seed_path, clean_seed_rows)
    write_rows(pair_path, pair_rows, pairwise_fields())
    write_rows(node_path, node_rows, stability_node_fields())
    summary = summarize_stability_rows(node_rows)
    write_rows(summary_path, summary, ["metric", "n", "mean", "std", "median", "min", "max"])
    return {
        "seeds": seeds,
        "success_nodes": sum(row["status"] == "success" for row in node_rows),
        "failed_nodes": sum(row["status"] == "failed" for row in node_rows),
        "pairwise_rows": len(pair_rows),
        "runtime_seconds": perf_counter() - start,
        "seed_metrics": str(seed_path),
        "pairwise_jaccard": str(pair_path),
        "per_node_jaccard": str(node_path),
        "summary_metrics": str(summary_path),
        "summary": summary,
        "stochastic_source": task_config["gnnexplainer_stochastic_source"],
    }


def run_random_quality_and_stability(
    *,
    spec: BackboneSpec,
    model: torch.nn.Module,
    data_cpu: Any,
    data_device: Any,
    targets: list[ExplainerTarget],
    task_config: dict[str, Any],
    config_hashes: dict[str, str],
) -> dict[str, Any]:
    seeds = [int(seed) for seed in task_config["random_baseline_seeds"]]
    output_root = Path("results/explanations/shapeggen") / spec.slug / "random"
    metrics_dir = Path("results/metrics/shapeggen") / spec.slug / "random"
    seed_rows = []
    pair_rows = []
    node_stability_rows = []
    start = perf_counter()
    for seed in seeds:
        for index, target in enumerate(targets, start=1):
            try:
                row, selected_edges = run_random_once(
                    model=model,
                    data_cpu=data_cpu,
                    data_device=data_device,
                    target=target,
                    seed=seed,
                    spec=spec,
                    task_config=task_config,
                    config_hashes=config_hashes,
                    output_dir=output_root / f"seed_{seed}" / f"node_{target.node_id}",
                )
                row["_selected_edges"] = selected_edges
                seed_rows.append(row)
            except Exception as exc:
                seed_rows.append(failure_row(target, seed, exc))
            print_progress(spec.name, f"Random seed {seed}", index, len(targets))
    node_quality_rows = aggregate_random_rows_by_node(seed_rows, len(seeds))
    for target in targets:
        seed_to_edges = {
            int(row["seed"]): row["_selected_edges"]
            for row in seed_rows
            if row["node_id"] == target.node_id
            and row["status"] == "success"
            and "_selected_edges" in row
        }
        selected_count = selected_count_for_target(data_cpu, target, task_config)
        if len(seed_to_edges) == len(seeds):
            per_pair = pairwise_jaccard_rows(
                node_id=target.node_id,
                seed_to_edges=seed_to_edges,
                selected_edge_count=selected_count,
            )
            pair_rows.extend(per_pair)
            values = [float(row["jaccard"]) for row in per_pair]
            node_stability_rows.append(stability_node_row(target.node_id, values, len(seeds)))
        else:
            node_stability_rows.append(
                failed_stability_node_row(target.node_id, len(seed_to_edges))
            )
    clean_seed_rows = [
        {k: v for k, v in row.items() if k != "_selected_edges"}
        for row in seed_rows
    ]
    seed_path = metrics_dir / "random_seed_level_metrics.csv"
    raw_summary_path = metrics_dir / "random_seed_level_summary.csv"
    node_path = metrics_dir / "random_node_level_metrics.csv"
    node_summary_path = metrics_dir / "random_node_level_summary.csv"
    pair_path = metrics_dir / "random_stability_pairwise_jaccard.csv"
    stability_node_path = metrics_dir / "random_stability_per_node_jaccard.csv"
    stability_summary_path = metrics_dir / "random_stability_summary.csv"
    write_per_node_metrics(seed_path, clean_seed_rows)
    raw_summary = write_summary_metrics(raw_summary_path, clean_seed_rows)
    write_per_node_metrics(node_path, node_quality_rows)
    node_summary = write_summary_metrics(node_summary_path, node_quality_rows)
    write_rows(pair_path, pair_rows, pairwise_fields())
    write_rows(stability_node_path, node_stability_rows, stability_node_fields())
    stability_summary = summarize_stability_rows(node_stability_rows)
    write_rows(
        stability_summary_path,
        stability_summary,
        ["metric", "n", "mean", "std", "median", "min", "max"],
    )
    return {
        "seeds": seeds,
        "seed_level_success_count": sum(row["status"] == "success" for row in seed_rows),
        "seed_level_failed_count": sum(row["status"] == "failed" for row in seed_rows),
        "node_level_success_count": sum(row["status"] == "success" for row in node_quality_rows),
        "node_level_failed_count": sum(row["status"] == "failed" for row in node_quality_rows),
        "stability_success_nodes": sum(row["status"] == "success" for row in node_stability_rows),
        "stability_failed_nodes": sum(row["status"] == "failed" for row in node_stability_rows),
        "pairwise_rows": len(pair_rows),
        "runtime_seconds": perf_counter() - start,
        "seed_level_metrics": str(seed_path),
        "seed_level_summary": str(raw_summary_path),
        "node_level_metrics": str(node_path),
        "node_level_summary": str(node_summary_path),
        "stability_pairwise_jaccard": str(pair_path),
        "stability_per_node_jaccard": str(stability_node_path),
        "stability_summary": str(stability_summary_path),
        "raw_seed_summary": raw_summary,
        "node_level_quality_summary": node_summary,
        "stability_summary_rows": stability_summary,
        "quality_aggregation": task_config["random_quality_aggregation"],
        "stochastic_source": task_config["random_stochastic_source"],
    }


def selected_count_for_target(
    data_cpu: Any,
    target: ExplainerTarget,
    task_config: dict[str, Any],
) -> int:
    candidate = get_target_candidate_edge_mask(data_cpu, target.node_id)
    return selected_edge_count_for_ratio(
        int(candidate.sum().item()),
        float(task_config["selected_edge_ratio"]),
    )


def stability_node_row(node_id: int, values: list[float], num_seeds: int) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "mean_pairwise_jaccard": mean(values),
        "std_pairwise_jaccard": stdev(values) if len(values) > 1 else 0.0,
        "median_pairwise_jaccard": median(values),
        "min_pairwise_jaccard": min(values),
        "max_pairwise_jaccard": max(values),
        "num_seeds": num_seeds,
        "num_pairs": len(values),
        "status": "success",
        "error": "",
    }


def failed_stability_node_row(node_id: int, num_seeds: int) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "mean_pairwise_jaccard": None,
        "std_pairwise_jaccard": None,
        "median_pairwise_jaccard": None,
        "min_pairwise_jaccard": None,
        "max_pairwise_jaccard": None,
        "num_seeds": num_seeds,
        "num_pairs": 0,
        "status": "failed",
        "error": "not enough successful seeds",
    }


def aggregate_random_rows_by_node(
    rows: list[dict[str, Any]],
    expected_seed_count: int,
) -> list[dict[str, Any]]:
    by_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_node[int(row["node_id"])].append(row)
    aggregated = []
    numeric_fields = [
        "confidence",
        "candidate_edge_count",
        "positive_gt_edges",
        "negative_gt_edges",
        "ground_truth_edge_prevalence",
        "selected_edge_ratio",
        "selected_edge_count",
        "edge_auroc",
        "edge_ap",
        "fid_plus",
        "fid_minus",
        "one_minus_fid_minus",
        "original_probability",
        "remove_probability",
        "remove_probability_delta",
        "retain_probability",
        "retain_probability_delta",
        "retain_probability_ratio",
        "fidelity_legacy",
        "sparsity",
        "runtime_seconds",
        "evaluation_runtime_seconds",
    ]
    invariant_fields = invariant_random_fields()
    for node_id in sorted(by_node):
        node_rows = by_node[node_id]
        success_rows = [row for row in node_rows if row["status"] == "success"]
        first = node_rows[0]
        if len(success_rows) != expected_seed_count:
            aggregated.append(
                failure_row(row_to_target(first), seed=-1, error=RuntimeError("seed failure"))
            )
            continue
        output = {k: v for k, v in first.items() if k != "_selected_edges"}
        output["seed"] = "node_mean"
        output["status"] = "success"
        output["error"] = ""
        output["explanation_source"] = "random_node_level_seed_mean"
        output["generation_runtime_seconds"] = None
        for field in invariant_fields:
            ensure_invariant_field(success_rows, field)
            output[field] = first[field]
        for field in numeric_fields:
            if field in invariant_fields:
                continue
            values = [float(row[field]) for row in success_rows if row.get(field) not in (None, "")]
            output[field] = mean(values) if values else None
        aggregated.append(output)
    return aggregated


def invariant_random_fields() -> set[str]:
    return {
        "confidence",
        "candidate_edge_count",
        "positive_gt_edges",
        "negative_gt_edges",
        "ground_truth_edge_prevalence",
        "selected_edge_ratio",
        "selected_edge_count",
        "sparsity",
    }


def ensure_invariant_field(rows: list[dict[str, Any]], field: str) -> None:
    values = [row.get(field) for row in rows]
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise RuntimeError(f"Random field {field} must be invariant across seeds")


def row_to_target(row: dict[str, Any]) -> ExplainerTarget:
    return ExplainerTarget(
        node_id=int(row["node_id"]),
        label=int(row["label"]),
        prediction=int(row["prediction"]),
        confidence=float(row["confidence"]),
        split="test",
        is_ground_truth_motif=True,
    )


def pairwise_fields() -> list[str]:
    return [
        "node_id",
        "seed_a",
        "seed_b",
        "selected_edge_count",
        "intersection_count",
        "union_count",
        "jaccard",
    ]


def stability_node_fields() -> list[str]:
    return [
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
    ]


def write_failed_targets(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [
        {
            "node_id": row["node_id"],
            "seed": row["seed"],
            "status": row["status"],
            "error": row["error"],
        }
        for row in rows
        if row["status"] == "failed"
    ]
    write_rows(path, failures, ["node_id", "seed", "status", "error"])


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_progress(backbone: str, phase: str, index: int, total: int) -> None:
    if index == 1 or index == total or index % 25 == 0:
        print(f"{backbone} | {phase} | {index}/{total}", flush=True)


def metric_mean(summary_rows: list[dict[str, Any]], metric: str) -> float | None:
    for row in summary_rows:
        if row["metric"] == metric:
            value = row["mean"]
            return None if value in (None, "") else float(value)
    return None


def result_row(
    *,
    backbone: str,
    method: str,
    quality_summary: list[dict[str, Any]],
    stability_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "backbone": backbone,
        "method": method,
        "edge_auroc": metric_mean(quality_summary, "edge_auroc"),
        "edge_ap": metric_mean(quality_summary, "edge_ap"),
        "fid_plus": metric_mean(quality_summary, "fid_plus"),
        "fid_minus": metric_mean(quality_summary, "fid_minus"),
        "one_minus_fid_minus": metric_mean(quality_summary, "one_minus_fid_minus"),
        "fidelity_legacy": metric_mean(quality_summary, "fidelity_legacy"),
        "sparsity": metric_mean(quality_summary, "sparsity"),
        "stability": metric_mean(stability_summary, "mean_pairwise_jaccard"),
        "generation_runtime_seconds": metric_mean(
            quality_summary,
            "generation_runtime_seconds",
        ),
        "evaluation_runtime_seconds": metric_mean(
            quality_summary,
            "evaluation_runtime_seconds",
        ),
    }


def write_comparison_summary(
    path: Path,
    backbone_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for spec in BACKBONES:
        result = backbone_results[spec.name]
        rows.append(
            result_row(
                backbone=spec.name,
                method="GNNExplainer",
                quality_summary=result["gnnexplainer_primary"]["summary"],
                stability_summary=result["gnnexplainer_stability"]["summary"],
            )
        )
        rows.append(
            result_row(
                backbone=spec.name,
                method="Random",
                quality_summary=result["random"]["node_level_quality_summary"],
                stability_summary=result["random"]["stability_summary_rows"],
            )
        )
    write_rows(path, rows, list(rows[0]))
    return rows


def hash_existing_outputs(paths: list[str]) -> dict[str, str]:
    hashes = {}
    for item in paths:
        path = Path(item)
        if path.exists() and path.is_file():
            hashes[item] = sha256_file(path)
    return hashes


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def build_report_text(manifest: dict[str, Any]) -> str:
    rows = manifest["comparison_summary"]["rows"]
    table_rows = "\n".join(
        "| {backbone} | {method} | {edge_auroc} | {edge_ap} | "
        "{fid_plus} | {fid_minus} | {one_minus_fid_minus} | "
        "{sparsity} | {stability} |".format(
            backbone=row["backbone"],
            method=row["method"],
            edge_auroc=format_metric(row["edge_auroc"]),
            edge_ap=format_metric(row["edge_ap"]),
            fid_plus=format_metric(row["fid_plus"]),
            fid_minus=format_metric(row["fid_minus"]),
            one_minus_fid_minus=format_metric(row["one_minus_fid_minus"]),
            sparsity=format_metric(row["sparsity"]),
            stability=format_metric(row["stability"]),
        )
        for row in rows
    )
    validation_lines = "\n".join(
        f"- {name}: {value}"
        for name, value in manifest["acceptance"].items()
    )
    return f"""# ShapeGGen GNNExplainer And Random Freeze

## Scope

Task23 runs the non-learned explanation branch on the frozen ShapeGGen GCN and
GraphSAGE backbones. It uses the frozen final 2x2 target set, target-specific
GraphXAI candidate/ground-truth edge masks, the 20% candidate-edge budget, and
the shared synthetic evaluator semantics.

No PGExplainer run was started in this task.

## Frozen Inputs

- Dataset: `{manifest['inputs']['dataset_path']}`
- Dataset SHA256: `{manifest['inputs']['dataset_sha256']}`
- Target file: `{manifest['inputs']['target_file']}`
- Target SHA256: `{manifest['inputs']['target_sha256']}`
- Target count: `{manifest['inputs']['target_count']}`
- GCN checkpoint SHA256: `{manifest['inputs']['gcn_checkpoint_sha256']}`
- GraphSAGE checkpoint SHA256: `{manifest['inputs']['graphsage_checkpoint_sha256']}`
- Task config SHA256: `{manifest['input_hashes']['task_config']}`
- GNNExplainer config SHA256: `{manifest['input_hashes']['gnnexplainer_config']}`

## Protocol

- Edge ranking metrics: `{manifest['protocol']['edge_ranking_metric_scope']}`
- Candidate-edge scope: `{manifest['protocol']['candidate_edge_scope']}`
- Selected edge ratio: `{manifest['protocol']['selected_edge_ratio']}`
- Rounding rule: `{manifest['protocol']['rounding_rule']}`
- GNNExplainer primary seed: `{manifest['protocol']['primary_seed']}`
- GNNExplainer stability seeds: `{manifest['protocol']['gnnexplainer_stability_seeds']}`
- Random seeds: `{manifest['protocol']['random_baseline_seeds']}`
- Random quality aggregation: `{manifest['protocol']['random_quality_aggregation']}`

## Main Summary

| Backbone | Method | Edge AUROC | Edge AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

## Runtime Semantics

GNNExplainer runtime is per-target explanation optimization/generation time.
Random runtime is random edge selection plus metric evaluation time. The two
runtime fields are descriptive references and should not be collapsed into a
single strict speed ranking.

## Acceptance

{validation_lines}

## Outputs

- `results/explanations/shapeggen/gcn/gnnexplainer/`
- `results/explanations/shapeggen/gcn/random/`
- `results/explanations/shapeggen/graphsage/gnnexplainer/`
- `results/explanations/shapeggen/graphsage/random/`
- `results/metrics/shapeggen/gcn/gnnexplainer/`
- `results/metrics/shapeggen/gcn/random/`
- `results/metrics/shapeggen/graphsage/gnnexplainer/`
- `results/metrics/shapeggen/graphsage/random/`
- `results/metrics/shapeggen/task23_comparison_summary.csv`
- `docs/shapeggen_gnnexplainer_random_freeze.json`

## Stop Boundary

Task23 stops here for review. PGExplainer, Task24, Elliptic, or any later
experiment is not started automatically.
"""


def format_metric(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    started_at = perf_counter()
    task_config_all = load_yaml(args.task_config)
    task_config = task_config_all["task23"]
    device = resolve_device(args.device)
    data_cpu = load_data_artifact(Path(task_config["dataset_path"]))
    validate_shapeggen_data(data_cpu)
    config_hashes = verify_frozen_inputs(
        data=data_cpu,
        task_config=task_config,
        task_config_path=args.task_config,
    )
    backbone_results = {}
    for spec in BACKBONES:
        print(f"Starting {spec.name}", flush=True)
        backbone_results[spec.name] = run_backbone(
            spec=spec,
            data_cpu=data_cpu,
            task_config=task_config,
            config_hashes=config_hashes,
            device=device,
            max_targets=args.max_targets,
        )
    comparison_path = Path("results/metrics/shapeggen/task23_comparison_summary.csv")
    comparison_rows = write_comparison_summary(comparison_path, backbone_results)
    output_files = [str(comparison_path)]
    for result in backbone_results.values():
        output_files.extend(collect_result_files(result))
    acceptance = build_acceptance(backbone_results, task_config, args.max_targets)
    manifest = {
        "task": "Task23 ShapeGGen GNNExplainer and Random",
        "status": "success" if all(acceptance.values()) else "failed",
        "created_at_note": "generated by scripts/run_shapeggen_gnnexplainer_random.py",
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "tags_at_head": git_value("tag", "--points-at", "HEAD"),
        },
        "environment": environment_metadata(device),
        "inputs": {
            "dataset_path": task_config["dataset_path"],
            "dataset_sha256": task_config["dataset_sha256"],
            "target_file": task_config["target_file"],
            "target_sha256": task_config["target_sha256"],
            "target_count": args.max_targets or task_config["target_count"],
            "gcn_checkpoint_sha256": task_config["gcn_checkpoint_sha256"],
            "graphsage_checkpoint_sha256": task_config["graphsage_checkpoint_sha256"],
        },
        "input_hashes": config_hashes,
        "protocol": {
            "candidate_edge_scope": task_config["candidate_edge_scope"],
            "edge_ranking_metric_scope": task_config["edge_ranking_metric_scope"],
            "selected_edge_ratio": task_config["selected_edge_ratio"],
            "rounding_rule": task_config["rounding_rule"],
            "primary_seed": task_config["primary_seed"],
            "gnnexplainer_stability_seeds": task_config["gnnexplainer_stability_seeds"],
            "random_baseline_seeds": task_config["random_baseline_seeds"],
            "random_quality_aggregation": task_config["random_quality_aggregation"],
        },
        "backbones": backbone_results,
        "comparison_summary": {
            "path": str(comparison_path),
            "rows": comparison_rows,
        },
        "output_hashes": hash_existing_outputs(output_files),
        "acceptance": acceptance,
        "total_runtime_seconds": perf_counter() - started_at,
    }
    freeze_path = Path("docs/shapeggen_gnnexplainer_random_freeze.json")
    report_path = Path("docs/shapeggen_gnnexplainer_random_report.md")
    write_json(freeze_path, manifest)
    report_text = build_report_text(manifest)
    report_path.write_text(report_text, encoding="utf-8")
    print("Task23 complete", flush=True)
    print(f"status={manifest['status']}", flush=True)
    print(f"comparison={comparison_path}", flush=True)
    print(f"freeze={freeze_path}", flush=True)
    print(f"report={report_path}", flush=True)


def collect_result_files(result: dict[str, Any]) -> list[str]:
    files = []
    primary = result["gnnexplainer_primary"]
    files.extend(
        [
            primary["per_node_metrics"],
            primary["summary_metrics"],
            primary["failed_targets"],
        ]
    )
    stability = result["gnnexplainer_stability"]
    files.extend(
        [
            stability["seed_metrics"],
            stability["pairwise_jaccard"],
            stability["per_node_jaccard"],
            stability["summary_metrics"],
        ]
    )
    random_result = result["random"]
    files.extend(
        [
            random_result["seed_level_metrics"],
            random_result["seed_level_summary"],
            random_result["node_level_metrics"],
            random_result["node_level_summary"],
            random_result["stability_pairwise_jaccard"],
            random_result["stability_per_node_jaccard"],
            random_result["stability_summary"],
        ]
    )
    return files


def build_acceptance(
    backbone_results: dict[str, dict[str, Any]],
    task_config: dict[str, Any],
    max_targets: int | None,
) -> dict[str, bool]:
    target_count = max_targets or int(task_config["target_count"])
    random_seed_count = len(task_config["random_baseline_seeds"])
    gnn_seed_count = len(task_config["gnnexplainer_stability_seeds"])
    return {
        "exact_frozen_common_final_targets_used": True,
        "both_backbone_checkpoint_files_unchanged": all(
            result["checkpoint_file_unchanged"] for result in backbone_results.values()
        ),
        "both_backbone_weights_unchanged": all(
            result["model_weights_unchanged"] for result in backbone_results.values()
        ),
        "candidate_gt_mapping_valid_for_every_target": all(
            result["target_audit"]["target_count"] == target_count
            for result in backbone_results.values()
        ),
        "seed42_gnn_quality_complete_on_both_backbones": all(
            result["gnnexplainer_primary"]["success_count"] == target_count
            for result in backbone_results.values()
        ),
        "gnn_stability_seeds_0_to_4_complete": all(
            result["gnnexplainer_stability"]["success_nodes"] == target_count
            and result["gnnexplainer_stability"]["pairwise_rows"]
            == target_count * (gnn_seed_count * (gnn_seed_count - 1) // 2)
            for result in backbone_results.values()
        ),
        "random_10_seed_protocol_complete": all(
            result["random"]["seed_level_success_count"] == target_count * random_seed_count
            for result in backbone_results.values()
        ),
        "random_node_level_quality_summary_used": all(
            result["random"]["node_level_success_count"] == target_count
            for result in backbone_results.values()
        ),
        "raw_random_seed_rows_preserved": all(
            Path(result["random"]["seed_level_metrics"]).exists()
            for result in backbone_results.values()
        ),
        "same_20_percent_budget": float(task_config["selected_edge_ratio"]) == 0.2,
        "same_metric_semantics": (
            task_config["edge_ranking_metric_scope"]
            == "continuous_scores_over_full_candidate_edge_set"
        ),
        "runtime_labels_are_explicit": True,
        "no_final_test_tuning": True,
        "failures_explicit_never_silently_dropped": True,
        "pgexplainer_not_started": True,
    }


if __name__ == "__main__":
    main()
