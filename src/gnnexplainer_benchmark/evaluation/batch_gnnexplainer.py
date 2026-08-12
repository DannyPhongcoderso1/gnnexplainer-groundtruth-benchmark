from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch
from torch_geometric.data import Data

from gnnexplainer_benchmark.data.target_selection import ExplainerTarget
from gnnexplainer_benchmark.evaluation.budget import select_top_ratio_edges
from gnnexplainer_benchmark.evaluation.calibration import assert_gcnconv_cached_false
from gnnexplainer_benchmark.evaluation.edge_ground_truth import (
    candidate_edge_scores_and_labels,
    compute_edge_ground_truth_metrics,
)
from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity
from gnnexplainer_benchmark.evaluation.runtime import measure_runtime
from gnnexplainer_benchmark.explain.gnnexplainer_runner import run_gnnexplainer_for_node
from gnnexplainer_benchmark.explain.mask_export import export_single_node_explanation
from gnnexplainer_benchmark.explain.node_selection import TargetNode
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask

PER_NODE_FIELDS = [
    "node_id",
    "label",
    "prediction",
    "confidence",
    "candidate_edge_count",
    "positive_gt_edges",
    "negative_gt_edges",
    "ground_truth_edge_prevalence",
    "selected_edge_ratio",
    "selected_edge_count",
    "edge_auroc",
    "edge_ap",
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
    "fidelity_legacy",
    "sparsity",
    "runtime_seconds",
    "generation_runtime_seconds",
    "evaluation_runtime_seconds",
    "explanation_source",
    "seed",
    "status",
    "error",
]

SUMMARY_METRICS = [
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "remove_probability_delta",
    "retain_probability",
    "retain_probability_delta",
    "fidelity_legacy",
    "sparsity",
    "runtime_seconds",
    "generation_runtime_seconds",
    "evaluation_runtime_seconds",
]


@dataclass(frozen=True)
class BatchConfig:
    checkpoint_path: Path
    target_file: Path
    explainer_config: dict[str, Any]
    metrics_config: dict[str, Any]
    seed: int
    selected_edge_ratio: float
    output_root: Path
    metrics_output_dir: Path
    device: str
    dataset_checksum: str
    num_hops: int
    raw_reuse_root: Path | None = None


def read_explainer_targets(path: Path) -> list[ExplainerTarget]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    targets = [
        ExplainerTarget(
            node_id=int(row["node_id"]),
            label=int(row["label"]),
            prediction=int(row["prediction"]),
            confidence=float(row["confidence"]),
            split=row["split"],
            is_ground_truth_motif=str(row["is_ground_truth_motif"]).lower() == "true",
        )
        for row in rows
    ]
    validate_target_order_and_uniqueness(targets)
    return targets


def validate_target_order_and_uniqueness(targets: list[ExplainerTarget]) -> None:
    node_ids = [target.node_id for target in targets]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate target node ids are not allowed")
    if node_ids != sorted(node_ids):
        raise ValueError("Target rows must be sorted by node_id for deterministic batch order")


def batch_signature(config: BatchConfig, target: ExplainerTarget) -> str:
    payload = {
        "checkpoint_path": str(config.checkpoint_path),
        "dataset_checksum": config.dataset_checksum,
        "explainer_config": config.explainer_config,
        "metrics_config": config.metrics_config,
        "node_id": target.node_id,
        "seed": config.seed,
        "selected_edge_ratio": config.selected_edge_ratio,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def can_resume_node(output_dir: Path, signature: str) -> bool:
    metadata_path = output_dir / "metadata.json"
    required = [metadata_path, output_dir / "edge_mask.pt", output_dir / "node_mask.pt"]
    if not all(path.exists() for path in required):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    current_signature = metadata.get("batch_run_signature")
    if current_signature is None:
        current_signature = legacy_signature_from_metadata(metadata)
    return current_signature == signature


def find_reusable_raw_explanation(
    *,
    target: ExplainerTarget,
    config: BatchConfig,
) -> Path | None:
    if config.raw_reuse_root is None:
        return None
    candidate_dir = config.raw_reuse_root / f"node_{target.node_id}"
    metadata_path = candidate_dir / "metadata.json"
    edge_mask_path = candidate_dir / "edge_mask.pt"
    node_mask_path = candidate_dir / "node_mask.pt"
    if not (metadata_path.exists() and edge_mask_path.exists() and node_mask_path.exists()):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not raw_explanation_matches(metadata, target, config):
        return None
    return candidate_dir


def raw_explanation_matches(
    metadata: dict[str, Any],
    target: ExplainerTarget,
    config: BatchConfig,
) -> bool:
    metadata_target = metadata.get("target", {})
    metadata_checkpoint = metadata.get("checkpoint", {})
    return (
        metadata.get("raw_masks_saved") is True
        and metadata.get("dataset_checksum") == config.dataset_checksum
        and metadata.get("explainer_config") == config.explainer_config
        and metadata.get("num_hops") == config.num_hops
        and metadata_checkpoint.get("path") == str(config.checkpoint_path)
        and metadata_target.get("node_id") == target.node_id
        and metadata_target.get("label") == target.label
        and metadata_target.get("prediction") == target.prediction
    )


def materialize_reused_raw_explanation(
    *,
    source_dir: Path,
    output_dir: Path,
    signature: str,
    target: ExplainerTarget,
    config: BatchConfig,
    candidate_edge_mask: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "edge_mask.pt", output_dir / "edge_mask.pt")
    shutil.copy2(source_dir / "node_mask.pt", output_dir / "node_mask.pt")
    source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "dataset_checksum": config.dataset_checksum,
        "checkpoint": {"path": str(config.checkpoint_path)},
        "explainer_config": config.explainer_config,
        "metrics_config": config.metrics_config,
        "num_hops": config.num_hops,
        "subgraph_edge_count": int(candidate_edge_mask.sum().item()),
        "budget_protocol": "fixed_candidate_edge_ratio",
        "selected_edge_ratio": config.selected_edge_ratio,
        "rounding_rule": "ceil(candidate_edge_count * selected_edge_ratio), minimum 1",
        "runtime_seconds": 0.0,
        "generation_runtime_seconds": source_metadata.get("runtime_seconds"),
        "evaluation_runtime_seconds": None,
        "raw_explanation_reused": True,
        "reused_from": str(source_dir),
        "source_runtime_seconds": source_metadata.get("runtime_seconds"),
        "batch_run_signature": signature,
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


def legacy_signature_from_metadata(metadata: dict[str, Any]) -> str | None:
    if not metadata.get("raw_masks_saved"):
        return None
    if "explainer_config" not in metadata or "checkpoint" not in metadata:
        return None
    if "target" not in metadata:
        return None
    return None


def target_node_from_explainer_target(target: ExplainerTarget) -> TargetNode:
    return TargetNode(
        node_id=target.node_id,
        label=target.label,
        prediction=target.prediction,
        confidence=target.confidence,
    )


def run_batch_target(
    *,
    model: torch.nn.Module,
    data_cpu: Data,
    data_device: Data,
    target: ExplainerTarget,
    config: BatchConfig,
) -> dict[str, Any]:
    assert_gcnconv_cached_false(model)
    output_dir = config.output_root / f"node_{target.node_id}"
    signature = batch_signature(config, target)
    runtime_seconds: float | None = None
    generation_runtime_seconds: float | None = None
    evaluation_runtime_seconds: float | None = None
    explanation_source = "generated_raw_mask"
    candidate_edge_mask = computation_subgraph_edge_mask(data_cpu, target.node_id, config.num_hops)
    if not can_resume_node(output_dir, signature):
        reusable_raw_dir = find_reusable_raw_explanation(target=target, config=config)
        if reusable_raw_dir is not None:
            materialize_reused_raw_explanation(
                source_dir=reusable_raw_dir,
                output_dir=output_dir,
                signature=signature,
                target=target,
                config=config,
                candidate_edge_mask=candidate_edge_mask,
            )
            reused_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            generation_runtime_seconds = reused_metadata.get("generation_runtime_seconds")
            explanation_source = "reused_existing_raw_mask"
        else:
            target_node = target_node_from_explainer_target(target)
            with measure_runtime() as timer:
                explanation = run_gnnexplainer_for_node(
                    model=model,
                    data=data_device,
                    target=target_node,
                    config=config.explainer_config,
                )
            if timer.runtime_seconds is None:
                raise RuntimeError("Runtime measurement failed")
            runtime_seconds = timer.runtime_seconds
            generation_runtime_seconds = runtime_seconds
            evaluation_runtime_seconds = None
            selected_count = selected_edge_count_from_explanation(
                explanation.edge_mask.detach().cpu(),
                candidate_edge_mask,
                config.selected_edge_ratio,
            )
            export_single_node_explanation(
                output_dir=output_dir,
                explanation=explanation,
                data=data_cpu,
                target=target_node,
                metadata={
                    "dataset_checksum": config.dataset_checksum,
                    "checkpoint": {
                        "path": str(config.checkpoint_path),
                    },
                    "explainer_config": config.explainer_config,
                    "metrics_config": config.metrics_config,
                    "num_hops": config.num_hops,
                    "subgraph_edge_count": int(candidate_edge_mask.sum().item()),
                    "budget_protocol": "fixed_candidate_edge_ratio",
                    "selected_edge_ratio": config.selected_edge_ratio,
                    "rounding_rule": "ceil(candidate_edge_count * selected_edge_ratio), minimum 1",
                    "runtime_seconds": runtime_seconds,
                    "raw_explanation_reused": False,
                    "batch_run_signature": signature,
                },
                top_k_edges=selected_count,
                subgraph_edge_mask=candidate_edge_mask,
            )
    else:
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        generation_runtime_seconds = metadata.get("generation_runtime_seconds")
        if generation_runtime_seconds is None:
            generation_runtime_seconds = metadata.get("source_runtime_seconds")
        if generation_runtime_seconds is None:
            generation_runtime_seconds = metadata.get("runtime_seconds")
        evaluation_runtime_seconds = metadata.get("evaluation_runtime_seconds")
        explanation_source = (
            "reused_existing_raw_mask"
            if metadata.get("raw_explanation_reused")
            else "generated_raw_mask"
        )
    runtime_seconds = generation_runtime_seconds

    edge_scores = torch.load(output_dir / "edge_mask.pt", map_location="cpu", weights_only=False)
    metrics = compute_ratio_metrics(
        model=model,
        data=data_device,
        node_id=target.node_id,
        target=target,
        edge_scores=edge_scores,
        candidate_edge_mask=candidate_edge_mask,
        selected_edge_ratio=config.selected_edge_ratio,
        runtime_seconds=runtime_seconds,
        explainer=config.explainer_config["explainer"]["algorithm"],
        candidate_edge_scope=config.metrics_config["evaluation"]["candidate_edge_scope"],
    )
    metrics["edge_ground_truth"]["candidate_edge_scope_detail"] = (
        f"{config.num_hops}-hop computation subgraph around target node"
    )
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
    return flatten_metrics_row(
        metrics,
        target,
        config.seed,
        status="success",
        generation_runtime_seconds=generation_runtime_seconds,
        evaluation_runtime_seconds=evaluation_runtime_seconds,
        explanation_source=explanation_source,
    )


def selected_edge_count_from_explanation(
    edge_scores: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
    ratio: float,
) -> int:
    _, budget = select_top_ratio_edges(edge_scores, candidate_edge_mask, ratio)
    return budget.selected_edge_count


def compute_ratio_metrics(
    *,
    model: torch.nn.Module,
    data: Data,
    node_id: int,
    target: ExplainerTarget,
    edge_scores: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
    selected_edge_ratio: float,
    runtime_seconds: float,
    explainer: str,
    candidate_edge_scope: str,
) -> dict[str, Any]:
    candidate_scores, candidate_ground_truth = candidate_edge_scores_and_labels(
        edge_scores=edge_scores.detach().cpu(),
        ground_truth_edge_mask=data.edge_mask.detach().cpu(),
        candidate_edge_mask=candidate_edge_mask.detach().cpu(),
    )
    edge_metrics = compute_edge_ground_truth_metrics(candidate_scores, candidate_ground_truth)
    selected_edge_mask, budget = select_top_ratio_edges(
        edge_scores=edge_scores.detach().cpu(),
        candidate_edge_mask=candidate_edge_mask.detach().cpu(),
        ratio=selected_edge_ratio,
    )
    fidelity = compute_fidelity(
        model=model,
        data=data,
        node_id=node_id,
        selected_edge_mask=selected_edge_mask.to(data.edge_index.device),
    )
    positive_edge_count = int(candidate_ground_truth.sum().item())
    negative_edge_count = int(candidate_ground_truth.numel() - positive_edge_count)
    return {
        "schema_version": "core_metrics_v1",
        "explainer": explainer,
        "node_id": node_id,
        "target": {
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "confidence": target.confidence,
        },
        "edge_ground_truth": {
            "candidate_edge_scope": candidate_edge_scope,
            "candidate_edge_count": budget.candidate_edge_count,
            "candidate_positive_edge_count": positive_edge_count,
            "candidate_negative_edge_count": negative_edge_count,
            "ground_truth_edge_prevalence": positive_edge_count / budget.candidate_edge_count,
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
        "_selected_edge_mask": selected_edge_mask,
    }


def write_selected_edges_csv(
    path: Path,
    *,
    edge_scores: torch.Tensor,
    data: Data,
    selected_edge_mask: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_indices = torch.nonzero(selected_edge_mask.detach().cpu(), as_tuple=False).view(-1)
    selected_scores = edge_scores.detach().cpu()[selected_indices]
    order = torch.argsort(selected_scores, descending=True)
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
        for rank, order_idx in enumerate(order.tolist(), start=1):
            edge_idx = int(selected_indices[order_idx].item())
            writer.writerow(
                {
                    "rank": rank,
                    "global_edge_index": edge_idx,
                    "src": int(data.edge_index[0, edge_idx].item()),
                    "dst": int(data.edge_index[1, edge_idx].item()),
                    "score": float(edge_scores[edge_idx].item()),
                    "is_ground_truth": bool(data.edge_mask[edge_idx].item()),
                    "in_candidate_edge_set": bool(candidate_edge_mask[edge_idx].item()),
                }
            )


def flatten_metrics_row(
    metrics: dict[str, Any],
    target: ExplainerTarget,
    seed: int,
    status: str,
    error: str = "",
    generation_runtime_seconds: float | None = None,
    evaluation_runtime_seconds: float | None = None,
    explanation_source: str = "",
) -> dict[str, Any]:
    return {
        "node_id": target.node_id,
        "label": target.label,
        "prediction": target.prediction,
        "confidence": target.confidence,
        "candidate_edge_count": metrics["edge_ground_truth"]["candidate_edge_count"],
        "positive_gt_edges": metrics["edge_ground_truth"]["candidate_positive_edge_count"],
        "negative_gt_edges": metrics["edge_ground_truth"]["candidate_negative_edge_count"],
        "ground_truth_edge_prevalence": metrics["edge_ground_truth"][
            "ground_truth_edge_prevalence"
        ],
        "selected_edge_ratio": metrics["sparsity"]["selected_edge_ratio"],
        "selected_edge_count": metrics["sparsity"]["selected_edge_count"],
        "edge_auroc": metrics["edge_ground_truth"]["edge_auroc"],
        "edge_ap": metrics["edge_ground_truth"]["edge_ap"],
        "original_prediction": metrics["fidelity"]["original"]["prediction"],
        "remove_prediction": metrics["fidelity"]["remove_result"]["prediction"],
        "retain_prediction": metrics["fidelity"]["retain_result"]["prediction"],
        "fid_plus": metrics["fidelity"]["fid_plus"],
        "fid_minus": metrics["fidelity"]["fid_minus"],
        "one_minus_fid_minus": metrics["fidelity"]["one_minus_fid_minus"],
        "original_probability": metrics["fidelity"]["original"]["probability"],
        "remove_probability": metrics["fidelity"]["remove_result"]["probability"],
        "remove_probability_delta": metrics["fidelity"]["remove_result"][
            "remove_probability_delta"
        ],
        "retain_probability": metrics["fidelity"]["retain_result"]["probability"],
        "retain_probability_delta": metrics["fidelity"]["retain_result"][
            "retain_probability_delta"
        ],
        "retain_probability_ratio": metrics["fidelity"]["retain_result"]["probability_ratio"],
        "fidelity_legacy": metrics["fidelity"]["fidelity_legacy"],
        "sparsity": metrics["sparsity"]["sparsity"],
        "runtime_seconds": metrics["runtime"]["runtime_seconds"],
        "generation_runtime_seconds": generation_runtime_seconds,
        "evaluation_runtime_seconds": evaluation_runtime_seconds,
        "explanation_source": explanation_source,
        "seed": seed,
        "status": status,
        "error": error,
    }


def failure_row(target: ExplainerTarget, seed: int, error: Exception) -> dict[str, Any]:
    return {
        "node_id": target.node_id,
        "label": target.label,
        "prediction": target.prediction,
        "confidence": target.confidence,
        "candidate_edge_count": None,
        "positive_gt_edges": None,
        "negative_gt_edges": None,
        "ground_truth_edge_prevalence": None,
        "selected_edge_ratio": None,
        "selected_edge_count": None,
        "edge_auroc": None,
        "edge_ap": None,
        "original_prediction": None,
        "remove_prediction": None,
        "retain_prediction": None,
        "fid_plus": None,
        "fid_minus": None,
        "one_minus_fid_minus": None,
        "original_probability": None,
        "remove_probability": None,
        "remove_probability_delta": None,
        "retain_probability": None,
        "retain_probability_delta": None,
        "retain_probability_ratio": None,
        "fidelity_legacy": None,
        "sparsity": None,
        "runtime_seconds": None,
        "generation_runtime_seconds": None,
        "evaluation_runtime_seconds": None,
        "explanation_source": None,
        "seed": seed,
        "status": "failed",
        "error": str(error),
    }


def write_per_node_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    validate_probability_diagnostics(rows)
    validate_runtime_reporting(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_NODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_metrics(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = summarize_metric_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "n", "mean", "std", "median", "min", "max"],
        )
        writer.writeheader()
        writer.writerows(summary)
    return summary


def summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_probability_diagnostics(rows)
    validate_runtime_reporting(rows)
    success_rows = [row for row in rows if row["status"] == "success"]
    summaries = []
    for metric in SUMMARY_METRICS:
        values = [
            float(row[metric])
            for row in success_rows
            if row.get(metric) not in (None, "")
        ]
        if values:
            summaries.append(
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
            summaries.append(
                {
                    "metric": metric,
                    "n": 0,
                    "mean": None,
                    "std": None,
                    "median": None,
                    "min": None,
                    "max": None,
                }
            )
    return summaries


def validate_fidelity_alias(rows: list[dict[str, Any]], tolerance: float = 1e-12) -> None:
    for row in rows:
        fidelity = row.get("fidelity")
        remove_delta = row.get("remove_delta")
        if fidelity in (None, "") and remove_delta in (None, ""):
            continue
        if fidelity in (None, "") or remove_delta in (None, ""):
            raise ValueError("fidelity and remove_delta must both be present")
        if abs(float(fidelity) - float(remove_delta)) >= tolerance:
            raise ValueError("fidelity must equal remove_delta within tolerance")


def validate_probability_diagnostics(
    rows: list[dict[str, Any]],
    tolerance: float = 1e-12,
) -> None:
    for row in rows:
        legacy = row.get("fidelity_legacy")
        remove_delta = row.get("remove_probability_delta")
        if legacy in (None, "") and remove_delta in (None, ""):
            continue
        if legacy in (None, "") or remove_delta in (None, ""):
            raise ValueError("fidelity_legacy and remove_probability_delta must both be present")
        if abs(float(legacy) - float(remove_delta)) >= tolerance:
            raise ValueError("fidelity_legacy must equal remove_probability_delta within tolerance")


def validate_runtime_reporting(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        source = row.get("explanation_source")
        generation_runtime = row.get("generation_runtime_seconds")
        runtime = row.get("runtime_seconds")
        if source == "reused_existing_raw_mask" and generation_runtime in (None, ""):
            continue
        if source == "reused_existing_raw_mask" and float(generation_runtime) == 0.0:
            raise ValueError("reused explanations must not report 0.0 generation runtime")
        if source == "reused_existing_raw_mask" and runtime not in (None, ""):
            if abs(float(runtime) - float(generation_runtime)) >= 1e-12:
                raise ValueError("runtime_seconds must mirror recovered generation runtime")
