from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

import torch

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import (
    compute_ratio_metrics,
    read_explainer_targets,
    write_selected_edges_csv,
)
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.explainers.pgexplainer_smoke import model_parameter_hash
from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    build_pgexplainer_from_config,
    load_yaml,
    materialize_pgexplainer,
    wrap_pgexplainer,
)
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)

PGEXPLAINER_PER_NODE_FIELDS = [
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
    "inference_runtime_seconds",
    "explainer_checkpoint",
    "explainer_training_seed",
    "seed",
    "status",
    "error",
]

PGEXPLAINER_SUMMARY_METRICS = [
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "remove_probability_delta",
    "retain_probability",
    "retain_probability_delta",
    "retain_probability_ratio",
    "fidelity_legacy",
    "sparsity",
    "inference_runtime_seconds",
]


@dataclass(frozen=True)
class PGExplainerEvaluationResult:
    per_node_metrics_path: Path
    summary_metrics_path: Path
    manifest_path: Path
    success_count: int
    failure_count: int
    model_weights_unchanged: bool


def evaluate_pgexplainer_gcn(
    *,
    artifact_path: Path = Path("artifacts/data/ba_house.pt"),
    checkpoint_path: Path = Path("artifacts/checkpoints/ba_house_gcn_best.pt"),
    explainer_checkpoint_path: Path = Path(
        "artifacts/explainers/pgexplainer_gcn_ba_house_seed42.pt"
    ),
    explainer_manifest_path: Path = Path(
        "artifacts/explainers/pgexplainer_gcn_ba_house_seed42_manifest.json"
    ),
    explainer_config_path: Path = Path("configs/explainer/pgexplainer.yaml"),
    evaluation_config_path: Path = Path("configs/evaluation/pgexplainer_gcn.yaml"),
    target_file: Path = Path("results/metrics/explainer_test_targets.csv"),
    phase4_reference_per_node_path: Path = Path(
        "results/metrics/frozen_phase4c_v2/per_node_metrics.csv"
    ),
    output_root: Path = Path("results/explanations/pgexplainer_gcn_seed42"),
    metrics_output_dir: Path = Path("results/metrics/pgexplainer_gcn_seed42"),
    device: str = "cpu",
) -> PGExplainerEvaluationResult:
    data_cpu = load_data_artifact(artifact_path)
    validate_ba_house_data(data_cpu)
    checkpoint = load_checkpoint(checkpoint_path)
    explainer_config = load_yaml(explainer_config_path)
    evaluation_config = load_yaml(evaluation_config_path)
    explainer_manifest = json.loads(explainer_manifest_path.read_text(encoding="utf-8"))
    explainer_payload = torch.load(
        explainer_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    targets = read_explainer_targets(target_file)
    phase4_rows = read_phase4_reference_rows(phase4_reference_per_node_path)
    validate_target_order(targets, phase4_rows)

    torch_device = torch.device(device)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_hash_before = model_parameter_hash(model)
    data = data_cpu.clone().to(torch_device)
    with torch.no_grad():
        frozen_predictions = model(data.x, data.edge_index).argmax(dim=-1).detach()

    algorithm = build_pgexplainer_from_config(explainer_config).to(torch_device)
    explainer = wrap_pgexplainer(model, algorithm)
    materialize_pgexplainer(algorithm, model, data.x, data.edge_index, targets[0].node_id)
    algorithm.load_state_dict(explainer_payload["algorithm_state_dict"])
    algorithm._curr_epoch = int(
        explainer_payload.get("algorithm_current_epoch", algorithm.epochs - 1)
    )

    rows = []
    selected_edge_ratio = float(evaluation_config["evaluation"]["selected_edge_ratio"])
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])
    for target in targets:
        reference = phase4_rows[target.node_id]
        output_dir = output_root / f"node_{target.node_id}"
        try:
            candidate_edge_mask = computation_subgraph_edge_mask(
                data_cpu,
                target.node_id,
                num_hops,
            )
            verify_candidate_parity(data_cpu, candidate_edge_mask, reference)
            started_at = perf_counter()
            explanation = explainer(
                data.x,
                data.edge_index,
                target=frozen_predictions,
                index=target.node_id,
            )
            inference_runtime_seconds = perf_counter() - started_at
            edge_scores = explanation.edge_mask.detach().cpu()
            if edge_scores.numel() != data_cpu.edge_index.size(1):
                raise RuntimeError("PGExplainer edge mask is not full-edge aligned")
            if not torch.isfinite(edge_scores).all():
                raise RuntimeError("PGExplainer edge mask contains NaN or Inf")
            metrics = compute_ratio_metrics(
                model=model,
                data=data,
                node_id=target.node_id,
                target=target,
                edge_scores=edge_scores,
                candidate_edge_mask=candidate_edge_mask,
                selected_edge_ratio=selected_edge_ratio,
                runtime_seconds=inference_runtime_seconds,
                explainer="PGExplainer",
                candidate_edge_scope=evaluation_config["evaluation"]["candidate_edge_scope"],
            )
            selected_edge_mask = metrics.pop("_selected_edge_mask")
            row = flatten_pgexplainer_metrics(
                metrics=metrics,
                target=target,
                explainer_checkpoint_path=explainer_checkpoint_path,
                explainer_training_seed=int(explainer_manifest["seed"]),
                inference_runtime_seconds=inference_runtime_seconds,
                seed=int(explainer_manifest["seed"]),
                status="success",
            )
            verify_phase4_metric_parity(row, reference)
            write_raw_pgexplainer_explanation(
                output_dir=output_dir,
                data=data_cpu,
                edge_scores=edge_scores,
                selected_edge_mask=selected_edge_mask,
                candidate_edge_mask=candidate_edge_mask,
                metadata={
                    "explainer": "PGExplainer",
                    "explainer_checkpoint": str(explainer_checkpoint_path),
                    "explainer_checkpoint_sha256": sha256_file(explainer_checkpoint_path),
                    "explainer_training_seed": int(explainer_manifest["seed"]),
                    "target": {
                        "node_id": target.node_id,
                        "label": target.label,
                        "prediction": target.prediction,
                        "confidence": target.confidence,
                    },
                    "candidate_edge_count": row["candidate_edge_count"],
                    "selected_edge_ratio": selected_edge_ratio,
                    "selected_edge_count": row["selected_edge_count"],
                    "inference_runtime_seconds": inference_runtime_seconds,
                    "edge_ranking_metric_scope": evaluation_config["evaluation"][
                        "edge_ranking_metric_scope"
                    ],
                    "budget_protocol": evaluation_config["evaluation"]["budget_protocol"],
                    "rounding_rule": evaluation_config["evaluation"]["rounding_rule"],
                    "raw_masks_saved": True,
                },
            )
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
            rows.append(row)
        except Exception as exc:
            rows.append(failure_pgexplainer_row(target, explainer_checkpoint_path, exc))

    model_hash_after = model_parameter_hash(model)
    model_weights_unchanged = model_hash_before == model_hash_after
    if not model_weights_unchanged:
        raise RuntimeError("Frozen model weights changed during PGExplainer evaluation")
    per_node_path = metrics_output_dir / "per_node_metrics.csv"
    summary_path = metrics_output_dir / "summary_metrics.csv"
    manifest_path = metrics_output_dir / "manifest.json"
    write_pgexplainer_per_node_metrics(per_node_path, rows)
    summary = write_pgexplainer_summary_metrics(summary_path, rows)
    manifest = {
        "explainer": "PGExplainer",
        "explainer_checkpoint": str(explainer_checkpoint_path),
        "explainer_checkpoint_sha256": sha256_file(explainer_checkpoint_path),
        "explainer_manifest": str(explainer_manifest_path),
        "explainer_training_seed": int(explainer_manifest["seed"]),
        "target_file": str(target_file),
        "target_count": len(targets),
        "success_count": sum(row["status"] == "success" for row in rows),
        "failure_count": sum(row["status"] != "success" for row in rows),
        "phase4_reference_per_node": str(phase4_reference_per_node_path),
        "candidate_counts_match_phase4": True,
        "ground_truth_counts_match_phase4": True,
        "selected_counts_match_phase4": True,
        "original_predictions_match_phase4": True,
        "dataset_sha256": sha256_file(artifact_path),
        "model_type": str(checkpoint["resolved_config"]["model"].get("type", "GCN")),
        "model_checkpoint_sha256": sha256_file(checkpoint_path),
        "gcn_checkpoint_sha256": sha256_file(checkpoint_path),
        "model_hash_before": model_hash_before,
        "model_hash_after": model_hash_after,
        "model_weights_unchanged": model_weights_unchanged,
        "selected_edge_ratio": selected_edge_ratio,
        "rounding_rule": evaluation_config["evaluation"]["rounding_rule"],
        "edge_ranking_metric_scope": evaluation_config["evaluation"][
            "edge_ranking_metric_scope"
        ],
        "per_node_metrics": str(per_node_path),
        "summary_metrics": str(summary_path),
        "summary": summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PGExplainerEvaluationResult(
        per_node_metrics_path=per_node_path,
        summary_metrics_path=summary_path,
        manifest_path=manifest_path,
        success_count=manifest["success_count"],
        failure_count=manifest["failure_count"],
        model_weights_unchanged=model_weights_unchanged,
    )


def read_phase4_reference_rows(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["node_id"]): row for row in rows}


def validate_target_order(targets: list[Any], phase4_rows: dict[int, dict[str, Any]]) -> None:
    target_ids = [target.node_id for target in targets]
    reference_ids = list(phase4_rows)
    if target_ids != reference_ids:
        raise ValueError(
            f"PGExplainer target order {target_ids} does not match Phase-4 order {reference_ids}"
        )


def verify_candidate_parity(
    data: Any,
    candidate_edge_mask: torch.Tensor,
    reference: dict[str, Any],
) -> None:
    candidate_count = int(candidate_edge_mask.sum().item())
    ground_truth_edge_mask = data.edge_mask.detach().cpu().bool()
    positive_count = int((ground_truth_edge_mask & candidate_edge_mask).sum().item())
    negative_count = candidate_count - positive_count
    expected = {
        "candidate_edge_count": candidate_count,
        "positive_gt_edges": positive_count,
        "negative_gt_edges": negative_count,
    }
    for key, value in expected.items():
        if int(reference[key]) != value:
            raise ValueError(
                f"Phase-4 parity mismatch for {key}: PGExplainer={value}, "
                f"Phase4={reference[key]}"
            )


def verify_phase4_metric_parity(row: dict[str, Any], reference: dict[str, Any]) -> None:
    for key in [
        "candidate_edge_count",
        "positive_gt_edges",
        "negative_gt_edges",
        "selected_edge_count",
        "original_prediction",
    ]:
        if int(row[key]) != int(reference[key]):
            raise ValueError(
                f"Phase-4 parity mismatch for {key}: PGExplainer={row[key]}, "
                f"Phase4={reference[key]}"
            )


def flatten_pgexplainer_metrics(
    *,
    metrics: dict[str, Any],
    target: Any,
    explainer_checkpoint_path: Path,
    explainer_training_seed: int,
    inference_runtime_seconds: float,
    seed: int,
    status: str,
    error: str = "",
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
        "inference_runtime_seconds": inference_runtime_seconds,
        "explainer_checkpoint": str(explainer_checkpoint_path),
        "explainer_training_seed": explainer_training_seed,
        "seed": seed,
        "status": status,
        "error": error,
    }


def failure_pgexplainer_row(target: Any, explainer_checkpoint_path: Path, error: Exception) -> dict:
    row = {field: None for field in PGEXPLAINER_PER_NODE_FIELDS}
    row.update(
        {
            "node_id": target.node_id,
            "label": target.label,
            "prediction": target.prediction,
            "confidence": target.confidence,
            "explainer_checkpoint": str(explainer_checkpoint_path),
            "status": "failed",
            "error": str(error),
        }
    )
    return row


def write_pgexplainer_per_node_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PGEXPLAINER_PER_NODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_pgexplainer_summary_metrics(
    path: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    success_rows = [row for row in rows if row["status"] == "success"]
    summary = []
    for metric in PGEXPLAINER_SUMMARY_METRICS:
        values = [
            float(row[metric])
            for row in success_rows
            if row.get(metric) not in (None, "")
        ]
        summary.append(summary_row(metric, values))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "n", "mean", "std", "median", "min", "max"],
        )
        writer.writeheader()
        writer.writerows(summary)
    return summary


def summary_row(metric: str, values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "metric": metric,
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "metric": metric,
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def write_raw_pgexplainer_explanation(
    *,
    output_dir: Path,
    data: Any,
    edge_scores: torch.Tensor,
    selected_edge_mask: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(edge_scores.detach().cpu(), output_dir / "edge_mask.pt")
    torch.save(None, output_dir / "node_mask.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                **metadata,
                "mask_shapes": {
                    "edge_mask": list(edge_scores.shape),
                    "node_mask": None,
                },
                "node_mask_available": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_selected_edges_csv(
        output_dir / "selected_edges.csv",
        edge_scores=edge_scores,
        data=data,
        selected_edge_mask=selected_edge_mask,
        candidate_edge_mask=candidate_edge_mask,
    )
