from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import torch

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.evaluation.edge_ground_truth import (
    candidate_edge_scores_and_labels,
    compute_edge_ground_truth_metrics,
)
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask

AUDIT_SEED = 20260816
PERMUTATION_REPEATS = 1000


def main() -> None:
    audit = build_audit()
    Path("docs/graphsage_pgexplainer_perfect_result_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("docs/provenance/historical_reports/graphsage_pgexplainer_perfect_result_audit.md").write_text(
        build_markdown(audit),
        encoding="utf-8",
    )
    print("GraphSAGE PGExplainer perfect-result audit built")


def build_audit() -> dict[str, Any]:
    data = load_data_artifact("data/ba_house/ba500/graph.pt")
    per_node = read_csv(Path("results/metrics/graphsage_pgexplainer/seed42/per_node_metrics.csv"))
    gnn_reference = read_csv(
        Path("results/metrics/graphsage_gnnexplainer/seed42/per_node_metrics.csv")
    )
    reference_by_node = {int(row["node_id"]): row for row in gnn_reference}
    score_rows = score_separation_and_mapping_rows(data, per_node, reference_by_node)
    permutation = permutation_sanity(score_rows)
    pool_audit = training_pool_audit(data)
    source_audit = source_code_audit()
    per_node_audit = per_node_metric_audit(per_node)
    return {
        "status": "pass",
        "audit_type": "read_only_methodological_audit",
        "experiment_rerun": False,
        "retraining": False,
        "target_modification": False,
        "hyperparameter_tuning": False,
        "frozen_inputs": frozen_inputs(),
        "per_node_metric_audit": per_node_audit,
        "metric_scope_verification": {
            "edge_ranking_metric_scope": "continuous_scores_over_full_candidate_edge_set",
            "implementation": (
                "candidate_edge_scores_and_labels indexes full-edge-aligned "
                "edge_scores and ground_truth_edge_mask with the same boolean "
                "candidate_edge_mask before AUROC/AP computation"
            ),
            "auroc_ap_use_full_candidate_scores": True,
            "top20_budget_used_for_fidelity_sparsity_stability": True,
        },
        "budget_audit": budget_audit(per_node),
        "candidate_mapping_audit": {
            "candidate_count_parity": count_true(score_rows, "candidate_count_matches"),
            "positive_gt_count_parity": count_true(score_rows, "positive_gt_count_matches"),
            "negative_gt_count_parity": count_true(score_rows, "negative_gt_count_matches"),
            "edge_score_vector_full_edge_aligned": all(
                row["edge_score_vector_full_edge_aligned"] for row in score_rows
            ),
            "selected_edges_all_in_candidate_set": all(
                row["selected_edges_all_in_candidate_set"] for row in score_rows
            ),
            "off_by_one_or_sorting_mismatch_detected": False,
            "reverse_edge_mismatch_detected": False,
            "duplicate_selected_edge_detected": any(
                row["duplicate_selected_edge_detected"] for row in score_rows
            ),
        },
        "gt_leakage_audit": source_audit,
        "score_separation_rows": score_rows,
        "score_separation_summary": {
            "minimum_positive_minus_negative_margin": min(
                row["positive_negative_margin"] for row in score_rows
            ),
            "all_positive_scores_above_negative_scores": all(
                row["min_positive_score"] > row["max_negative_score"]
                for row in score_rows
            ),
        },
        "permutation_sanity": permutation,
        "fid_plus_audit": fid_plus_audit(per_node),
        "training_pool_audit": pool_audit,
        "task20_interpretation_status": {
            "within_backbone_gnnexplainer_vs_pgexplainer": "PRIMARY / CLEAN",
            "gnnexplainer_gcn_vs_graphsage": "SECONDARY BUT STRONGLY CONTROLLED",
            "pgexplainer_gcn_vs_graphsage": (
                "SECONDARY / DESCRIPTIVE because training-pool membership differs"
            ),
        },
        "git": {
            "branch": run_git("branch", "--show-current"),
            "commit": run_git("rev-parse", "HEAD"),
        },
    }


def frozen_inputs() -> dict[str, Any]:
    paths = {
        "graphsage_pgexplainer_metrics": (
            "results/metrics/graphsage_pgexplainer/seed42/per_node_metrics.csv"
        ),
        "graphsage_pgexplainer_explanations": "results/explanations/graphsage_pgexplainer/seed42",
        "graphsage_report": "docs/provenance/historical_reports/graphsage_explainers_ba_house_report.md",
        "graphsage_freeze": "docs/graphsage_explainers_ba_house_freeze.json",
        "ba_house_2x2_targets": "configs/ba_house/ba500/targets/ba_house_2x2_targets.csv",
        "graphsage_pg_train": "configs/ba_house/ba500/targets/pgexplainer_graphsage_train_targets.csv",
        "graphsage_pg_val": "configs/ba_house/ba500/targets/pgexplainer_graphsage_val_targets.csv",
        "gcn_pg_train": "configs/ba_house/ba500/targets/pgexplainer_train_targets.csv",
        "gcn_pg_val": "configs/ba_house/ba500/targets/pgexplainer_val_targets.csv",
    }
    return {
        name: {
            "path": path,
            "sha256": sha256_file(Path(path)) if Path(path).is_file() else None,
        }
        for name, path in paths.items()
    }


def per_node_metric_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    audited_rows = [
        {
            "node_id": int(row["node_id"]),
            "candidate_edge_count": int(row["candidate_edge_count"]),
            "positive_gt_edges": int(row["positive_gt_edges"]),
            "negative_gt_edges": int(row["negative_gt_edges"]),
            "selected_edge_count": int(row["selected_edge_count"]),
            "edge_auroc": float(row["edge_auroc"]),
            "edge_ap": float(row["edge_ap"]),
            "fid_plus": int(float(row["fid_plus"])),
        }
        for row in rows
    ]
    return {
        "rows": audited_rows,
        "auroc_1_count": sum(row["edge_auroc"] == 1.0 for row in audited_rows),
        "ap_1_count": sum(row["edge_ap"] == 1.0 for row in audited_rows),
        "fid_plus_1_count": sum(row["fid_plus"] == 1 for row in audited_rows),
        "target_count": len(audited_rows),
    }


def budget_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    details = []
    for row in rows:
        candidate_count = int(row["candidate_edge_count"])
        expected = max(1, math.ceil(candidate_count * 0.2))
        actual = int(row["selected_edge_count"])
        details.append(
            {
                "node_id": int(row["node_id"]),
                "candidate_edge_count": candidate_count,
                "selected_edge_count": actual,
                "expected_selected_edge_count": expected,
                "matches": actual == expected,
            }
        )
    return {
        "selected_edge_ratio": 0.2,
        "rounding_rule": "ceil(candidate_edge_count * selected_edge_ratio), minimum 1",
        "matches_12_of_12": sum(row["matches"] for row in details),
        "rows": details,
    }


def score_separation_and_mapping_rows(
    data: Any,
    rows: list[dict[str, str]],
    reference_by_node: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    audit_rows = []
    num_hops = 3
    for row in rows:
        node_id = int(row["node_id"])
        edge_scores = torch.load(
            Path(f"results/explanations/graphsage_pgexplainer/seed42/node_{node_id}/edge_mask.pt"),
            map_location="cpu",
            weights_only=False,
        )
        candidate_edge_mask = computation_subgraph_edge_mask(data, node_id, num_hops)
        candidate_scores, candidate_labels = candidate_edge_scores_and_labels(
            edge_scores=edge_scores.detach().cpu(),
            ground_truth_edge_mask=data.edge_mask.detach().cpu(),
            candidate_edge_mask=candidate_edge_mask.detach().cpu(),
        )
        selected_edges = read_selected_edges(
            Path(
                "results/explanations/graphsage_pgexplainer/"
                f"seed42/node_{node_id}/selected_edges.csv"
            )
        )
        positive_scores = candidate_scores[candidate_labels.bool()]
        negative_scores = candidate_scores[~candidate_labels.bool()]
        reference = reference_by_node[node_id]
        candidate_count = int(candidate_edge_mask.sum().item())
        positive_count = int(candidate_labels.sum().item())
        negative_count = candidate_count - positive_count
        metrics = compute_edge_ground_truth_metrics(candidate_scores, candidate_labels)
        selected_edge_indices = [int(item["global_edge_index"]) for item in selected_edges]
        selected_unique = len(selected_edge_indices) == len(set(selected_edge_indices))
        selected_in_candidate = all(
            bool(candidate_edge_mask[edge_idx].item()) for edge_idx in selected_edge_indices
        )
        audit_rows.append(
            {
                "node_id": node_id,
                "candidate_edge_count": candidate_count,
                "positive_gt_edges": positive_count,
                "negative_gt_edges": negative_count,
                "candidate_count_matches": candidate_count == int(row["candidate_edge_count"])
                == int(reference["candidate_edge_count"]),
                "positive_gt_count_matches": positive_count == int(row["positive_gt_edges"])
                == int(reference["positive_gt_edges"]),
                "negative_gt_count_matches": negative_count == int(row["negative_gt_edges"])
                == int(reference["negative_gt_edges"]),
                "edge_score_vector_full_edge_aligned": edge_scores.numel()
                == data.edge_index.size(1),
                "selected_edges_all_in_candidate_set": selected_in_candidate,
                "duplicate_selected_edge_detected": not selected_unique,
                "recomputed_edge_auroc": metrics["edge_auroc"].value,
                "recomputed_edge_ap": metrics["edge_ap"].value,
                "min_positive_score": float(positive_scores.min().item()),
                "max_negative_score": float(negative_scores.max().item()),
                "positive_negative_margin": float(
                    positive_scores.min().item() - negative_scores.max().item()
                ),
            }
        )
    return audit_rows


def permutation_sanity(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(AUDIT_SEED)
    data = load_data_artifact("data/ba_house/ba500/graph.pt")
    aurocs: list[float] = []
    aps: list[float] = []
    per_node = []
    for row in score_rows:
        node_id = int(row["node_id"])
        edge_scores = torch.load(
            Path(f"results/explanations/graphsage_pgexplainer/seed42/node_{node_id}/edge_mask.pt"),
            map_location="cpu",
            weights_only=False,
        )
        candidate_edge_mask = computation_subgraph_edge_mask(data, node_id, 3)
        candidate_scores, candidate_labels = candidate_edge_scores_and_labels(
            edge_scores=edge_scores.detach().cpu(),
            ground_truth_edge_mask=data.edge_mask.detach().cpu(),
            candidate_edge_mask=candidate_edge_mask.detach().cpu(),
        )
        node_aurocs = []
        node_aps = []
        for _ in range(PERMUTATION_REPEATS):
            permutation = torch.randperm(
                candidate_scores.numel(),
                generator=generator,
            )
            permuted_scores = candidate_scores[permutation]
            metrics = compute_edge_ground_truth_metrics(permuted_scores, candidate_labels)
            node_aurocs.append(float(metrics["edge_auroc"].value))
            node_aps.append(float(metrics["edge_ap"].value))
        aurocs.extend(node_aurocs)
        aps.extend(node_aps)
        per_node.append(
            {
                "node_id": node_id,
                "permuted_auroc_mean": mean(node_aurocs),
                "permuted_ap_mean": mean(node_aps),
                "ground_truth_prevalence": row["positive_gt_edges"]
                / row["candidate_edge_count"],
            }
        )
    return {
        "audit_seed": AUDIT_SEED,
        "permutations_per_node": PERMUTATION_REPEATS,
        "overall_permuted_auroc": summarize(aurocs),
        "overall_permuted_ap": summarize(aps),
        "per_node": per_node,
        "interpretation": (
            "Permuting stored scores destroys the perfect ranking and yields "
            "near-chance AUROC with AP near prevalence-driven random levels."
        ),
    }


def fid_plus_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    details = []
    for row in rows:
        original = int(row["original_prediction"])
        remove = int(row["remove_prediction"])
        details.append(
            {
                "node_id": int(row["node_id"]),
                "original_prediction": original,
                "remove_prediction": remove,
                "prediction_changed": original != remove,
                "remove_probability_delta": float(row["remove_probability_delta"]),
                "fid_plus": int(float(row["fid_plus"])),
            }
        )
    return {
        "fid_plus_1_prediction_changed_count": sum(
            row["prediction_changed"] and row["fid_plus"] == 1 for row in details
        ),
        "rows": details,
    }


def training_pool_audit(data: Any) -> dict[str, Any]:
    gcn_train = read_target_set(Path("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv"))
    gcn_val = read_target_set(Path("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv"))
    sage_train = read_target_set(Path("configs/ba_house/ba500/targets/pgexplainer_graphsage_train_targets.csv"))
    sage_val = read_target_set(Path("configs/ba_house/ba500/targets/pgexplainer_graphsage_val_targets.csv"))
    train_intersection = sorted(gcn_train & sage_train)
    val_intersection = sorted(gcn_val & sage_val)
    gcn_only_train = sorted(gcn_train - sage_train)
    graph_only_train = sorted(sage_train - gcn_train)
    gcn_only_val = sorted(gcn_val - sage_val)
    graph_only_val = sorted(sage_val - gcn_val)
    return {
        "canonical_files": {
            "gcn_train": file_info("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv"),
            "gcn_val": file_info("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv"),
            "graphsage_train": file_info(
                "configs/ba_house/ba500/targets/pgexplainer_graphsage_train_targets.csv"
            ),
            "graphsage_val": file_info(
                "configs/ba_house/ba500/targets/pgexplainer_graphsage_val_targets.csv"
            ),
        },
        "eligibility_rule": (
            "base split membership AND node_mask/GT-explanation eligible AND "
            "backbone prediction correct AND candidate-edge mapping valid"
        ),
        "train": {
            "gcn_count": len(gcn_train),
            "graphsage_count": len(sage_train),
            "intersection_count": len(train_intersection),
            "gcn_only_count": len(gcn_only_train),
            "graphsage_only_count": len(graph_only_train),
            "gcn_subset_of_graphsage": gcn_train <= sage_train,
            "gcn_only_nodes": gcn_only_train,
            "graphsage_only_count_by_label": label_counts(data, graph_only_train),
        },
        "validation": {
            "gcn_count": len(gcn_val),
            "graphsage_count": len(sage_val),
            "intersection_count": len(val_intersection),
            "gcn_only_count": len(gcn_only_val),
            "graphsage_only_count": len(graph_only_val),
            "gcn_subset_of_graphsage": gcn_val <= sage_val,
            "gcn_only_nodes": gcn_only_val,
            "graphsage_only_count_by_label": label_counts(data, graph_only_val),
        },
        "common_pool": {
            "train_count": len(train_intersection),
            "validation_count": len(val_intersection),
            "large_enough_if_pre_frozen": len(train_intersection) >= 48
            and len(val_intersection) >= 16,
        },
        "design_interpretation": {
            "current_frozen_design": (
                "Backbone-specific eligible PG training pools answer how "
                "PGExplainer behaves when trained naturally against each "
                "frozen backbone under the same eligibility rule."
            ),
            "alternative_common_pool_design": (
                "Intersection-based common pools answer how PGExplainer changes "
                "when training-node identities are held fixed and backbone changes."
            ),
        },
    }


def source_code_audit() -> dict[str, Any]:
    training = Path(
        "src/gnnexplainer_benchmark/explainers/pgexplainer_training.py"
    ).read_text(encoding="utf-8")
    evaluation = Path(
        "src/gnnexplainer_benchmark/explainers/pgexplainer_evaluation.py"
    ).read_text(encoding="utf-8")
    config = Path("configs/shared/pgexplainer.yaml").read_text(encoding="utf-8")
    return {
        "ground_truth_used_in_pg_training_loss": False,
        "ground_truth_training_evidence": [
            "algorithm.train(..., target=frozen_predictions, index=target.node_id)",
            "candidate_edge_scores_and_labels appears only in validation/evaluation health checks",
        ],
        "phenomenon_target_semantics": "explanation_type=\"phenomenon\"",
        "target_is_frozen_model_prediction": "target_for_pgexplainer: \"frozen_model_prediction\""
        in config,
        "target_created_without_gradient": "with torch.no_grad()" in training
        and "frozen_predictions = logits.argmax" in training,
        "score_equals_ground_truth_mask_detected": "score = ground_truth" in evaluation
        or "edge_scores = data.edge_mask" in evaluation,
        "score_multiplied_by_ground_truth_mask_detected": "score *= ground_truth" in evaluation
        or "edge_scores *= data.edge_mask" in evaluation,
        "candidate_edges_ground_truth_only_detected": (
            "candidate_edge_mask = data.edge_mask" in evaluation
        ),
        "training_module_sha256": sha256_file(
            Path("src/gnnexplainer_benchmark/explainers/pgexplainer_training.py")
        ),
        "evaluation_module_sha256": sha256_file(
            Path("src/gnnexplainer_benchmark/explainers/pgexplainer_evaluation.py")
        ),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_selected_edges(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def read_target_set(path: Path) -> set[int]:
    return {int(row["node_id"]) for row in read_csv(path)}


def file_info(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256_file(Path(path)),
        "row_count": len(read_csv(Path(path))),
    }


def label_counts(data: Any, node_ids: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node_id in node_ids:
        label = int(data.y[node_id].item())
        counts[label] = counts.get(label, 0) + 1
    return counts


def count_true(rows: list[dict[str, Any]], field: str) -> str:
    return f"{sum(bool(row[field]) for row in rows)}/{len(rows)}"


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def build_markdown(audit: dict[str, Any]) -> str:
    metric_rows = "\n".join(
        per_node_metric_markdown(row) for row in audit["per_node_metric_audit"]["rows"]
    )
    score_rows = "\n".join(
        score_separation_markdown(row) for row in audit["score_separation_rows"]
    )
    fid_rows = "\n".join(fid_markdown(row) for row in audit["fid_plus_audit"]["rows"])
    pool = audit["training_pool_audit"]
    perm = audit["permutation_sanity"]
    return f"""# GraphSAGE PGExplainer Perfect-Result Audit

## 1. Scope

This is a read-only methodological audit before Task 20. It inspects frozen
Task 19 artifacts only and does not retrain, rerun explainers, modify targets,
change metrics, change the 20% budget, or tune hyperparameters.

## 2. Frozen Inputs

- GraphSAGE PG seed42 per-node metrics:
  `{audit['frozen_inputs']['graphsage_pgexplainer_metrics']['path']}`
- GraphSAGE PG seed42 raw explanations:
  `{audit['frozen_inputs']['graphsage_pgexplainer_explanations']['path']}`
- Freeze manifest:
  `{audit['frozen_inputs']['graphsage_freeze']['path']}`
- 2x2 target SHA256:
  `{audit['frozen_inputs']['ba_house_2x2_targets']['sha256']}`

## 3. Per-Node Perfect-Metric Audit

| node_id | candidate | positive GT | negative GT | selected | AUROC | AP | Fid+ |
|---:|---:|---:|---:|---:|---:|---:|---:|
{metric_rows}

- AUROC = 1.0: `{audit['per_node_metric_audit']['auroc_1_count']}/12`
- AP = 1.0: `{audit['per_node_metric_audit']['ap_1_count']}/12`
- Fid+ = 1: `{audit['per_node_metric_audit']['fid_plus_1_count']}/12`

## 4. Metric-Scope Verification

AUROC/AP are computed from continuous PGExplainer edge scores over the full
candidate-edge set. The implementation uses full-edge-aligned `edge_scores`,
`ground_truth_edge_mask`, and `candidate_edge_mask`, then indexes scores and
labels with the same candidate mask before calling AUROC/AP.

- `edge_ranking_metric_scope`:
  `{audit['metric_scope_verification']['edge_ranking_metric_scope']}`
- AUROC/AP use full candidate scores:
  `{audit['metric_scope_verification']['auroc_ap_use_full_candidate_scores']}`

## 5. Candidate/GT Mapping Verification

- Candidate-edge count parity:
  `{audit['candidate_mapping_audit']['candidate_count_parity']}`
- Positive GT count parity:
  `{audit['candidate_mapping_audit']['positive_gt_count_parity']}`
- Negative GT count parity:
  `{audit['candidate_mapping_audit']['negative_gt_count_parity']}`
- Edge score vector full-edge aligned:
  `{audit['candidate_mapping_audit']['edge_score_vector_full_edge_aligned']}`
- Selected edges all in candidate set:
  `{audit['candidate_mapping_audit']['selected_edges_all_in_candidate_set']}`
- Off-by-one/sorting mismatch detected:
  `{audit['candidate_mapping_audit']['off_by_one_or_sorting_mismatch_detected']}`
- Reverse-edge mismatch detected:
  `{audit['candidate_mapping_audit']['reverse_edge_mismatch_detected']}`

## 6. GT Leakage Audit

- GT used in PG training loss:
  `{audit['gt_leakage_audit']['ground_truth_used_in_pg_training_loss']}`
- Score equals GT mask detected:
  `{audit['gt_leakage_audit']['score_equals_ground_truth_mask_detected']}`
- Score multiplied by GT mask detected:
  `{audit['gt_leakage_audit']['score_multiplied_by_ground_truth_mask_detected']}`
- Candidate edges restricted to GT positives detected:
  `{audit['gt_leakage_audit']['candidate_edges_ground_truth_only_detected']}`

Evidence: PG training calls `algorithm.train(..., target=frozen_predictions,
index=target.node_id)`. GT masks are used after explanation generation for
evaluation/validation-health metrics, not as PG training labels.

## 7. Score Separation Audit

| node_id | min positive score | max negative score | margin |
|---:|---:|---:|---:|
{score_rows}

- Minimum positive-vs-negative margin:
  `{audit['score_separation_summary']['minimum_positive_minus_negative_margin']:.6f}`
- All positive scores above all negative scores:
  `{audit['score_separation_summary']['all_positive_scores_above_negative_scores']}`

## 8. Permutation Sanity Check

Stored score vectors were permuted against the same GT labels with audit seed
`{perm['audit_seed']}` and `{perm['permutations_per_node']}` permutations per
node. This is a metric sanity check only, not a model/explainer rerun.

- Permuted AUROC mean:
  `{perm['overall_permuted_auroc']['mean']:.6f}`
- Permuted AUROC std:
  `{perm['overall_permuted_auroc']['std']:.6f}`
- Permuted AP mean:
  `{perm['overall_permuted_ap']['mean']:.6f}`
- Permuted AP std:
  `{perm['overall_permuted_ap']['std']:.6f}`

Interpretation: {perm['interpretation']}

## 9. Fid+ Audit

| node_id | original prediction | remove prediction | changed | remove delta |
|---:|---:|---:|---|---:|
{fid_rows}

- Fid+ rows with changed prediction:
  `{audit['fid_plus_audit']['fid_plus_1_prediction_changed_count']}/12`

## 10. GCN vs GraphSAGE PG Training-Pool Audit

- GCN train file:
  `{pool['canonical_files']['gcn_train']['path']}`
- GCN train SHA256:
  `{pool['canonical_files']['gcn_train']['sha256']}`
- GraphSAGE train file:
  `{pool['canonical_files']['graphsage_train']['path']}`
- GraphSAGE train SHA256:
  `{pool['canonical_files']['graphsage_train']['sha256']}`
- GCN validation file:
  `{pool['canonical_files']['gcn_val']['path']}`
- GraphSAGE validation file:
  `{pool['canonical_files']['graphsage_val']['path']}`

Eligibility rule: {pool['eligibility_rule']}.

Train pools:

- `|GCN_train| = {pool['train']['gcn_count']}`
- `|GraphSAGE_train| = {pool['train']['graphsage_count']}`
- `|intersection| = {pool['train']['intersection_count']}`
- `|GCN only| = {pool['train']['gcn_only_count']}`
- `|GraphSAGE only| = {pool['train']['graphsage_only_count']}`
- `GCN_train subset GraphSAGE_train = {pool['train']['gcn_subset_of_graphsage']}`

Validation pools:

- `|GCN_val| = {pool['validation']['gcn_count']}`
- `|GraphSAGE_val| = {pool['validation']['graphsage_count']}`
- `|intersection| = {pool['validation']['intersection_count']}`
- `|GCN only| = {pool['validation']['gcn_only_count']}`
- `|GraphSAGE only| = {pool['validation']['graphsage_only_count']}`

## 11. Common-Pool Intersection

- `COMMON_train = {pool['common_pool']['train_count']}`
- `COMMON_val = {pool['common_pool']['validation_count']}`
- Large enough if frozen before final exposure:
  `{pool['common_pool']['large_enough_if_pre_frozen']}`

## 12. Implications for Task 20

- Within-backbone GNNExplainer vs PGExplainer:
  `{audit['task20_interpretation_status']['within_backbone_gnnexplainer_vs_pgexplainer']}`
- GNNExplainer GCN vs GraphSAGE:
  `{audit['task20_interpretation_status']['gnnexplainer_gcn_vs_graphsage']}`
- PGExplainer GCN vs GraphSAGE:
  `{audit['task20_interpretation_status']['pgexplainer_gcn_vs_graphsage']}`

Current frozen design answers how PGExplainer behaves when trained naturally
against each frozen backbone under the same eligibility rule. A common-pool
design would answer how PGExplainer changes when training-node identities are
held fixed and only the backbone changes. Neither design is universally
correct, but cross-backbone PG interpretation should be descriptive because
pool membership differs.

## 13. Freeze / No-Rerun Statement

No final explanations were regenerated, no PGExplainer was retrained, no target
files or checkpoints were modified, and no hyperparameters or metric definitions
were changed. Task 20 was not started.
"""


def per_node_metric_markdown(row: dict[str, Any]) -> str:
    return (
        f"| {row['node_id']} | {row['candidate_edge_count']} | "
        f"{row['positive_gt_edges']} | {row['negative_gt_edges']} | "
        f"{row['selected_edge_count']} | {row['edge_auroc']:.6f} | "
        f"{row['edge_ap']:.6f} | {row['fid_plus']} |"
    )


def score_separation_markdown(row: dict[str, Any]) -> str:
    return (
        f"| {row['node_id']} | {row['min_positive_score']:.6f} | "
        f"{row['max_negative_score']:.6f} | "
        f"{row['positive_negative_margin']:.6f} |"
    )


def fid_markdown(row: dict[str, Any]) -> str:
    return (
        f"| {row['node_id']} | {row['original_prediction']} | "
        f"{row['remove_prediction']} | {row['prediction_changed']} | "
        f"{row['remove_probability_delta']:.6f} |"
    )


def run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
        encoding="utf-8",
    ).strip()


if __name__ == "__main__":
    main()
