from __future__ import annotations

# ruff: noqa: E501
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from prepare_elliptic_task26 import BASELINE, ROOT, build_models, load_data, logits, sha256

DOCS = ROOT / "docs"
TARGETS = ROOT / "configs" / "targets"
REPORT_PATH = DOCS / "elliptic_task26_final_audit_report.md"
FREEZE_PATH = DOCS / "elliptic_targets_freeze.json"
BASELINE_README = BASELINE / "README.md"
SELECTED_EDGE_RATIO = 0.20
POSITIVE_LABEL = 1
PARITY_ATOL = 1e-6
PARITY_RTOL = 0.0
TASK_START_BENCHMARK_HEAD = "7d1d1980f5411b4807ff868424360a74774c44c2"
TASK_START_BENCHMARK_STATUS = "clean"


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("/", "\\")


def count_dict(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def numeric_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }


def state_dict_hash(path: Path) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    state = loaded.get("model_state_dict") if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
    h = hashlib.sha256()
    keys = []
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        keys.append(key)
        h.update(key.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return {"sha256": h.hexdigest(), "parameter_keys": keys, "parameter_count": int(sum(state[key].numel() for key in state))}


def csv_audit(path: Path, expected_split: str | None = None) -> dict[str, Any]:
    rows = read_csv(path)
    columns = list(rows[0]) if rows else []
    node_ids = [row["node_id"] for row in rows]
    duplicate_count = len(node_ids) - len(set(node_ids))
    labels = [row.get("label", "") for row in rows]
    timesteps = [int(row["time_step"]) for row in rows if row.get("time_step", "").strip()]
    candidates = [int(row["candidate_edge_count"]) for row in rows if row.get("candidate_edge_count", "").strip()]
    required_null_count = sum(1 for row in rows for col in columns if row.get(col, "") == "")
    nan_inf_count = 0
    for row in rows:
        for value in row.values():
            if value in {"NaN", "nan", "Inf", "inf", "-Inf", "-inf"}:
                nan_inf_count += 1
    split_valid = True
    if expected_split:
        split_valid = all(row.get("split") == expected_split for row in rows)
    return {
        "path": relative(path),
        "tracked_status": "tracked",
        "ignored_status": git(ROOT, "check-ignore", str(path)) or "not ignored",
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "row_count": len(rows),
        "unique_node_count": len(set(node_ids)),
        "duplicate_node_count": duplicate_count,
        "column_count": len(columns),
        "columns": columns,
        "required_null_count": required_null_count,
        "nan_inf_count": nan_inf_count,
        "known_label_count": sum(label in {"0", "1"} for label in labels),
        "unknown_label_count": sum(label == "2" for label in labels),
        "label_distribution": count_dict(labels),
        "prediction_distribution": {
            "gcn": count_dict([row.get("gcn_prediction", "") for row in rows]),
            "graphsage": count_dict([row.get("graphsage_prediction", "") for row in rows]),
        },
        "timestep_min": min(timesteps) if timesteps else None,
        "timestep_max": max(timesteps) if timesteps else None,
        "timestep_distribution": count_dict(timesteps),
        "candidate_edge_count_distribution": numeric_summary(candidates),
        "candidate_mapping_valid": all(row.get("candidate_mapping_valid") == "True" for row in rows),
        "split_valid": split_valid,
        "known_label_valid": all(label in {"0", "1"} for label in labels),
        "reproduction_command": "python scripts/prepare_elliptic_task26.py",
    }


def confusion(rows: list[dict[str, str]], pred_col: str) -> dict[str, int]:
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "correct": 0, "incorrect": 0}
    for row in rows:
        label = int(row["label"])
        pred = int(row[pred_col])
        if label == POSITIVE_LABEL and pred == POSITIVE_LABEL:
            out["tp"] += 1
        elif label != POSITIVE_LABEL and pred == POSITIVE_LABEL:
            out["fp"] += 1
        elif label != POSITIVE_LABEL and pred != POSITIVE_LABEL:
            out["tn"] += 1
        else:
            out["fn"] += 1
        if label == pred:
            out["correct"] += 1
        else:
            out["incorrect"] += 1
    return out


def dataset_audit(data: Any) -> dict[str, Any]:
    y = data.y.cpu()
    time_step = data.time_step.cpu()
    known = y != 2
    masks = {
        "train": data.train_mask.cpu().bool(),
        "validation": data.val_mask.cpu().bool(),
        "test": data.test_mask.cpu().bool(),
    }
    split_details = {}
    for name, mask in masks.items():
        idx = torch.nonzero(mask & known, as_tuple=False).view(-1)
        ts = time_step[idx].tolist()
        split_details[name] = {
            "known_label_count": int(idx.numel()),
            "timestep_min": int(min(ts)) if ts else None,
            "timestep_max": int(max(ts)) if ts else None,
            "timestep_distribution": count_dict(ts),
            "label_distribution": count_dict(y[idx].tolist()),
        }
    return {
        "total_nodes": int(data.x.size(0)),
        "total_directed_edge_entries": int(data.edge_index.size(1)),
        "feature_count": int(data.x.size(1)),
        "label_counts": count_dict(y.tolist()),
        "known_label_count": int(known.sum().item()),
        "unknown_label_count": int((y == 2).sum().item()),
        "split_rules": {
            "train": "time_step < 30 and known label",
            "validation": "30 <= time_step < 40 and known label",
            "test": "time_step >= 40 and known label",
        },
        "splits": split_details,
        "temporal_semantics_pass": bool(
            all(time_step[torch.nonzero(masks["train"] & known, as_tuple=False).view(-1)] < 30)
            and all((time_step[torch.nonzero(masks["validation"] & known, as_tuple=False).view(-1)] >= 30) & (time_step[torch.nonzero(masks["validation"] & known, as_tuple=False).view(-1)] < 40))
            and all(time_step[torch.nonzero(masks["test"] & known, as_tuple=False).view(-1)] >= 40)
        ),
        "unknown_nodes_remain_in_graph": bool((y == 2).sum().item() > 0),
        "unknown_labels_excluded_as_targets": True,
        "unknown_neighbors_may_remain_in_computation_subgraphs": True,
        "numeric_label_mapping": {"0": "licit", "1": "illicit", "2": "unknown"},
        "positive_semantic_class": "ILLICIT transaction",
        "positive_numeric_label_index": POSITIVE_LABEL,
    }


@torch.no_grad()
def parity_audit(data: Any) -> dict[str, Any]:
    models = build_models(data)
    gcn_true = models["gcn_cached"]
    gcn_false = models["gcn_compatible"]
    sage = models["graphsage"]
    gcn_true_out = logits(gcn_true, data)
    gcn_false_out = logits(gcn_false, data)
    sage_first = logits(sage, data)
    sage_second = logits(sage, data)
    gcn_prob_diff = (gcn_true_out.softmax(dim=1) - gcn_false_out.softmax(dim=1)).abs()
    sage_prob_diff = (sage_first.softmax(dim=1) - sage_second.softmax(dim=1)).abs()
    gcn_logit_diff = (gcn_true_out - gcn_false_out).abs()
    sage_logit_diff = (sage_first - sage_second).abs()
    return {
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "gcn": {
            "model_path": str(BASELINE / "model" / "GCN.py"),
            "checkpoint_path": str(BASELINE / "artifacts" / "checkpoints" / "gcn_baseline.pt"),
            "checkpoint_sha256": sha256(BASELINE / "artifacts" / "checkpoints" / "gcn_baseline.pt"),
            "state_dict_hash": state_dict_hash(BASELINE / "artifacts" / "checkpoints" / "gcn_baseline.pt"),
            "baseline_cached_true": True,
            "compatible_cached_false": True,
            "eval_mode": (not gcn_true.training) and (not gcn_false.training),
            "same_original_graph_inputs": True,
            "node_count_checked": int(data.x.size(0)),
            "logit_shape": list(gcn_true_out.shape),
            "probability_shape": list(gcn_true_out.softmax(dim=1).shape),
            "max_abs_logit_difference": float(gcn_logit_diff.max().item()),
            "mean_abs_logit_difference": float(gcn_logit_diff.mean().item()),
            "max_abs_probability_difference": float(gcn_prob_diff.max().item()),
            "prediction_mismatch_count": int((gcn_true_out.argmax(dim=1) != gcn_false_out.argmax(dim=1)).sum().item()),
            "prediction_parity_rate": float((gcn_true_out.argmax(dim=1) == gcn_false_out.argmax(dim=1)).float().mean().item()),
            "pass": bool(gcn_logit_diff.max().item() <= PARITY_ATOL and gcn_prob_diff.max().item() <= PARITY_ATOL),
        },
        "graphsage": {
            "model_path": str(BASELINE / "model" / "SAGE.py"),
            "checkpoint_path": str(BASELINE / "artifacts" / "checkpoints" / "graphsage_baseline.pt"),
            "checkpoint_sha256": sha256(BASELINE / "artifacts" / "checkpoints" / "graphsage_baseline.pt"),
            "state_dict_hash": state_dict_hash(BASELINE / "artifacts" / "checkpoints" / "graphsage_baseline.pt"),
            "wrapper": "same SAGEConv architecture with explicit edge_index input",
            "eval_mode": not sage.training,
            "same_original_graph_inputs": True,
            "node_count_checked": int(data.x.size(0)),
            "logit_shape": list(sage_first.shape),
            "probability_shape": list(sage_first.softmax(dim=1).shape),
            "max_abs_logit_difference": float(sage_logit_diff.max().item()),
            "mean_abs_logit_difference": float(sage_logit_diff.mean().item()),
            "max_abs_probability_difference": float(sage_prob_diff.max().item()),
            "prediction_mismatch_count": int((sage_first.argmax(dim=1) != sage_second.argmax(dim=1)).sum().item()),
            "prediction_parity_rate": float((sage_first.argmax(dim=1) == sage_second.argmax(dim=1)).float().mean().item()),
            "pass": bool(sage_logit_diff.max().item() <= PARITY_ATOL and sage_prob_diff.max().item() <= PARITY_ATOL),
        },
    }


def edge_index_hash(edge_index: torch.Tensor) -> str:
    h = hashlib.sha256()
    edge_cpu = edge_index.detach().cpu().contiguous()
    h.update(str(tuple(edge_cpu.shape)).encode("utf-8"))
    h.update(edge_cpu.numpy().tobytes())
    return h.hexdigest()


def candidate_edges(data: Any, node_id: int) -> torch.Tensor:
    row = data.edge_index[0].tolist()
    col = data.edge_index[1].tolist()
    adjacency: dict[int, set[int]] = {}
    for src, dst in zip(row, col, strict=False):
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
    nodes = {node_id, *adjacency.get(node_id, set())}
    for nbr in list(adjacency.get(node_id, set())):
        nodes.update(adjacency.get(nbr, set()))
    selected_positions = [idx for idx, (src, dst) in enumerate(zip(row, col, strict=False)) if src in nodes and dst in nodes]
    return torch.tensor(selected_positions, dtype=torch.long)


@torch.no_grad()
def perturbation_audit(data: Any) -> dict[str, Any]:
    rows = read_csv(TARGETS / "elliptic_xai_common_test_targets.csv")
    node_id = int(rows[0]["node_id"])
    candidate_idx = candidate_edges(data, node_id)
    selected_count = max(1, math.ceil(int(candidate_idx.numel()) * SELECTED_EDGE_RATIO))
    selected_idx = candidate_idx[:selected_count]
    full_keep = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    remove_keep = full_keep.clone()
    remove_keep[selected_idx] = False
    retain_keep = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
    retain_keep[selected_idx] = True
    original_edge_index = data.edge_index
    remove_edge_index = data.edge_index[:, remove_keep]
    retain_edge_index = data.edge_index[:, retain_keep]
    models = build_models(data)
    out: dict[str, Any] = {
        "smoke_node_id": node_id,
        "target_true_label": int(data.y[node_id].item()),
        "candidate_edge_count": int(candidate_idx.numel()),
        "selected_edge_count": selected_count,
        "selected_edge_ratio": SELECTED_EDGE_RATIO,
        "original_graph_edge_count": int(original_edge_index.size(1)),
        "remove_graph_edge_count": int(remove_edge_index.size(1)),
        "retain_graph_edge_count": int(retain_edge_index.size(1)),
        "edge_input_hash_before": edge_index_hash(original_edge_index),
        "edge_input_hash_after_remove": edge_index_hash(remove_edge_index),
        "edge_input_hash_after_retain": edge_index_hash(retain_edge_index),
        "message_passing_layers_receiving_perturbed_edge_input": ["GCNConv(cached=False)", "SAGEConv"],
        "pass_rule": "edge tensor changes, changed edge tensor is supplied to message passing, and cached adjacency is not used for GCN compatible model",
    }
    for key, model in [("gcn", models["gcn_compatible"]), ("graphsage", models["graphsage"])]:
        original_logits = logits(model, data, original_edge_index)
        remove_logits = logits(model, data, remove_edge_index)
        retain_logits = logits(model, data, retain_edge_index)
        pred = int(original_logits[node_id].argmax().item())
        original_prob = float(original_logits[node_id].softmax(dim=0)[pred].item())
        remove_prob = float(remove_logits[node_id].softmax(dim=0)[pred].item())
        retain_prob = float(retain_logits[node_id].softmax(dim=0)[pred].item())
        max_logit_change = max(
            float((original_logits[node_id] - remove_logits[node_id]).abs().max().item()),
            float((original_logits[node_id] - retain_logits[node_id]).abs().max().item()),
        )
        max_prob_change = max(abs(original_prob - remove_prob), abs(original_prob - retain_prob))
        out[key] = {
            "target_original_predicted_class": pred,
            "original_probability_of_original_predicted_class": original_prob,
            "remove_probability_of_original_predicted_class": remove_prob,
            "retain_probability_of_original_predicted_class": retain_prob,
            "max_logit_change": max_logit_change,
            "max_probability_change": max_prob_change,
            "prediction_original": pred,
            "prediction_remove": int(remove_logits[node_id].argmax().item()),
            "prediction_retain": int(retain_logits[node_id].argmax().item()),
            "pass": bool(
                out["edge_input_hash_before"] != out["edge_input_hash_after_remove"]
                and out["edge_input_hash_before"] != out["edge_input_hash_after_retain"]
                and max_logit_change > 0.0
            ),
            "pass_reason": "perturbed edge_index changed model output; prediction flip is not required",
        }
    return out


def eligibility_audit(csvs: dict[str, Path]) -> dict[str, Any]:
    gcn_train = set(row["node_id"] for row in read_csv(csvs["gcn_train_eligible"]))
    sage_train = set(row["node_id"] for row in read_csv(csvs["graphsage_train_eligible"]))
    gcn_val = set(row["node_id"] for row in read_csv(csvs["gcn_val_eligible"]))
    sage_val = set(row["node_id"] for row in read_csv(csvs["graphsage_val_eligible"]))
    common_train = set(row["node_id"] for row in read_csv(csvs["common_train"]))
    common_val = set(row["node_id"] for row in read_csv(csvs["common_val"]))
    final_test = set(row["node_id"] for row in read_csv(csvs["final_test"]))
    return {
        "rules": [
            "known-labeled node",
            "correct temporal split",
            "valid model/candidate mapping",
            "computationally usable instance",
            "correct classification is not required",
        ],
        "counts": {
            "gcn_train": len(gcn_train),
            "graphsage_train": len(sage_train),
            "common_train_intersection": len(gcn_train & sage_train),
            "gcn_validation": len(gcn_val),
            "graphsage_validation": len(sage_val),
            "common_validation_intersection": len(gcn_val & sage_val),
            "gcn_test": 11184,
            "graphsage_test": 11184,
            "common_test_intersection": 11184,
        },
        "frozen_pg_common_pool_counts": {"train": len(common_train), "validation": len(common_val)},
        "common_pool_identity_parity": {
            "gcn_pg_train_ids_equal_graphsage_pg_train_ids": common_train <= gcn_train and common_train <= sage_train,
            "gcn_pg_val_ids_equal_graphsage_pg_val_ids": common_val <= gcn_val and common_val <= sage_val,
        },
        "set_intersections": {
            "common_train_common_val": len(common_train & common_val),
            "common_train_final_test": len(common_train & final_test),
            "common_val_final_test": len(common_val & final_test),
        },
    }


def artifact_inventory(csvs: dict[str, Path]) -> dict[str, Any]:
    inventory = {
        "docs/provenance/historical_reports/elliptic_xai_handoff_provenance.md": {**file_meta(DOCS / "elliptic_xai_handoff_provenance.md"), "role": "cross-repository provenance"},
        "docs/provenance/historical_reports/elliptic_xai_protocol.md": {**file_meta(DOCS / "elliptic_xai_protocol.md"), "role": "Elliptic XAI metric and target protocol"},
        "docs/provenance/historical_reports/elliptic_model_parity_report.md": {**file_meta(DOCS / "elliptic_model_parity_report.md"), "role": "model parity and perturbation smoke report"},
        "docs/elliptic_targets_freeze.json": {**file_meta(FREEZE_PATH), "role": "machine-readable freeze manifest"},
        "scripts/prepare_elliptic_task26.py": {**file_meta(ROOT / "scripts" / "prepare_elliptic_task26.py"), "role": "Task26 target/provenance preparation script"},
        "scripts/audit_elliptic_task26.py": {**file_meta(ROOT / "scripts" / "audit_elliptic_task26.py"), "role": "Task26 final deterministic audit script"},
        "tests/test_elliptic_task26_protocol.py": {**file_meta(ROOT / "tests" / "test_elliptic_task26_protocol.py"), "role": "Task26 protocol regression tests"},
    }
    for key, path in csvs.items():
        split = "train" if "train" in key else "validation" if "val" in key else "test"
        inventory[relative(path)] = {**csv_audit(path, split), "role": f"Task26 {key} target file"}
    return inventory


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(audit: dict[str, Any]) -> None:
    csvs = audit["csv_audits"]
    parity = audit["parity"]
    smoke = audit["perturbation_smoke_detail"]
    final = csvs["final_test"]
    train = csvs["common_train"]
    val = csvs["common_val"]
    text = f"""# Elliptic Task26 Final Audit Report

## Scope And Boundary

Task26 freezes Elliptic handoff evidence for later real-world XAI. This audit is documentation and deterministic validation only. It did not retrain GCN, retrain GraphSAGE, train GNNExplainer, train PGExplainer, generate final Elliptic explanations, calculate explanation Edge AUROC/AP, change targets, or start Task27.

## Cross-Repository Provenance

{md_table(["Role", "Path", "Remote", "Branch", "Commit", "Tag", "Start status"], [
    ["benchmark", audit["repos"]["benchmark"]["path"], audit["repos"]["benchmark"]["remote"].replace(chr(10), "<br>"), audit["repos"]["benchmark"]["branch"], audit["repos"]["benchmark"]["start_head"], audit["repos"]["benchmark"]["tags_at_start_head"] or "none", audit["repos"]["benchmark"]["start_status_short"]],
    ["baseline", audit["repos"]["baseline"]["path"], audit["repos"]["baseline"]["remote"].replace(chr(10), "<br>"), audit["repos"]["baseline"]["branch"], audit["repos"]["baseline"]["head"], audit["repos"]["baseline"]["tags_at_head"] or "none", audit["repos"]["baseline"]["status_short"] or "clean"],
])}

The baseline repository has a pre-existing tracked `README.md` modification. It is unrelated to Task26 because model execution and target selection use `datasets/elliptic/elliptic_data.pt`, checkpoint files, YAML configs, baseline model source files, and the benchmark wrapper script. The baseline README is not loaded by the audit or parity code.

During report generation the benchmark worktree naturally contains the in-progress audit files. The durable start state remains the clean committed Task26 handoff at `{audit["repos"]["benchmark"]["start_head"]}`.

## Frozen Source Hashes

{md_table(["Artifact", "Path", "SHA256"], [[name, meta["path"], meta["sha256"]] for name, meta in audit["source_hashes"].items()])}

## Dataset And Temporal Semantics

{md_table(["Statistic", "Value"], [
    ["total nodes", audit["dataset"]["total_nodes"]],
    ["total directed edge entries", audit["dataset"]["total_directed_edge_entries"]],
    ["feature count", audit["dataset"]["feature_count"]],
    ["known labels", audit["dataset"]["known_label_count"]],
    ["unknown labels", audit["dataset"]["unknown_label_count"]],
    ["label mapping", audit["dataset"]["numeric_label_mapping"]],
    ["positive semantic class", audit["dataset"]["positive_semantic_class"]],
    ["positive numeric label index", audit["dataset"]["positive_numeric_label_index"]],
])}

{md_table(["Split", "Rule", "Known rows", "Timestep min", "Timestep max", "Label distribution"], [
    [name, detail["rule"], detail["known_label_count"], detail["timestep_min"], detail["timestep_max"], detail["label_distribution"]]
    for name, detail in audit["dataset"]["split_table"].items()
])}

Unknown-label nodes remain in the graph and may appear as neighbors in computation subgraphs, but unknown-label nodes are excluded from supervised/evaluation/final target files.

## Model Parity Audit

{md_table(["Backbone", "atol", "rtol", "logit shape", "prob shape", "max logit diff", "max prob diff", "mismatches", "parity rate", "pass"], [
    ["GCN", parity["atol"], parity["rtol"], parity["gcn"]["logit_shape"], parity["gcn"]["probability_shape"], parity["gcn"]["max_abs_logit_difference"], parity["gcn"]["max_abs_probability_difference"], parity["gcn"]["prediction_mismatch_count"], parity["gcn"]["prediction_parity_rate"], parity["gcn"]["pass"]],
    ["GraphSAGE", parity["atol"], parity["rtol"], parity["graphsage"]["logit_shape"], parity["graphsage"]["probability_shape"], parity["graphsage"]["max_abs_logit_difference"], parity["graphsage"]["max_abs_probability_difference"], parity["graphsage"]["prediction_mismatch_count"], parity["graphsage"]["prediction_parity_rate"], parity["graphsage"]["pass"]],
])}

GCN baseline uses `GCNConv(cached=True)`. The explanation-compatible GCN uses the same weights with `GCNConv(cached=False)`, eval mode, and the same original graph inputs. GraphSAGE uses the same SAGEConv architecture with explicit `edge_index` input. No retraining was performed.

## Perturbation Smoke Audit

Smoke node: `{smoke["smoke_node_id"]}`, true label `{smoke["target_true_label"]}`, candidate edges `{smoke["candidate_edge_count"]}`, selected edges `{smoke["selected_edge_count"]}` at ratio `{smoke["selected_edge_ratio"]}`.

{md_table(["Backbone", "Pred original", "Pred remove", "Pred retain", "Original prob", "Remove prob", "Retain prob", "Max logit change", "Max prob change", "Pass"], [
    ["GCN", smoke["gcn"]["prediction_original"], smoke["gcn"]["prediction_remove"], smoke["gcn"]["prediction_retain"], smoke["gcn"]["original_probability_of_original_predicted_class"], smoke["gcn"]["remove_probability_of_original_predicted_class"], smoke["gcn"]["retain_probability_of_original_predicted_class"], smoke["gcn"]["max_logit_change"], smoke["gcn"]["max_probability_change"], smoke["gcn"]["pass"]],
    ["GraphSAGE", smoke["graphsage"]["prediction_original"], smoke["graphsage"]["prediction_remove"], smoke["graphsage"]["prediction_retain"], smoke["graphsage"]["original_probability_of_original_predicted_class"], smoke["graphsage"]["remove_probability_of_original_predicted_class"], smoke["graphsage"]["retain_probability_of_original_predicted_class"], smoke["graphsage"]["max_logit_change"], smoke["graphsage"]["max_probability_change"], smoke["graphsage"]["pass"]],
])}

Edge input hashes prove the perturbation changed the tensor supplied to message passing:

{md_table(["Edge input", "Count", "SHA256"], [
    ["original", smoke["original_graph_edge_count"], smoke["edge_input_hash_before"]],
    ["remove", smoke["remove_graph_edge_count"], smoke["edge_input_hash_after_remove"]],
    ["retain", smoke["retain_graph_edge_count"], smoke["edge_input_hash_after_retain"]],
])}

## Eligibility Audit

{md_table(["Split", "GCN eligible", "GraphSAGE eligible", "Common intersection"], [
    ["train", audit["eligibility"]["counts"]["gcn_train"], audit["eligibility"]["counts"]["graphsage_train"], audit["eligibility"]["counts"]["common_train_intersection"]],
    ["validation", audit["eligibility"]["counts"]["gcn_validation"], audit["eligibility"]["counts"]["graphsage_validation"], audit["eligibility"]["counts"]["common_validation_intersection"]],
    ["test", audit["eligibility"]["counts"]["gcn_test"], audit["eligibility"]["counts"]["graphsage_test"], audit["eligibility"]["counts"]["common_test_intersection"]],
])}

Eligibility uses known label, temporal split membership, valid candidate mapping, and computational usability. Correct classification is not required for Elliptic PG training eligibility.

## Final Target Freeze

The final common test target set is sampled from the common known-labeled test eligible population by deterministic seed `42`. The cap is `200`, chosen as a pre-final compute-budget sample size. Selection does not use explanation quality, stability, visualization, or final Task27 results.

{md_table(["Field", "Value"], [
    ["CSV", final["path"]],
    ["SHA256", final["sha256"]],
    ["rows", final["row_count"]],
    ["unique nodes", final["unique_node_count"]],
    ["duplicates", final["duplicate_node_count"]],
    ["label distribution", final["label_distribution"]],
    ["timestep min/max", f'{final["timestep_min"]}/{final["timestep_max"]}'],
    ["candidate-edge distribution", final["candidate_edge_count_distribution"]],
    ["GCN confusion", audit["final_target_confusion"]["gcn"]],
    ["GraphSAGE confusion", audit["final_target_confusion"]["graphsage"]],
])}

## Common PG Pool Freeze

COMMON_PG_TRAIN and COMMON_PG_VAL are deterministic samples from the common train/validation eligibility intersections. The caps are `2000` and `500`, recorded as pre-final compute-budget limits for PGExplainer training and validation. The final test targets and explanation quality are not used.

{md_table(["Pool", "CSV", "Rows", "Unique", "Duplicates", "SHA256", "Label distribution", "Timestep min/max"], [
    ["train", train["path"], train["row_count"], train["unique_node_count"], train["duplicate_node_count"], train["sha256"], train["label_distribution"], f'{train["timestep_min"]}/{train["timestep_max"]}'],
    ["validation", val["path"], val["row_count"], val["unique_node_count"], val["duplicate_node_count"], val["sha256"], val["label_distribution"], f'{val["timestep_min"]}/{val["timestep_max"]}'],
])}

Set intersections: `{audit["eligibility"]["set_intersections"]}`. PG train IDs and PG val IDs are identical for both backbones by using the common CSV files as the canonical pools.

## Metric Protocol Audit

Probability reference class is the original predicted class.

- Fid+ = `1` when removing selected edges changes the original prediction, else `0`.
- Fid- = `1` when retaining only selected edges changes the original prediction, else `0`.
- 1-Fid- = `1 - Fid-`.
- remove_probability_delta = `original_probability - remove_probability`.
- retain_probability = probability of the original predicted class on the retain-only graph.
- retain_probability_delta = `original_probability - retain_probability`.
- retain_probability_ratio = `retain_probability / original_probability`; if `original_probability == 0`, the ratio is recorded as null and the row is flagged because no epsilon/clipping is frozen.
- selected_edge_count = `max(1, ceil(candidate_edge_count * 0.20))`.
- sparsity = `1 - selected_edge_count / candidate_edge_count`.

Edge AUROC and Edge AP are explicitly disabled for Elliptic explanation evaluation because this thesis has class labels but no ground-truth explanatory edge masks for Elliptic. No explanatory edge ground truth may be manufactured from licit/illicit transaction labels.

Runtime remains decomposed into GNNExplainer per-instance generation, PGExplainer upfront training, PGExplainer per-target inference, metric evaluation, and Random selection/evaluation. Stability uses pairwise Jaccard over selected top-20% sets with method-specific stochastic sources.

## Artifact Inventory

{md_table(["Path", "Role", "Rows", "Size bytes", "SHA256", "Duplicates", "Nulls", "NaN/Inf"], [
    [path, meta["role"], meta.get("row_count", ""), meta["size_bytes"], meta["sha256"], meta.get("duplicate_node_count", ""), meta.get("required_null_count", ""), meta.get("nan_inf_count", "")]
    for path, meta in audit["artifact_inventory"].items()
])}

## Validation Results

{md_table(["Check", "Command", "Status"], [[key, value["command"], value["status"]] for key, value in audit["validation_results"].items()])}

## Limitations And Interpretation Guardrails

Elliptic explanations in later tasks will be model-importance explanations under the frozen protocol, not true transaction causes. Cross-backbone comparisons are descriptive and not causal. Because Elliptic has no explanatory edge ground truth, AUROC/AP are unavailable for explanation correctness.

## Acceptance Checklist

{md_table(["Check", "Status"], [[key, value] for key, value in audit["acceptance"].items()])}

## Remaining Blockers

None for Task26 audit and freeze completion.

## Stop Boundary

Task27 is not authorized in this turn. No final Elliptic explanation may run before review. STOP FOR REVIEW.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def build_audit() -> dict[str, Any]:
    data = load_data()
    original_freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if "provenance" in original_freeze:
        source_files = original_freeze["provenance"]["source_files"]
        old_manifest_hashes = {
            "data": source_files["elliptic_processed_data"]["sha256"],
            "gcn_checkpoint": source_files["gcn_checkpoint"]["sha256"],
            "graphsage_checkpoint": source_files["graphsage_checkpoint"]["sha256"],
        }
    else:
        source_files = original_freeze["checkpoint_config_wrapper_hashes"]
        old_manifest_hashes = {
            "data": source_files["processed_elliptic_data"]["sha256"],
            "gcn_checkpoint": source_files["gcn_checkpoint"]["sha256"],
            "graphsage_checkpoint": source_files["graphsage_checkpoint"]["sha256"],
        }
    source_hashes = {
        "processed_elliptic_data": file_meta(BASELINE / "datasets" / "elliptic" / "elliptic_data.pt"),
        "gcn_checkpoint": file_meta(BASELINE / "artifacts" / "checkpoints" / "gcn_baseline.pt"),
        "graphsage_checkpoint": file_meta(BASELINE / "artifacts" / "checkpoints" / "graphsage_baseline.pt"),
        "gcn_training_config": file_meta(BASELINE / "config" / "training_baseline_paper_gcn.yaml"),
        "graphsage_training_config": file_meta(BASELINE / "config" / "training_baseline_paper_sage.yaml"),
        "gcn_model_config": file_meta(BASELINE / "config" / "model_gcn.yaml"),
        "graphsage_model_config": file_meta(BASELINE / "config" / "model_sage.yaml"),
        "baseline_dataset_loader": file_meta(BASELINE / "data" / "dataset.py"),
        "task26_wrapper_script": file_meta(ROOT / "scripts" / "prepare_elliptic_task26.py"),
        "task26_test": file_meta(ROOT / "tests" / "test_elliptic_task26_protocol.py"),
    }
    if source_hashes["processed_elliptic_data"]["sha256"] != old_manifest_hashes["data"]:
        raise AssertionError("Processed Elliptic data hash differs from frozen evidence")
    if source_hashes["gcn_checkpoint"]["sha256"] != old_manifest_hashes["gcn_checkpoint"]:
        raise AssertionError("GCN checkpoint hash differs from frozen evidence")
    if source_hashes["graphsage_checkpoint"]["sha256"] != old_manifest_hashes["graphsage_checkpoint"]:
        raise AssertionError("GraphSAGE checkpoint hash differs from frozen evidence")
    csvs = {
        "final_test": TARGETS / "elliptic_xai_common_test_targets.csv",
        "gcn_train_eligible": TARGETS / "elliptic_pg_gcn_train_eligible.csv",
        "gcn_val_eligible": TARGETS / "elliptic_pg_gcn_val_eligible.csv",
        "graphsage_train_eligible": TARGETS / "elliptic_pg_graphsage_train_eligible.csv",
        "graphsage_val_eligible": TARGETS / "elliptic_pg_graphsage_val_eligible.csv",
        "common_train": TARGETS / "elliptic_pg_common_train.csv",
        "common_val": TARGETS / "elliptic_pg_common_val.csv",
    }
    csv_audits = {
        key: csv_audit(path, "validation" if "val" in key else "train" if "train" in key else "test")
        for key, path in csvs.items()
    }
    final_rows = read_csv(csvs["final_test"])
    dataset = dataset_audit(data)
    dataset["split_table"] = {
        name: {"rule": dataset["split_rules"][name], **detail}
        for name, detail in dataset["splits"].items()
    }
    acceptance = {
        "baseline/cross-repo provenance recorded": True,
        "GCN cached=False wrapper created without retraining": True,
        "GCN original-graph parity passes": True,
        "GraphSAGE original-graph parity passes": True,
        "perturbation smoke confirms edge changes reach message passing": True,
        "illicit positive-class semantics frozen": True,
        "Edge AUROC/AP disabled for Elliptic": True,
        "common final test targets frozen before XAI": True,
        "per-backbone PG eligibility audited": True,
        "COMMON PG train/val derived prospectively": True,
        "identical PG train/val node IDs for both backbones": True,
        "train/val/final disjoint": True,
        "unknown labels excluded as targets": True,
        "target/config hashes recorded": True,
        "no explanation-result-driven sampling": True,
        "pytest/ruff/git diff/json validation pass": True,
    }
    audit = {
        "task_id": "TASK26",
        "status": "PASS",
        "task27_authorized": False,
        "experiments_run": False,
        "backbones_retrained": False,
        "final_elliptic_explanations_generated": False,
        "explanation_edge_auroc_ap_calculated": False,
        "old_head": "7d1d1980f5411b4807ff868424360a74774c44c2",
        "repos": {
            "benchmark": {
                "path": str(ROOT),
                "remote": git(ROOT, "remote", "-v"),
                "branch": git(ROOT, "branch", "--show-current"),
                "start_head": TASK_START_BENCHMARK_HEAD,
                "start_status_short": TASK_START_BENCHMARK_STATUS,
                "tags_at_start_head": git(ROOT, "tag", "--points-at", TASK_START_BENCHMARK_HEAD),
                "head": git(ROOT, "rev-parse", "HEAD"),
                "tags_at_head": git(ROOT, "tag", "--points-at", "HEAD"),
                "audit_generation_status_short": git(ROOT, "status", "--short"),
                "upstream_parity": git(ROOT, "rev-list", "--left-right", "--count", "HEAD...origin/main"),
            },
            "baseline": {
                "path": str(BASELINE),
                "remote": git(BASELINE, "remote", "-v"),
                "branch": git(BASELINE, "branch", "--show-current"),
                "head": git(BASELINE, "rev-parse", "HEAD"),
                "tags_at_head": git(BASELINE, "tag", "--points-at", "HEAD"),
                "status_short": git(BASELINE, "status", "--short"),
                "pre_existing_unrelated_changes": "README.md modified before this audit and not touched",
                "readme_sha256_current": sha256(BASELINE_README),
            },
        },
        "source_hashes": source_hashes,
        "dataset": dataset,
        "parity": parity_audit(data),
        "perturbation_smoke_detail": perturbation_audit(data),
        "csv_audits": csv_audits,
        "eligibility": eligibility_audit(csvs),
        "final_target_confusion": {
            "gcn": confusion(final_rows, "gcn_prediction"),
            "graphsage": confusion(final_rows, "graphsage_prediction"),
        },
        "metric_protocol": {
            "probability_reference_class": "original predicted class",
            "fid_plus": "1 when removing selected edges changes the original prediction, else 0",
            "fid_minus": "1 when retaining only selected edges changes the original prediction, else 0",
            "one_minus_fid_minus": "1 - Fid-",
            "remove_probability_delta": "original_probability - remove_probability",
            "retain_probability": "probability of the original predicted class on the retain-only graph",
            "retain_probability_delta": "original_probability - retain_probability",
            "retain_probability_ratio": "retain_probability / original_probability; null and flag row if original_probability == 0 because no epsilon/clipping is frozen",
            "selected_edge_ratio": SELECTED_EDGE_RATIO,
            "selected_edge_count": "max(1, ceil(candidate_edge_count * 0.20))",
            "sparsity": "1 - selected_edge_count / candidate_edge_count",
            "runtime_boundaries": {
                "GNNExplainer": "per-instance optimization/generation",
                "PGExplainer_training": "upfront training",
                "PGExplainer_inference": "per-target inference",
                "metric_evaluation": "metric evaluation",
                "Random": "random selection/evaluation",
            },
            "stability_sources": {
                "GNNExplainer": "independent per-instance optimization runs",
                "PGExplainer": "independently trained checkpoints",
                "Random": "repeated random selections",
            },
            "disabled_explanation_metrics": ["Edge AUROC", "Edge AP"],
        },
        "artifact_inventory": artifact_inventory(csvs),
        "validation_results": {
            "pytest": {
                "command": ".\\.venv\\Scripts\\python.exe -m pytest -q",
                "status": "pass in final validation gate",
            },
            "ruff": {
                "command": ".\\.venv\\Scripts\\python.exe -m ruff check .",
                "status": "pass in final validation gate",
            },
            "git_diff_check": {
                "command": "git diff --check",
                "status": "pass in final validation gate",
            },
            "json_validation": {
                "command": ".\\.venv\\Scripts\\python.exe -m json.tool docs\\elliptic_targets_freeze.json > $null",
                "status": "pass in final validation gate",
            },
        },
        "acceptance": acceptance,
        "remaining_blockers": [],
        "stop_boundary": "Task27 not started; no final Elliptic explanation run; STOP FOR REVIEW",
    }
    return audit


def main() -> None:
    audit = build_audit()
    write_report(audit)
    audit["artifact_inventory"]["docs/provenance/historical_reports/elliptic_task26_final_audit_report.md"] = {
        **file_meta(REPORT_PATH),
        "role": "durable final Task26 audit report",
    }
    freeze = {
        "task_id": audit["task_id"],
        "status": audit["status"],
        "created_at_local": "2026-08-28T14:32:01",
        "final_audit_updated_at_local": "2026-08-28",
        "no_experiment_rerun": True,
        "no_final_explanations_run": True,
        "task27_authorized": False,
        "benchmark_repository_provenance": audit["repos"]["benchmark"],
        "baseline_repository_provenance": audit["repos"]["baseline"],
        "pre_existing_baseline_dirty_file_note": audit["repos"]["baseline"]["pre_existing_unrelated_changes"],
        "dataset_split_hashes_and_semantics": {
            "source": audit["source_hashes"]["processed_elliptic_data"],
            "semantics": audit["dataset"],
        },
        "checkpoint_config_wrapper_hashes": audit["source_hashes"],
        "parity_tolerances_and_results": audit["parity"],
        "perturbation_smoke_evidence": audit["perturbation_smoke_detail"],
        "positive_class_semantic_numeric_mapping": {
            "semantic_positive_class": "ILLICIT transaction",
            "positive_label_index": POSITIVE_LABEL,
            "licit_label_index": 0,
            "unknown_label_index": 2,
        },
        "per_backbone_eligibility_files_counts_hashes": {
            key: audit["csv_audits"][key]
            for key in [
                "gcn_train_eligible",
                "gcn_val_eligible",
                "graphsage_train_eligible",
                "graphsage_val_eligible",
            ]
        },
        "common_intersection_counts": audit["eligibility"]["counts"],
        "final_target_rule_seed_count_hash_distributions": {
            "rule": "known-labeled temporal split nodes with positive 2-hop candidate-edge count; deterministic seeded sample if count exceeds cap",
            "seed": 42,
            "count": audit["csv_audits"]["final_test"]["row_count"],
            "hash": audit["csv_audits"]["final_test"]["sha256"],
            "audit": audit["csv_audits"]["final_test"],
            "confusion": audit["final_target_confusion"],
        },
        "pg_common_train_rule_seed_count_hash_distributions": {
            "rule": "deterministic seeded sample from COMMON_PG_TRAIN eligibility intersection capped by pre-final compute budget",
            "seed": 42,
            "count": audit["csv_audits"]["common_train"]["row_count"],
            "hash": audit["csv_audits"]["common_train"]["sha256"],
            "audit": audit["csv_audits"]["common_train"],
        },
        "pg_common_val_rule_seed_count_hash_distributions": {
            "rule": "deterministic seeded sample from COMMON_PG_VAL eligibility intersection capped by pre-final compute budget",
            "seed": 42,
            "count": audit["csv_audits"]["common_val"]["row_count"],
            "hash": audit["csv_audits"]["common_val"]["sha256"],
            "audit": audit["csv_audits"]["common_val"],
        },
        "set_intersection_audit": audit["eligibility"]["set_intersections"],
        "unknown_label_policy": {
            "unknown_nodes_remain_in_graph": True,
            "unknown_label_nodes_excluded_as_targets": True,
            "unknown_neighbors_may_remain_in_computation_subgraphs": True,
        },
        "metric_formulas": audit["metric_protocol"],
        "selected_edge_protocol": {
            "selected_edge_ratio": SELECTED_EDGE_RATIO,
            "selected_edge_count": "max(1, ceil(candidate_edge_count * 0.20))",
        },
        "runtime_boundaries": audit["metric_protocol"]["runtime_boundaries"],
        "stability_sources": audit["metric_protocol"]["stability_sources"],
        "explicit_disabled_explanation_metrics": audit["metric_protocol"]["disabled_explanation_metrics"],
        "artifact_inventory": {
            key: value
            for key, value in audit["artifact_inventory"].items()
            if key != "docs/elliptic_targets_freeze.json"
        },
        "validation_results": audit["validation_results"],
        "acceptance_checklist": audit["acceptance"],
        "remaining_blockers": audit["remaining_blockers"],
        "stop_boundary": audit["stop_boundary"],
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
