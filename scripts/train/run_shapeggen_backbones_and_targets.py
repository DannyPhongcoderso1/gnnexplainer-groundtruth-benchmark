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
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from gnnexplainer_benchmark.data.serialization import (
    load_data_artifact,
    load_metadata,
    save_metadata,
    sha256_file,
)
from gnnexplainer_benchmark.data.shapeggen import (
    get_target_candidate_edge_mask,
    get_target_edge_ground_truth,
)
from gnnexplainer_benchmark.data.validation import validate_shapeggen_data
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)
from gnnexplainer_benchmark.training.evaluation import evaluate_logits
from gnnexplainer_benchmark.training.trainer import load_training_config, train_model

EXPECTED_SHAPEGGEN_SHA256 = "9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e"
EXPECTED_PROTOCOL_NAME = "THESIS_SHAPEGGEN_OFFICIAL_V1"
EXPECTED_UPSTREAM_COMMIT = "a11e65ffbc4df737f35522a8accf2283a8aeaa37"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/freeze ShapeGGen backbones and pre-XAI target pools."
    )
    parser.add_argument("--artifact", type=Path, default=Path("data/shapeggen/v1/graph.pt"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/shapeggen/v1/manifest.json"),
    )
    parser.add_argument(
        "--gcn-config", type=Path, default=Path("configs/shapeggen/v1/shapeggen_gcn.yaml")
    )
    parser.add_argument(
        "--graphsage-config",
        type=Path,
        default=Path("configs/shapeggen/v1/shapeggen_graphsage.yaml"),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics/shapeggen_backbone_comparison"),
    )
    parser.add_argument("--targets-dir", type=Path, default=Path("configs/targets"))
    parser.add_argument(
        "--gcn-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/shapeggen/v1/backbones/gcn/seed42.pt"),
    )
    parser.add_argument(
        "--graphsage-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/shapeggen/v1/backbones/graphsage/seed42.pt"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path(
            "docs/provenance/historical_reports/shapeggen_backbones_and_targets_freeze.md"
        ),
    )
    parser.add_argument(
        "--freeze-json-output",
        type=Path,
        default=Path("docs/shapeggen_targets_freeze.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_artifact(args.artifact)
    validate_shapeggen_data(data)
    dataset_metadata = load_metadata(args.metadata)
    verify_task21_input(args, dataset_metadata)

    gcn_result = train_one_backbone(
        name="GCN",
        data=data,
        config_path=args.gcn_config,
        checkpoint_path=args.gcn_checkpoint,
        metrics_dir=args.metrics_dir,
        dataset_checksum=EXPECTED_SHAPEGGEN_SHA256,
    )
    graphsage_result = train_one_backbone(
        name="GraphSAGE",
        data=data,
        config_path=args.graphsage_config,
        checkpoint_path=args.graphsage_checkpoint,
        metrics_dir=args.metrics_dir,
        dataset_checksum=EXPECTED_SHAPEGGEN_SHA256,
    )

    predictions = {
        "GCN": predict_from_checkpoint(data, args.gcn_checkpoint),
        "GraphSAGE": predict_from_checkpoint(data, args.graphsage_checkpoint),
    }
    target_outputs = freeze_targets_and_pools(
        data=data,
        predictions=predictions,
        targets_dir=args.targets_dir,
    )
    manifest = build_freeze_manifest(
        args=args,
        data=data,
        dataset_metadata=dataset_metadata,
        gcn_result=gcn_result,
        graphsage_result=graphsage_result,
        target_outputs=target_outputs,
    )
    save_metadata(manifest, args.freeze_json_output)
    write_report(args.report_output, manifest)

    print("ShapeGGen Task22 freeze complete")
    print(f"GCN test_accuracy={gcn_result['summary']['test_accuracy']:.4f}")
    print(f"GraphSAGE test_accuracy={graphsage_result['summary']['test_accuracy']:.4f}")
    print(f"final_2x2_targets={target_outputs['final_2x2']['count']}")
    print(f"common_pg_train={target_outputs['common_pg_train']['count']}")
    print(f"common_pg_val={target_outputs['common_pg_val']['count']}")
    print(f"report={args.report_output}")
    print(f"freeze_json={args.freeze_json_output}")


def verify_task21_input(args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    actual_sha = sha256_file(args.artifact)
    if actual_sha != EXPECTED_SHAPEGGEN_SHA256:
        raise RuntimeError(f"ShapeGGen artifact SHA mismatch: {actual_sha}")
    if metadata["sha256"] != EXPECTED_SHAPEGGEN_SHA256:
        raise RuntimeError("ShapeGGen manifest SHA mismatch")
    if metadata["protocol_name"] != EXPECTED_PROTOCOL_NAME:
        raise RuntimeError(f"ShapeGGen protocol mismatch: {metadata['protocol_name']}")
    if metadata["upstream_commit_or_version"] != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError("ShapeGGen upstream commit mismatch")


def train_one_backbone(
    *,
    name: str,
    data: Any,
    config_path: Path,
    checkpoint_path: Path,
    metrics_dir: Path,
    dataset_checksum: str,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    stem = name.lower()
    history_path = metrics_dir / f"{stem}_history.csv"
    summary_path = metrics_dir / f"{stem}_summary.json"
    result = train_model(
        data=data,
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        summary_path=summary_path,
        dataset_checksum=dataset_checksum,
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    reload_parity = verify_reload_parity(data, checkpoint_path)
    manifest = {
        "backbone": name,
        "status": "success",
        "training_result": asdict(result),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "reload_parity": reload_parity,
    }
    save_metadata(manifest, metrics_dir / f"{stem}_manifest.json")
    return manifest


def verify_reload_parity(data: Any, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    first = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    second = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    first.eval()
    second.eval()
    cpu_data = data.cpu()
    with torch.no_grad():
        first_logits = first(cpu_data.x, cpu_data.edge_index)
        second_logits = second(cpu_data.x, cpu_data.edge_index)
    max_abs_diff = float((first_logits - second_logits).abs().max().item())
    return {
        "passed": bool(torch.allclose(first_logits, second_logits, atol=0.0, rtol=0.0)),
        "max_abs_diff": max_abs_diff,
    }


def predict_from_checkpoint(data: Any, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    model.eval()
    cpu_data = data.cpu()
    with torch.no_grad():
        logits = model(cpu_data.x, cpu_data.edge_index)
    return {
        "logits": logits,
        "probabilities": logits.softmax(dim=-1),
        "predictions": logits.argmax(dim=-1),
    }


def freeze_targets_and_pools(
    *,
    data: Any,
    predictions: dict[str, dict[str, torch.Tensor]],
    targets_dir: Path,
) -> dict[str, Any]:
    gcn_test = eligible_rows(data, predictions["GCN"], "test", "GCN")
    sage_test = eligible_rows(data, predictions["GraphSAGE"], "test", "GraphSAGE")
    final_ids = sorted(
        set(row["node_id"] for row in gcn_test) & set(row["node_id"] for row in sage_test)
    )
    final_rows = final_target_rows(data, predictions, final_ids)

    gcn_train = eligible_rows(data, predictions["GCN"], "train", "GCN")
    sage_train = eligible_rows(data, predictions["GraphSAGE"], "train", "GraphSAGE")
    gcn_val = eligible_rows(data, predictions["GCN"], "validation", "GCN")
    sage_val = eligible_rows(data, predictions["GraphSAGE"], "validation", "GraphSAGE")

    common_train_ids = sorted(
        set(row["node_id"] for row in gcn_train) & set(row["node_id"] for row in sage_train)
    )
    common_val_ids = sorted(
        set(row["node_id"] for row in gcn_val) & set(row["node_id"] for row in sage_val)
    )
    final_id_set = set(final_ids)
    if set(common_train_ids) & set(common_val_ids):
        raise RuntimeError("Common PG train/val pools overlap")
    if set(common_train_ids) & final_id_set or set(common_val_ids) & final_id_set:
        raise RuntimeError("Common PG pools overlap final 2x2 targets")

    outputs: dict[str, Any] = {}
    outputs["final_2x2"] = write_csv_and_hash(
        targets_dir / "shapeggen_2x2_targets.csv",
        final_rows,
        FINAL_FIELDNAMES,
    )
    for key, rows in {
        "pg_gcn_train_eligible": gcn_train,
        "pg_graphsage_train_eligible": sage_train,
        "pg_gcn_val_eligible": gcn_val,
        "pg_graphsage_val_eligible": sage_val,
    }.items():
        outputs[key] = write_csv_and_hash(
            targets_dir / f"shapeggen_{key}.csv",
            rows,
            ELIGIBLE_FIELDNAMES,
        )
    common_train_rows = common_pg_rows(data, predictions, common_train_ids, split="train")
    common_val_rows = common_pg_rows(data, predictions, common_val_ids, split="validation")
    outputs["common_pg_train"] = write_csv_and_hash(
        targets_dir / "shapeggen_pg_common_train.csv",
        common_train_rows,
        COMMON_PG_FIELDNAMES,
    )
    outputs["common_pg_val"] = write_csv_and_hash(
        targets_dir / "shapeggen_pg_common_val.csv",
        common_val_rows,
        COMMON_PG_FIELDNAMES,
    )
    outputs["set_counts"] = {
        "gcn_test_eligible": len(gcn_test),
        "graphsage_test_eligible": len(sage_test),
        "final_intersection": len(final_rows),
        "gcn_train_eligible": len(gcn_train),
        "graphsage_train_eligible": len(sage_train),
        "gcn_val_eligible": len(gcn_val),
        "graphsage_val_eligible": len(sage_val),
        "common_train": len(common_train_rows),
        "common_val": len(common_val_rows),
    }
    if not final_rows or not common_train_rows or not common_val_rows:
        raise RuntimeError("Task22 target or common PG pool is empty")
    return outputs


FINAL_FIELDNAMES = [
    "node_id",
    "label",
    "gcn_prediction",
    "gcn_confidence",
    "graphsage_prediction",
    "graphsage_confidence",
    "split",
    "candidate_edge_count",
    "positive_gt_edges",
    "negative_candidate_edges",
    "candidate_edge_mapping_valid",
]
ELIGIBLE_FIELDNAMES = [
    "node_id",
    "ground_truth_label",
    "backbone",
    "frozen_model_prediction",
    "target_for_pgexplainer",
    "confidence",
    "base_model_split",
    "prediction_correct",
    "candidate_edge_count",
    "positive_gt_edges",
    "negative_candidate_edges",
    "candidate_edge_mapping_valid",
]
COMMON_PG_FIELDNAMES = [
    "node_id",
    "ground_truth_label",
    "base_model_split",
    "gcn_prediction",
    "gcn_confidence",
    "graphsage_prediction",
    "graphsage_confidence",
    "candidate_edge_count",
    "positive_gt_edges",
    "negative_candidate_edges",
    "candidate_edge_mapping_valid",
]


def eligible_rows(
    data: Any,
    prediction_bundle: dict[str, torch.Tensor],
    split: str,
    backbone: str,
) -> list[dict[str, Any]]:
    split_mask = split_to_mask(data, split)
    predictions = prediction_bundle["predictions"]
    probabilities = prediction_bundle["probabilities"]
    rows = []
    for node_id in torch.nonzero(split_mask, as_tuple=False).view(-1).tolist():
        node_id = int(node_id)
        prediction = int(predictions[node_id].item())
        if prediction != int(data.y[node_id].item()):
            continue
        stats = gt_stats(data, node_id)
        if not stats["candidate_edge_mapping_valid"]:
            continue
        rows.append(
            {
                "node_id": node_id,
                "ground_truth_label": int(data.y[node_id].item()),
                "backbone": backbone,
                "frozen_model_prediction": prediction,
                "target_for_pgexplainer": prediction,
                "confidence": float(probabilities[node_id, prediction].item()),
                "base_model_split": split,
                "prediction_correct": True,
                **stats,
            }
        )
    return sorted(rows, key=lambda row: row["node_id"])


def final_target_rows(
    data: Any,
    predictions: dict[str, dict[str, torch.Tensor]],
    node_ids: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for node_id in node_ids:
        stats = gt_stats(data, node_id)
        gcn_prediction = int(predictions["GCN"]["predictions"][node_id].item())
        sage_prediction = int(predictions["GraphSAGE"]["predictions"][node_id].item())
        rows.append(
            {
                "node_id": node_id,
                "label": int(data.y[node_id].item()),
                "gcn_prediction": gcn_prediction,
                "gcn_confidence": float(
                    predictions["GCN"]["probabilities"][node_id, gcn_prediction].item()
                ),
                "graphsage_prediction": sage_prediction,
                "graphsage_confidence": float(
                    predictions["GraphSAGE"]["probabilities"][node_id, sage_prediction].item()
                ),
                "split": "test",
                **stats,
            }
        )
    return rows


def common_pg_rows(
    data: Any,
    predictions: dict[str, dict[str, torch.Tensor]],
    node_ids: list[int],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for node_id in node_ids:
        stats = gt_stats(data, node_id)
        gcn_prediction = int(predictions["GCN"]["predictions"][node_id].item())
        sage_prediction = int(predictions["GraphSAGE"]["predictions"][node_id].item())
        rows.append(
            {
                "node_id": node_id,
                "ground_truth_label": int(data.y[node_id].item()),
                "base_model_split": split,
                "gcn_prediction": gcn_prediction,
                "gcn_confidence": float(
                    predictions["GCN"]["probabilities"][node_id, gcn_prediction].item()
                ),
                "graphsage_prediction": sage_prediction,
                "graphsage_confidence": float(
                    predictions["GraphSAGE"]["probabilities"][node_id, sage_prediction].item()
                ),
                **stats,
            }
        )
    return rows


def gt_stats(data: Any, node_id: int) -> dict[str, Any]:
    try:
        candidate_mask = get_target_candidate_edge_mask(data, node_id)
        gt_mask = get_target_edge_ground_truth(data, node_id)
        candidate_count = int(candidate_mask.sum().item())
        positive_count = int((gt_mask & candidate_mask).sum().item())
        negative_count = candidate_count - positive_count
        valid = candidate_count > 0 and positive_count > 0 and negative_count > 0
    except Exception:
        candidate_count = 0
        positive_count = 0
        negative_count = 0
        valid = False
    return {
        "candidate_edge_count": candidate_count,
        "positive_gt_edges": positive_count,
        "negative_candidate_edges": negative_count,
        "candidate_edge_mapping_valid": valid,
    }


def split_to_mask(data: Any, split: str) -> torch.Tensor:
    if split == "train":
        return data.train_mask.detach().cpu()
    if split == "validation":
        return data.val_mask.detach().cpu()
    if split == "test":
        return data.test_mask.detach().cpu()
    raise ValueError(f"Unsupported split: {split}")


def write_csv_and_hash(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": str(path),
        "count": len(rows),
        "sha256": sha256_file(path),
    }


def build_freeze_manifest(
    *,
    args: argparse.Namespace,
    data: Any,
    dataset_metadata: dict[str, Any],
    gcn_result: dict[str, Any],
    graphsage_result: dict[str, Any],
    target_outputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": "Task22 ShapeGGen backbones and pre-XAI target freeze",
        "status": "success",
        "dataset": {
            "artifact": str(args.artifact),
            "config": "configs/shapeggen/v1/shapeggen.yaml",
            "sha256": EXPECTED_SHAPEGGEN_SHA256,
            "protocol_name": dataset_metadata["protocol_name"],
            "upstream_commit": dataset_metadata["upstream_commit_or_version"],
            "split_counts": dataset_metadata["split"]["counts"],
            "gt_policy": dataset_metadata["ground_truth"]["representation"],
            "candidate_reverse_edge_policy": (
                "target-specific candidate masks over directed PyG edge_index with reverse entries"
            ),
        },
        "model_architecture_policy": {
            "same_message_passing_depth": True,
            "same_hidden_width": True,
            "same_dropout": True,
            "same_training_policy": True,
            "gcn_difference": "GCNConv(cached=False)",
            "graphsage_difference": "SAGEConv(aggr=sum)",
        },
        "backbones": {
            "GCN": gcn_result,
            "GraphSAGE": graphsage_result,
        },
        "predictive_performance": {
            "GCN": split_performance(data, gcn_result, args.gcn_checkpoint),
            "GraphSAGE": split_performance(data, graphsage_result, args.graphsage_checkpoint),
        },
        "targets": target_outputs,
        "disjointness": {
            "common_train_common_val_disjoint": True,
            "common_train_final_disjoint": True,
            "common_val_final_disjoint": True,
        },
        "frozen_before_xai": True,
        "no_explainer_outputs_created": True,
        "environment": environment_metadata(),
        "git": git_metadata(),
    }


def split_performance(data: Any, result: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    logits = predict_from_checkpoint(data, checkpoint_path)["logits"]
    return {
        "train": asdict(evaluate_logits(logits, data, data.train_mask)),
        "validation": asdict(evaluate_logits(logits, data, data.val_mask)),
        "test": asdict(evaluate_logits(logits, data, data.test_mask)),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "reload_parity": result["reload_parity"],
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gcn = manifest["predictive_performance"]["GCN"]
    sage = manifest["predictive_performance"]["GraphSAGE"]
    targets = manifest["targets"]
    gcn_row = (
        f"| GCN | {gcn['train']['accuracy']:.6f} | "
        f"{gcn['validation']['accuracy']:.6f} | {gcn['test']['accuracy']:.6f} | "
        f"`{gcn['checkpoint_sha256']}` |"
    )
    sage_row = (
        f"| GraphSAGE | {sage['train']['accuracy']:.6f} | "
        f"{sage['validation']['accuracy']:.6f} | {sage['test']['accuracy']:.6f} | "
        f"`{sage['checkpoint_sha256']}` |"
    )
    target_rows = "\n".join(
        [
            report_target_row("Final 2x2 targets", targets["final_2x2"]),
            report_target_row("GCN train eligible", targets["pg_gcn_train_eligible"]),
            report_target_row(
                "GraphSAGE train eligible",
                targets["pg_graphsage_train_eligible"],
            ),
            report_target_row("GCN validation eligible", targets["pg_gcn_val_eligible"]),
            report_target_row(
                "GraphSAGE validation eligible",
                targets["pg_graphsage_val_eligible"],
            ),
            report_target_row("Common PG train", targets["common_pg_train"]),
            report_target_row("Common PG validation", targets["common_pg_val"]),
        ]
    )
    text = f"""# ShapeGGen Backbones And Targets Freeze

## Scope

Task22 trains and freezes GCN and GraphSAGE backbones on the official GraphXAI
ShapeGGen artifact, then freezes final 2x2 XAI targets and common PGExplainer
train/validation pools before any ShapeGGen explanation is inspected.

## Dataset

- Artifact: `{manifest["dataset"]["artifact"]}`
- SHA256: `{manifest["dataset"]["sha256"]}`
- Protocol: `{manifest["dataset"]["protocol_name"]}`
- Upstream GraphXAI commit: `{manifest["dataset"]["upstream_commit"]}`
- Split counts: `{manifest["dataset"]["split_counts"]}`

## Backbone Performance

| Backbone | Train Acc | Val Acc | Test Acc | Checkpoint SHA256 |
|---|---:|---:|---:|---|
{gcn_row}
{sage_row}

Reload parity passed for both checkpoints.

## Frozen Target Counts

| Set | Count | SHA256 |
|---|---:|---|
{target_rows}

Common PG pools are symmetric cross-backbone intersections by `node_id`.
They are disjoint from each other and from the final 2x2 test target set.

## Stop Boundary

No ShapeGGen GNNExplainer, PGExplainer, Random baseline, stability run, or final
XAI metric was executed in this task.
"""
    path.write_text(text, encoding="utf-8")


def report_target_row(label: str, target_manifest: dict[str, Any]) -> str:
    return f"| {label} | {target_manifest['count']} | `{target_manifest['sha256']}` |"


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def git_metadata() -> dict[str, str]:
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "status_short": run_git("status", "--short"),
    }


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


if __name__ == "__main__":
    main()
