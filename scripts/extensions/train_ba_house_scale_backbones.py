from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch_geometric
import yaml

from gnnexplainer_benchmark.models import GCN, GraphSAGE
from gnnexplainer_benchmark.training.evaluation import evaluate_logits
from gnnexplainer_benchmark.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/extension/ba_house_scale/phase2_backbones"
ART = ROOT / "artifacts/extensions/ba_house_scale/backbones"
RESULT = ROOT / "results/extensions/ba_house_scale/backbones"
BA1 = ROOT / "docs/extension/ba_house_scale/phase1_generation"
SCALES = {
    "ba1000_ext": "7680fbaba95bcfecb96d5f8de0d4935b469e16d31be75d803ed583e5ca8e5b9a",
    "ba1500_ext": "a4d227ac3b56cf40966c19a5afeef8edbd6a2ad65a9d4a36e34ae5a3dd135b95",
}
ORDER = [(scale, kind) for scale in SCALES for kind in ("gcn", "graphsage")]
EXPECTED_WEIGHTS = torch.tensor([1 / 2.4, 1.5625, 1.5625, 3.125])
EVENTS: list[str] = []


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def emit(value: Any) -> None:
    text = json.dumps(value)
    EVENTS.append(text)
    print(text, flush=True)


def paths(scale: str, kind: str) -> dict[str, Path]:
    name = f"{scale}_{kind}_best"
    return {
        "checkpoint": ART / scale / kind / f"{name}.pt",
        "manifest": ART / scale / kind / f"{name}_manifest.json",
        "history": RESULT / scale / kind / "training_history.csv",
        "summary": RESULT / scale / kind / "summary.json",
    }


def all_outputs() -> list[Path]:
    outputs = [p for scale, kind in ORDER for p in paths(scale, kind).values()]
    outputs += [
        DOC / name
        for name in (
            "01_backbone_training_report.md",
            "01_backbone_training_report.json",
            "02_predictive_comparison.md",
            "02_predictive_comparison.json",
            "03_common_correct_readiness.md",
            "03_common_correct_readiness.json",
            "04_verifier_receipt.json",
            "protected_inputs_before.json",
            "protected_inputs_after.json",
            "command_receipt.json",
            "training_log.json",
            "run_manifest.json",
        )
    ]
    return outputs


def protect() -> dict[str, Any]:
    reference = read(BA1 / "protected_baseline_after.json")
    expected = {r["path"]: r for r in reference["files"]}
    master_path = ROOT / "artifacts/extensions/ba_house_scale/ba_house_scale_dataset_manifest.json"
    master = read(master_path)
    for row in master["file_hashes"] + master["source_hashes"]:
        expected[row["path"]] = {key: row[key] for key in ("path", "bytes", "sha256")}
    expected[master_path.relative_to(ROOT).as_posix()] = record(master_path)
    verifier = BA1 / "04_verifier_receipt.json"
    expected[verifier.relative_to(ROOT).as_posix()] = record(verifier)
    if read(verifier)["status"] != "BA_HOUSE_SCALE_INDEPENDENT_VERIFIER_PASS":
        raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_PHASE_INVALID: BA1 not verified")
    # Threaded file reads preserve the complete scope while avoiding serial I/O latency.
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(lambda p: record(ROOT / p), sorted(expected)))
    for row in actual:
        if row != expected[row["path"]]:
            emit({"expected": expected[row["path"]], "actual": row})
            raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_INPUT_PRESERVATION_FAILURE")
    before = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(actual),
        "scope": "Exact BA1 protected scope plus all 28 frozen BA1 extension files",
        "files": actual,
        "phase_BA1_status": read(verifier)["status"],
    }
    write(DOC / "protected_inputs_before.json", before)
    emit({"protected_input_hashing": "PASS", "files": len(actual)})
    return before


def resolved(kind: str) -> dict[str, Any]:
    source = ROOT / (
        "configs/model/gcn.yaml" if kind == "gcn" else "configs/model/graphsage_ba_house.yaml"
    )
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    wanted = {
        "type": "GCN" if kind == "gcn" else "GraphSAGE",
        "convolution": "PyG GCNConv" if kind == "gcn" else "PyG SAGEConv",
        "hidden_channels": 32,
        "num_layers": 3,
        "dropout": 0.2,
        "seed": 42,
        **({"cached": False} if kind == "gcn" else {"aggr": "sum"}),
    }
    if config != {
        "model": wanted,
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0005,
            "max_epochs": 300,
            "patience": 60,
            "device": "auto",
            "class_weighting": True,
        },
        "checkpoint": {"monitor": "val_accuracy", "mode": "max"},
    }:
        raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_PHASE_INVALID: frozen config mismatch")
    return config


def model_for(kind: str) -> torch.nn.Module:
    if kind == "gcn":
        return GCN(10, 32, 4, num_layers=3, dropout=0.2, cached=False)
    return GraphSAGE(10, 32, 4, num_layers=3, dropout=0.2, aggr="sum")


def architecture(model: torch.nn.Module, kind: str) -> dict[str, Any]:
    for conv, (inputs, outputs) in zip(model.convs, ((10, 32), (32, 32), (32, 4)), strict=True):
        assert (conv.in_channels, conv.out_channels) == (inputs, outputs)
        if kind == "gcn":
            assert not conv.cached
        else:
            assert conv.aggr == "sum" and conv.root_weight and not conv.normalize
    return {
        "layers": 3,
        "channels": [10, 32, 32, 4],
        "dropout": 0.2,
        "convolution": "GCNConv" if kind == "gcn" else "SAGEConv",
        "cached": False if kind == "gcn" else None,
        "aggregation": None if kind == "gcn" else "sum",
        "root_weight": None if kind == "gcn" else True,
        "normalize": None if kind == "gcn" else False,
        "activation": "ReLU after first two convolutions",
        "output": "raw logits",
    }


def predictive(logits: torch.Tensor, data: Any) -> dict[str, Any]:
    prediction = logits.argmax(-1)
    metrics = {}
    for split in ("train", "val", "test"):
        mask = getattr(data, split + "_mask")
        metric = asdict(evaluate_logits(logits, data, mask))
        metric["motif_accuracy"] = metric["motif_correct"] / metric["motif_total"]
        metrics[split] = metric
    metrics["test_per_class"] = []
    for label in range(4):
        mask = data.test_mask & (data.y == label)
        support = int(mask.sum())
        correct = int((prediction[mask] == data.y[mask]).sum())
        metrics["test_per_class"].append(
            {"class": label, "support": support, "correct": correct, "accuracy": correct / support}
        )
    return metrics


def train_cell(scale: str, kind: str, environment: dict[str, Any]) -> tuple[dict, torch.Tensor]:
    dataset_path = ROOT / f"artifacts/extensions/ba_house_scale/{scale}/{scale}.pt"
    if sha(dataset_path) != SCALES[scale]:
        raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_INPUT_HASH_MISMATCH")
    config = resolved(kind)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False).to(device)
    model = model_for(kind).to(device)
    model_architecture = architecture(model, kind)
    counts = torch.bincount(data.y[data.train_mask], minlength=4).float()
    weights = data.train_mask.sum() / (4 * counts)
    if not torch.allclose(weights.cpu(), EXPECTED_WEIGHTS, atol=1e-7, rtol=1e-7):
        raise RuntimeError("BA_HOUSE_SCALE_CLASS_WEIGHT_MISMATCH")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    best_metric, best_epoch, stale = -1.0, 0, 0
    best_state, history = None, []
    emit({"training_started": scale, "backbone": kind, "device": str(device)})
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(
            logits[data.train_mask], data.y[data.train_mask], weight=weights
        )
        if not torch.isfinite(loss):
            raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_PHASE_INVALID: nonfinite loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            train = evaluate_logits(logits, data, data.train_mask)
            val = evaluate_logits(logits, data, data.val_mask)
        improved = val.accuracy > best_metric
        history.append(
            {
                "epoch": epoch,
                "optimization_loss_weighted": float(loss.detach()),
                "train_loss": train.loss,
                "train_accuracy": train.accuracy,
                "val_loss": val.loss,
                "val_accuracy": val.accuracy,
                "monitor_metric": "val_accuracy",
                "monitor_value": val.accuracy,
                "strict_improvement": improved,
            }
        )
        if improved:
            best_metric, best_epoch, stale = val.accuracy, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 60:
                break
    outputs = paths(scale, kind)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "resolved_config": config,
        "seed": 42,
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "dataset_checksum": SCALES[scale],
        "actual_epochs": len(history),
        "class_weights": weights.cpu(),
        "architecture": model_architecture,
    }
    with outputs["checkpoint"].open("xb") as stream:
        torch.save(checkpoint, stream)
    frozen = torch.load(outputs["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(frozen["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        metrics = predictive(logits, data)
    if metrics["val"]["accuracy"] != best_metric:
        raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_PHASE_INVALID: reload validation mismatch")
    with outputs["history"].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "scale": scale,
        "backbone": kind,
        "actual_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_metric,
        "class_weights": weights.cpu().tolist(),
        "train_class_counts": counts.cpu().tolist(),
        "metrics": metrics,
        "resolved_config": config,
        "architecture": model_architecture,
        "device": str(device),
        "seed": 42,
        "test_only_after_checkpoint_freeze": True,
        "history_losses": "Post-update eval CE unweighted, as frozen trainer; "
        "optimization_loss_weighted is train-only dropout-enabled weighted CE",
        "status": scale.upper().replace("_EXT", "") + "_" + kind.upper() + "_TRAINING_PASS",
    }
    write(outputs["summary"], summary)
    manifest = {
        **summary,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": record(dataset_path),
        "checkpoint": record(outputs["checkpoint"]),
        "history": record(outputs["history"]),
        "summary": record(outputs["summary"]),
        "environment": environment,
        "checkpoint_rule": "strict val_accuracy improvement; "
        "first occurrence wins ties; patience60; full-graph train-mask-only optimization",
    }
    write(outputs["manifest"], manifest)
    manifest["manifest_file"] = record(outputs["manifest"])
    emit(
        {
            "training_complete": scale,
            "backbone": kind,
            "epochs": len(history),
            "best_epoch": best_epoch,
            "val_accuracy": best_metric,
            "test_accuracy": metrics["test"]["accuracy"],
            "checkpoint": manifest["checkpoint"],
        }
    )
    return manifest, logits.argmax(-1).cpu()


def readiness(predictions: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    structural = read(BA1 / "03_structural_clean_pool.json")["records"]
    result = {}
    for scale in SCALES:
        data = torch.load(
            ROOT / f"artifacts/extensions/ba_house_scale/{scale}/{scale}.pt",
            map_location="cpu",
            weights_only=False,
        )
        clean = torch.zeros(data.num_nodes, dtype=torch.bool)
        for row in structural:
            if row["dataset_variant"] == scale.upper() and row["structurally_clean"]:
                clean[row["node_id"]] = True
        common = (predictions[scale]["gcn"] == data.y) & (predictions[scale]["graphsage"] == data.y)
        motif = data.y > 0
        rows = []
        for split in ("train", "val", "test"):
            mask = getattr(data, split + "_mask")
            rows.append(
                {
                    "split": split,
                    "common_correct": int((mask & common).sum()),
                    "common_correct_motif": int((mask & common & motif).sum()),
                    "common_correct_motif_structurally_clean": int(
                        (mask & common & motif & clean).sum()
                    ),
                }
            )
        test = data.test_mask & common & motif & clean
        result[scale] = {
            "splits": rows,
            "test_class_composition": {
                str(c): int((test & (data.y == c)).sum()) for c in (1, 2, 3)
            },
            "test_total": int(test.sum()),
        }
    return {
        "counts_only": True,
        "no_target_ids_or_pg_pools": True,
        "scales": result,
        "readiness_criterion": "Nonempty clean common-correct motif intersection in each "
        "split at both scales; no final quota or accuracy threshold is assumed",
        "verdict": "BA_HOUSE_SCALE_COMMON_CORRECT_SELECTION_READY"
        if all(
            r["common_correct_motif_structurally_clean"] > 0
            for value in result.values()
            for r in value["splits"]
        )
        else "BA_HOUSE_SCALE_COMMON_CORRECT_SELECTION_NOT_READY",
    }


def table(name: str, columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"name": name, "columns": columns, "rows": rows}


def markdown(tables: list[dict[str, Any]]) -> str:
    sections = [
        "# BA-House Scale Backbone Receipts",
        "Dataset/architecture/protocol fixed; "
        "counts-only readiness diagnostics. No targets, PG pools or XAI runs. "
        "Final verification is recorded in 04_verifier_receipt.json.",
    ]
    for tab in tables:
        lines = [
            "## " + tab["name"],
            "| " + " | ".join(tab["columns"]) + " |",
            "| " + " | ".join(["---"] * len(tab["columns"])) + " |",
        ]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in tab["rows"]]
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def reports(cells: list[dict], ready: dict) -> None:
    config = table(
        "BA-B1 Backbone Training Configuration",
        [
            "Scale",
            "Backbone",
            "Layers",
            "Hidden",
            "Dropout",
            "LR",
            "WeightDecay",
            "MaxEpochs",
            "Patience",
            "Seed",
            "CheckpointRule",
        ],
        [
            [
                c["scale"],
                c["backbone"],
                3,
                32,
                0.2,
                0.01,
                0.0005,
                300,
                60,
                42,
                "strict val_accuracy; first tie",
            ]
            for c in cells
        ],
    )
    performance = table(
        "BA-B2 Predictive Performance",
        ["Scale", "Backbone", "ActualEpochs", "BestEpoch", "TrainAcc", "ValAcc", "TestAcc"],
        [
            [c["scale"], c["backbone"], c["actual_epochs"], c["best_epoch"]]
            + [c["metrics"][s]["accuracy"] for s in ("train", "val", "test")]
            for c in cells
        ],
    )
    motif = table(
        "BA-B3 Motif Predictive Performance",
        ["Scale", "Backbone", "TrainMotifAcc", "ValMotifAcc", "TestMotifAcc"],
        [
            [c["scale"], c["backbone"]]
            + [c["metrics"][s]["motif_accuracy"] for s in ("train", "val", "test")]
            for c in cells
        ],
    )
    classes = table(
        "BA-B4 Test Per-Class Performance",
        ["Scale", "Backbone", "Class", "Support", "Correct", "Accuracy"],
        [
            [c["scale"], c["backbone"], r["class"], r["support"], r["correct"], r["accuracy"]]
            for c in cells
            for r in c["metrics"]["test_per_class"]
        ],
    )
    common = table(
        "BA-B5 Common-Correct Readiness",
        [
            "Scale",
            "Split",
            "CommonCorrect",
            "CommonCorrectMotif",
            "CommonCorrectMotifStructurallyClean",
        ],
        [
            [
                scale,
                r["split"],
                r["common_correct"],
                r["common_correct_motif"],
                r["common_correct_motif_structurally_clean"],
            ]
            for scale, value in ready["scales"].items()
            for r in value["splits"]
        ],
    )
    composition = table(
        "BA-B6 Test Clean Common-Correct Class Composition",
        ["Scale", "Class1", "Class2", "Class3", "Total"],
        [
            [scale]
            + [value["test_class_composition"][str(c)] for c in (1, 2, 3)]
            + [value["test_total"]]
            for scale, value in ready["scales"].items()
        ],
    )
    for stem, payload, tabs in (
        (
            "01_backbone_training_report",
            {"cells": cells, "tables": [config, performance]},
            [config, performance],
        ),
        (
            "02_predictive_comparison",
            {"cells": cells, "tables": [performance, motif, classes]},
            [performance, motif, classes],
        ),
        (
            "03_common_correct_readiness",
            {**ready, "tables": [common, composition]},
            [common, composition],
        ),
    ):
        write(DOC / (stem + ".json"), payload)
        with (DOC / (stem + ".md")).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(markdown(tabs))


def main() -> None:
    collisions = [record(p) for p in all_outputs() if p.exists()]
    if collisions:
        emit(collisions)
        raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_OUTPUT_ALREADY_EXISTS")
    for scale, checksum in SCALES.items():
        if sha(ROOT / f"artifacts/extensions/ba_house_scale/{scale}/{scale}.pt") != checksum:
            raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_INPUT_HASH_MISMATCH")
    for kind in ("gcn", "graphsage"):
        resolved(kind)
    protect()
    torch.set_num_threads(1)
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "pyg": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu_threads": 1,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_diff_status": subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True
        ).strip(),
        "determinism_policy": "Frozen seed_everything(42); no new algorithm flags",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    write(
        DOC / "command_receipt.json",
        {
            "argv": sys.argv,
            "executable": sys.executable,
            "cwd": str(ROOT),
            "training_order": ORDER,
            "environment": environment,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "BA500_training": False,
        },
    )
    cells, predictions = [], {scale: {} for scale in SCALES}
    for scale, kind in ORDER:
        cell, prediction = train_cell(scale, kind, environment)
        cells.append(cell)
        predictions[scale][kind] = prediction
    ready = readiness(predictions)
    reports(cells, ready)
    emit(
        {
            "training_finished": "four checkpoints frozen",
            "readiness": ready["verdict"],
            "verification": "Independent read-only verifier required",
        }
    )
    write(
        DOC / "training_log.json",
        {
            "stdout": "\n".join(EVENTS) + "\n",
            "stderr": "",
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "training_order": ORDER,
            "no_BA500_or_explainer_runs": True,
        },
    )
    sources = [Path(__file__), ROOT / "scripts/extensions/verify_ba_house_scale_backbones.py"]
    write(
        DOC / "run_manifest.json",
        {
            "order": ORDER,
            "cells": cells,
            "readiness": ready,
            "source_hashes": [record(p) for p in sources],
            "output_hashes": [record(p) for p in all_outputs() if p.exists()],
            "final_receipt": "04_verifier_receipt.json",
            "self_and_final_receipts_excluded": True,
        },
    )


if __name__ == "__main__":
    main()
