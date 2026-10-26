from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/extension/ba_house_scale/phase2_backbones"
BA1 = ROOT / "docs/extension/ba_house_scale/phase1_generation"
SCALES = {
    "ba1000_ext": "7680fbaba95bcfecb96d5f8de0d4935b469e16d31be75d803ed583e5ca8e5b9a",
    "ba1500_ext": "a4d227ac3b56cf40966c19a5afeef8edbd6a2ad65a9d4a36e34ae5a3dd135b95",
}
ORDER = [(s, b) for s in SCALES for b in ("gcn", "graphsage")]
CHECKS: list[dict[str, Any]] = []


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


def check(name: str, passed: bool, detail: Any = None) -> None:
    if not passed:
        raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_PHASE_INVALID: " + name + "; " + str(detail))
    CHECKS.append({"check": name, "status": "PASS", "detail": detail})


def rehash(before: dict) -> list[dict]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda r: record(ROOT / r["path"]), before["files"]))
    if rows != before["files"]:
        changed = [(a, b) for a, b in zip(before["files"], rows, strict=True) if a != b]
        print(json.dumps(changed[:10], indent=2))
        raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_INPUT_PRESERVATION_FAILURE")
    return rows


class IndependentBackbone(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.dropout = 0.2
        channels = [10, 32, 32, 4]
        self.convs = nn.ModuleList(
            [
                GCNConv(a, b, cached=False)
                if kind == "gcn"
                else SAGEConv(a, b, aggr="sum", root_weight=True, normalize=False)
                for a, b in zip(channels[:-1], channels[1:], strict=True)
            ]
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        for index, conv in enumerate(self.convs):
            x = conv(x, edges)
            if index < 2:
                x = nn.functional.relu(x)
                x = nn.functional.dropout(x, p=0.2, training=self.training)
        return x


def expected_config(kind: str) -> dict:
    return {
        "model": {
            "type": "GCN" if kind == "gcn" else "GraphSAGE",
            "convolution": "PyG GCNConv" if kind == "gcn" else "PyG SAGEConv",
            "hidden_channels": 32,
            "num_layers": 3,
            "dropout": 0.2,
            "seed": 42,
            **({"cached": False} if kind == "gcn" else {"aggr": "sum"}),
        },
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0005,
            "max_epochs": 300,
            "patience": 60,
            "device": "auto",
            "class_weighting": True,
        },
        "checkpoint": {"monitor": "val_accuracy", "mode": "max"},
    }


def recompute(logits: torch.Tensor, data: Any) -> tuple[dict, torch.Tensor]:
    pred = logits.argmax(-1)
    metrics = {}
    for split in ("train", "val", "test"):
        mask = getattr(data, split + "_mask")
        motif = mask & (data.y > 0)
        total, correct = int(mask.sum()), int((pred[mask] == data.y[mask]).sum())
        motif_total = int(motif.sum())
        motif_correct = int((pred[motif] == data.y[motif]).sum())
        metrics[split] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "motif_total": motif_total,
            "motif_correct": motif_correct,
            "motif_accuracy": motif_correct / motif_total,
            "loss": float(nn.functional.cross_entropy(logits[mask], data.y[mask])),
        }
    metrics["test_per_class"] = []
    for label in range(4):
        mask = data.test_mask & (data.y == label)
        support, correct = int(mask.sum()), int((pred[mask] == data.y[mask]).sum())
        metrics["test_per_class"].append(
            {"class": label, "support": support, "correct": correct, "accuracy": correct / support}
        )
    return metrics, pred.cpu()


def verify_history(path: Path, summary: dict, checkpoint: dict, tag: str) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    check(
        tag + "_actual_epochs",
        len(history) == summary["actual_epochs"] == checkpoint["actual_epochs"]
        and 0 < len(history) <= 300,
    )
    check(
        tag + "_history_contiguous",
        [int(r["epoch"]) for r in history] == list(range(1, len(history) + 1)),
    )
    best, best_epoch, stale = -1.0, 0, 0
    for index, row in enumerate(history, start=1):
        values = [
            float(row[k])
            for k in (
                "optimization_loss_weighted",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "monitor_value",
            )
        ]
        check(tag + "_finite_history_" + str(index), all(math.isfinite(v) for v in values))
        value = float(row["val_accuracy"])
        check(
            tag + "_monitor_" + str(index),
            row["monitor_metric"] == "val_accuracy"
            and float(row["monitor_value"]) == value
            and 0 <= value <= 1,
        )
        improved = value > best
        check(tag + "_strict_checkpoint_" + str(index), row["strict_improvement"] == str(improved))
        if improved:
            best, best_epoch, stale = value, index, 0
        else:
            stale += 1
        check(tag + "_no_premature_continuation_" + str(index), stale < 60 or index == len(history))
    check(tag + "_early_stopping", len(history) == 300 or stale == 60)
    check(
        tag + "_first_max_checkpoint",
        best_epoch == summary["best_epoch"] == checkpoint["best_epoch"]
        and best == summary["best_validation_accuracy"] == checkpoint["best_validation_metric"],
    )
    check(
        tag + "_best_train_receipt",
        float(history[best_epoch - 1]["train_accuracy"]) == summary["metrics"]["train"]["accuracy"],
    )


def main() -> None:
    receipt_path = DOC / "04_verifier_receipt.json"
    after_path = DOC / "protected_inputs_after.json"
    collisions = [record(p) for p in (receipt_path, after_path) if p.exists()]
    if collisions:
        print(json.dumps(collisions, indent=2))
        raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_OUTPUT_ALREADY_EXISTS")
    torch.set_num_threads(1)
    started = datetime.now(UTC).isoformat()
    before = read(DOC / "protected_inputs_before.json")
    rehash(before)
    protected = {r["path"]: r for r in before["files"]}
    prior = read(BA1 / "protected_baseline_after.json")
    check(
        "frozen_baseline_scope_complete", all(protected.get(r["path"]) == r for r in prior["files"])
    )
    ba1_master_path = (
        ROOT / "artifacts/extensions/ba_house_scale/ba_house_scale_dataset_manifest.json"
    )
    ba1_master = read(ba1_master_path)
    ba1_records = (
        ba1_master["file_hashes"]
        + ba1_master["source_hashes"]
        + [record(ba1_master_path), record(BA1 / "04_verifier_receipt.json")]
    )
    check(
        "all_28_BA1_inputs_protected",
        len(ba1_records) == 28
        and all(
            protected.get(r["path"]) == {k: r[k] for k in ("path", "bytes", "sha256")}
            for r in ba1_records
        ),
    )
    check(
        "BA500_checkpoint_firewall",
        all(
            p in protected
            for p in (
                "artifacts/checkpoints/ba_house_gcn_best.pt",
                "artifacts/checkpoints/ba_house_graphsage_best.pt",
            )
        ),
    )
    run_path = DOC / "run_manifest.json"
    run = read(run_path)
    check(
        "exact_four_training_order",
        run["order"] == [list(p) for p in ORDER]
        and [(c["scale"], c["backbone"]) for c in run["cells"]] == ORDER,
    )
    for row in run["source_hashes"] + run["output_hashes"]:
        check("frozen_output_hash_" + row["path"], record(ROOT / row["path"]) == row)
    predictions = {scale: {} for scale in SCALES}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verified = []
    for cell in run["cells"]:
        scale, kind = cell["scale"], cell["backbone"]
        tag = scale + "_" + kind
        dataset_path = ROOT / cell["dataset"]["path"]
        if sha(dataset_path) != SCALES[scale]:
            raise RuntimeError("BA_HOUSE_SCALE_BACKBONE_INPUT_HASH_MISMATCH")
        check(tag + "_dataset_hash", cell["dataset"]["sha256"] == SCALES[scale])
        data = torch.load(dataset_path, map_location="cpu", weights_only=False).to(device)
        checkpoint_path = ROOT / cell["checkpoint"]["path"]
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        summary = read(ROOT / cell["summary"]["path"])
        manifest = read(ROOT / cell["manifest_file"]["path"])
        check(
            tag + "_manifest_parity",
            {k: v for k, v in cell.items() if k != "manifest_file"} == manifest
            and all(summary[k] == manifest[k] for k in summary),
        )
        config = expected_config(kind)
        check(
            tag + "_architecture_and_protocol_config",
            checkpoint["resolved_config"] == summary["resolved_config"] == config
            and checkpoint["seed"] == summary["seed"] == 42,
        )
        check(tag + "_checkpoint_dataset_binding", checkpoint["dataset_checksum"] == SCALES[scale])
        counts = torch.bincount(data.y[data.train_mask], minlength=4).double().cpu()
        weights = float(data.train_mask.sum()) / (4 * counts)
        expected = torch.tensor([0.4166666666666667, 1.5625, 1.5625, 3.125], dtype=torch.float64)
        if not torch.allclose(weights, expected, atol=1e-12, rtol=1e-12):
            raise RuntimeError("BA_HOUSE_SCALE_CLASS_WEIGHT_MISMATCH")
        check(
            tag + "_recorded_class_weights",
            torch.allclose(
                weights, checkpoint["class_weights"].double().cpu(), atol=1e-7, rtol=1e-7
            )
            and torch.allclose(
                weights,
                torch.tensor(summary["class_weights"], dtype=torch.float64),
                atol=1e-7,
                rtol=1e-7,
            )
            and counts.tolist() == summary["train_class_counts"],
        )
        model = IndependentBackbone(kind).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        check(
            tag + "_finite_checkpoint",
            all(torch.isfinite(t).all().item() for t in checkpoint["model_state_dict"].values()),
        )
        state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            metrics, pred = recompute(logits, data)
        check(tag + "_finite_logits", bool(torch.isfinite(logits).all()))
        check(
            tag + "_evaluation_no_parameter_mutation",
            all(torch.equal(v, model.state_dict()[k]) for k, v in state_before.items()),
        )
        for split in ("train", "val", "test"):
            for field, value in metrics[split].items():
                actual = summary["metrics"][split][field]
                equal = (
                    math.isclose(actual, value, rel_tol=1e-6, abs_tol=1e-6)
                    if field == "loss"
                    else actual == value
                )
                check(tag + "_" + split + "_" + field, equal, value)
        check(
            tag + "_per_class_test",
            metrics["test_per_class"] == summary["metrics"]["test_per_class"],
        )
        check(
            tag + "_validation_matches_best",
            metrics["val"]["accuracy"] == checkpoint["best_validation_metric"],
        )
        verify_history(ROOT / cell["history"]["path"], summary, checkpoint, tag)
        predictions[scale][kind] = pred
        verified.append(
            {
                "cell": tag,
                "checkpoint": record(checkpoint_path),
                "metrics": metrics,
                "best_epoch": checkpoint["best_epoch"],
                "actual_epochs": checkpoint["actual_epochs"],
                "status": summary["status"],
            }
        )
        print(tag + " READONLY_PREDICTIVE_VERIFICATION_PASS", flush=True)
    structural = read(BA1 / "03_structural_clean_pool.json")["records"]
    independently_counted = {}
    for scale in SCALES:
        data = torch.load(
            ROOT / f"artifacts/extensions/ba_house_scale/{scale}/{scale}.pt",
            map_location="cpu",
            weights_only=False,
        )
        clean = torch.zeros(data.num_nodes, dtype=torch.bool)
        rows = [r for r in structural if r["dataset_variant"] == scale.upper()]
        check(scale + "_complete_frozen_structural_metadata", len(rows) == int((data.y > 0).sum()))
        for row in rows:
            if row["structurally_clean"]:
                clean[row["node_id"]] = True
        common = torch.stack(
            [predictions[scale][kind] == data.y for kind in ("gcn", "graphsage")]
        ).all(0)
        motif = data.y > 0
        splits = []
        for split in ("train", "val", "test"):
            mask = getattr(data, split + "_mask")
            splits.append(
                {
                    "split": split,
                    "common_correct": int(common[mask].sum()),
                    "common_correct_motif": int((mask & common & motif).sum()),
                    "common_correct_motif_structurally_clean": int(
                        (mask & common & motif & clean).sum()
                    ),
                }
            )
        mask = data.test_mask & common & motif & clean
        independently_counted[scale] = {
            "splits": splits,
            "test_class_composition": {
                str(c): int((mask & (data.y == c)).sum()) for c in (1, 2, 3)
            },
            "test_total": int(mask.sum()),
        }
    ready = read(DOC / "03_common_correct_readiness.json")
    check(
        "common_correct_and_clean_counts_reproduced",
        ready["scales"] == run["readiness"]["scales"] == independently_counted,
    )
    readiness_verdict = (
        "BA_HOUSE_SCALE_COMMON_CORRECT_SELECTION_READY"
        if all(
            r["common_correct_motif_structurally_clean"] > 0
            for value in independently_counted.values()
            for r in value["splits"]
        )
        else "BA_HOUSE_SCALE_COMMON_CORRECT_SELECTION_NOT_READY"
    )
    check("readiness_verdict", ready["verdict"] == run["readiness"]["verdict"] == readiness_verdict)
    check("counts_only_firewall", ready["counts_only"] and ready["no_target_ids_or_pg_pools"])
    training_report = read(DOC / "01_backbone_training_report.json")
    comparison = read(DOC / "02_predictive_comparison.json")
    check(
        "report_cells_match_frozen_manifest",
        training_report["cells"] == comparison["cells"] == run["cells"],
    )
    tables = training_report["tables"] + comparison["tables"] + ready["tables"]
    check(
        "six_required_tables",
        all(any(t["name"].startswith("BA-B" + str(n)) for t in tables) for n in range(1, 7)),
    )
    indexed_tables = {tab["name"].split()[0]: tab for tab in tables}
    cells = run["cells"]
    expected_rows = {
        "BA-B1": [
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
        "BA-B2": [
            [c["scale"], c["backbone"], c["actual_epochs"], c["best_epoch"]]
            + [c["metrics"][s]["accuracy"] for s in ("train", "val", "test")]
            for c in cells
        ],
        "BA-B3": [
            [c["scale"], c["backbone"]]
            + [c["metrics"][s]["motif_accuracy"] for s in ("train", "val", "test")]
            for c in cells
        ],
        "BA-B4": [
            [c["scale"], c["backbone"], r["class"], r["support"], r["correct"], r["accuracy"]]
            for c in cells
            for r in c["metrics"]["test_per_class"]
        ],
        "BA-B5": [
            [
                scale,
                r["split"],
                r["common_correct"],
                r["common_correct_motif"],
                r["common_correct_motif_structurally_clean"],
            ]
            for scale, value in independently_counted.items()
            for r in value["splits"]
        ],
        "BA-B6": [
            [scale]
            + [value["test_class_composition"][str(c)] for c in (1, 2, 3)]
            + [value["test_total"]]
            for scale, value in independently_counted.items()
        ],
    }
    for name, rows in expected_rows.items():
        check("table_values_" + name, indexed_tables[name]["rows"] == rows)
    # This after snapshot follows all model/metric verification, on the identical input list.
    after_rows = rehash(before)
    after = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(after_rows),
        "files": after_rows,
        "exact_equality": True,
        "verdict": "BA_HOUSE_SCALE_BACKBONE_INPUT_PRESERVATION_PASS",
    }
    with after_path.open("x", encoding="utf-8") as stream:
        json.dump(after, stream, indent=2)
        stream.write("\n")
    check(
        "protected_inputs_before_after_exact_equality", before["files"] == read(after_path)["files"]
    )
    for row in run["source_hashes"] + run["output_hashes"]:
        check("final_output_hash_" + row["path"], record(ROOT / row["path"]) == row)
    result = {
        "status": "BA_HOUSE_SCALE_BACKBONE_INDEPENDENT_VERIFIER_PASS",
        "phase_verdict": "BA_HOUSE_SCALE_BACKBONE_PHASE_COMPLETE",
        "readiness_verdict": readiness_verdict,
        "preservation_verdict": after["verdict"],
        "protected_file_count": len(after_rows),
        "pass_gates": {f"G{i}": "PASS" for i in range(1, 16)},
        "check_count": len(CHECKS),
        "checks": CHECKS,
        "cells": verified,
        "counts_only_readiness": independently_counted,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "independence": "Own backbone class, metric arithmetic, history replay, "
        "class-weight calculation and protected-input rehash; no training/helper imports",
        "evaluation_device": str(device),
        "training_performed": False,
        "command": {"argv": sys.argv, "executable": sys.executable, "cwd": str(ROOT)},
        "stdout": "\n".join(v["cell"] + " READONLY_PREDICTIVE_VERIFICATION_PASS" for v in verified)
        + "\nBA_HOUSE_SCALE_BACKBONE_INDEPENDENT_VERIFIER_PASS\n",
        "stderr": "",
        "run_manifest": record(run_path),
        "protected_inputs_after": record(after_path),
        "verifier_source": record(Path(__file__)),
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "phase_verdict",
                    "readiness_verdict",
                    "preservation_verdict",
                    "protected_file_count",
                    "check_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
