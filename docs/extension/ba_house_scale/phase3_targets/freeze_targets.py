from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch_geometric

from gnnexplainer_benchmark.training.checkpointing import build_model_from_checkpoint

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parent
ART = ROOT / "artifacts/extensions/ba_house_scale/targets"
BA1 = ROOT / "docs/extension/ba_house_scale/phase1_generation"
BA2 = ROOT / "docs/extension/ba_house_scale/phase2_backbones"
INPUTS = {
    "ba1000_ext": {
        "dataset": "7680fbaba95bcfecb96d5f8de0d4935b469e16d31be75d803ed583e5ca8e5b9a",
        "gcn": "812e123a2ce3c3ed0d9b04eaca78cbba8a4f72d2fa61abf09cf336dde01d75ef",
        "graphsage": "2c51117461c3eb2ff1cb29b291296f5202d114ba4ac4e8543754b6ceeeb607ca",
    },
    "ba1500_ext": {
        "dataset": "a4d227ac3b56cf40966c19a5afeef8edbd6a2ad65a9d4a36e34ae5a3dd135b95",
        "gcn": "be0ffcacc4031eb0dec6195120c608791082475a3d498c59bee1acbb89451bd7",
        "graphsage": "01dcc7c4818e34a91d5d19c36acbefec818b095c29adb13dff208cb4a2a1b56e",
    },
}
FILES = (
    "eligible_test_pool.json",
    "final_test_targets.json",
    "pg_train_pool.json",
    "pg_validation_pool.json",
    "target_manifest.json",
)
REPORTS = ("01_target_freeze_report", "02_pg_pool_report", "03_selection_determinism")
WATCH = (
    "artifacts/extensions/ba_house_scale",
    "results/extensions/ba_house_scale",
    "artifacts/checkpoints",
    "results/explanations",
    "results/metrics",
)
EVENTS: list[str] = []


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def emit(payload: Any) -> None:
    line = json.dumps(payload)
    EVENTS.append(line)
    print(line, flush=True)


def input_path(scale: str, kind: str) -> Path:
    if kind == "dataset":
        return ROOT / f"artifacts/extensions/ba_house_scale/{scale}/{scale}.pt"
    return (
        ROOT
        / f"artifacts/extensions/ba_house_scale/backbones/{scale}/{kind}/{scale}_{kind}_best.pt"
    )


def outputs() -> list[Path]:
    files = [ART / scale / name for scale in INPUTS for name in FILES]
    files += [DOC / (stem + suffix) for stem in REPORTS for suffix in (".md", ".json")]
    files += [
        DOC / name
        for name in (
            "protected_inputs_before.json",
            "protected_inputs_after.json",
            "04_verifier_receipt.json",
            "command_receipt.json",
            "selection_log.json",
        )
    ]
    return [*files, ART / "target_freeze_master_manifest.json"]


def inventory() -> list[str]:
    return sorted(
        {
            p.relative_to(ROOT).as_posix()
            for name in WATCH
            for p in (ROOT / name).rglob("*")
            if p.is_file()
        }
    )


def protect() -> None:
    previous = read(BA2 / "protected_inputs_after.json")
    expected = {r["path"]: r for r in previous["files"]}
    run = read(BA2 / "run_manifest.json")
    for row in run["source_hashes"] + run["output_hashes"]:
        expected[row["path"]] = row
    receipt = read(BA2 / "04_verifier_receipt.json")
    if receipt["status"] != "BA_HOUSE_SCALE_BACKBONE_INDEPENDENT_VERIFIER_PASS":
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_FREEZE_INVALID: BA2 not verified")
    for row in (
        receipt["run_manifest"],
        receipt["protected_inputs_after"],
        record(BA2 / "04_verifier_receipt.json"),
    ):
        expected[row["path"]] = row
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(lambda p: record(ROOT / p), sorted(expected)))
    for row in actual:
        if row != expected[row["path"]]:
            emit({"expected": expected[row["path"]], "actual": row})
            raise RuntimeError("BA_HOUSE_SCALE_TARGET_INPUT_PRESERVATION_FAILURE")
    write(
        DOC / "protected_inputs_before.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "file_count": len(actual),
            "scope": "Complete frozen baseline, BA1 and BA2 scopes, including source receipts",
            "files": actual,
            "artifact_inventory_before": inventory(),
            "inventory_roots": WATCH,
        },
    )
    emit({"protected_inputs": len(actual), "before_hashing": "PASS"})


def composition(ids: list[int], labels: torch.Tensor) -> dict[str, int]:
    return {str(c): sum(int(labels[node]) == c for node in ids) for c in (1, 2, 3)}


def selection(eligible: list[int], quota: int) -> tuple[list[int], list[int]]:
    ordered = sorted(eligible)
    generator = torch.Generator(device="cpu").manual_seed(42)
    randomized = [ordered[i] for i in torch.randperm(len(ordered), generator=generator).tolist()]
    return randomized, sorted(randomized[:quota])


def freeze_scale(scale: str, index: int, environment: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(input_path(scale, "dataset"), map_location="cpu", weights_only=False)
    inference_data = data.clone().to(device)
    predictions, confidences, prediction_hashes = {}, {}, {}
    for kind in ("gcn", "graphsage"):
        checkpoint = torch.load(input_path(scale, kind), map_location=device, weights_only=False)
        model = build_model_from_checkpoint(checkpoint, in_channels=10, out_channels=4).to(device)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            logits = model(inference_data.x, inference_data.edge_index)
        if not torch.isfinite(logits).all() or any(
            not torch.equal(v, model.state_dict()[k]) for k, v in state.items()
        ):
            raise RuntimeError("BA_HOUSE_SCALE_TARGET_FREEZE_INVALID: inference state")
        predictions[kind] = logits.argmax(-1).cpu()
        confidences[kind] = logits.softmax(-1)[:, 1].cpu()
        prediction_hashes[kind] = hashlib.sha256(predictions[kind].numpy().tobytes()).hexdigest()
    structural_rows = [
        r
        for r in read(BA1 / "03_structural_clean_pool.json")["records"]
        if r["dataset_variant"] == scale.upper()
    ]
    rows = {r["node_id"]: r for r in structural_rows}
    clean = torch.zeros(data.num_nodes, dtype=torch.bool)
    for node, row in rows.items():
        clean[node] = row["structurally_clean"]
    common = (predictions["gcn"] == data.y) & (predictions["graphsage"] == data.y)
    motif = (data.y > 0) & data.node_mask.bool() & (data.motif_instance_id > 0)
    combined = common & motif & clean
    pools = {
        split: torch.where(getattr(data, split + "_mask") & combined)[0].tolist()
        for split in ("train", "val", "test")
    }
    expected_counts = {
        "ba1000_ext": {"train": 94, "val": 32, "test": 32},
        "ba1500_ext": {"train": 141, "val": 48, "test": 48},
    }[scale]
    frozen_ready = read(BA2 / "03_common_correct_readiness.json")["scales"][scale]
    if {s: len(ids) for s, ids in pools.items()} != expected_counts or composition(
        pools["test"], data.y
    ) != {"1": expected_counts["test"], "2": 0, "3": 0}:
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_POOL_RECOMPUTE_MISMATCH")
    diagnostic_rows = [
        {
            "split": s,
            "common_correct": int((getattr(data, s + "_mask") & common).sum()),
            "common_correct_motif": int((getattr(data, s + "_mask") & common & motif).sum()),
            "common_correct_motif_structurally_clean": len(pools[s]),
        }
        for s in ("train", "val", "test")
    ]
    if diagnostic_rows != frozen_ready["splits"]:
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_POOL_RECOMPUTE_MISMATCH")
    eligible = [node for node in pools["test"] if int(data.y[node]) == 1]
    quota = {"ba1000_ext": 24, "ba1500_ext": 36}[scale]
    if len(eligible) != expected_counts["test"] or quota / len(eligible) != 0.75:
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_FREEZE_INVALID: quota")
    randomized, selected = selection(eligible, quota)

    def target_record(node: int) -> dict:
        row = rows[node]
        return {
            "node_id": node,
            "label": int(data.y[node]),
            "motif_instance_id": int(data.motif_instance_id[node]),
            "motif_node": bool(motif[node]),
            "test_mask": bool(data.test_mask[node]),
            "structurally_clean": row["structurally_clean"],
            "global_gt_directed_count": row["global_gt_directed_count"],
            "own_house_gt_directed_count": row["own_house_gt_directed_count"],
            "non_own_house_gt_directed_count": row["non_own_house_gt_directed_count"],
            "candidate_directed_edge_count": row["candidate_directed_edge_count"],
            "candidate_node_count": row["candidate_node_count"],
            "global_gt_prevalence": row["global_gt_prevalence"],
            "gcn_predicted_class": int(predictions["gcn"][node]),
            "graphsage_predicted_class": int(predictions["graphsage"][node]),
            "gcn_probability_class1": float(confidences["gcn"][node]),
            "graphsage_probability_class1": float(confidences["graphsage"][node]),
        }

    selected_records = [target_record(node) for node in selected]
    for row in selected_records:
        if not (
            row["test_mask"]
            and row["label"] == row["gcn_predicted_class"] == row["graphsage_predicted_class"] == 1
            and row["motif_node"]
            and row["structurally_clean"]
            and row["global_gt_directed_count"] == row["own_house_gt_directed_count"] == 12
            and row["non_own_house_gt_directed_count"] == 0
        ):
            raise RuntimeError("BA_HOUSE_SCALE_TARGET_FREEZE_INVALID: selected record")
    repeated = []
    for _ in range(2):
        new_eligible = torch.where(data.test_mask & common & motif & clean & (data.y == 1))[
            0
        ].tolist()
        new_random, new_selected = selection(new_eligible, quota)
        repeated.append(
            {
                "eligible_ids": new_eligible,
                "randomized_selection_order": new_random,
                "selected_ids": new_selected,
                "selected_target_records": [target_record(n) for n in new_selected],
            }
        )
    expected = {
        "eligible_ids": eligible,
        "randomized_selection_order": randomized,
        "selected_ids": selected,
        "selected_target_records": selected_records,
    }
    if not all(result == expected for result in repeated):
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_FREEZE_INVALID: selection determinism")
    train_classes, val_classes = (
        composition(pools["train"], data.y),
        composition(pools["val"], data.y),
    )
    directory = ART / scale
    write(
        directory / "eligible_test_pool.json",
        {
            "dataset_variant": scale.upper(),
            "node_ids": eligible,
            "count": len(eligible),
            "class_composition": composition(eligible, data.y),
            "eligibility": "test AND motif AND structurally_clean AND common_correct AND y==1",
        },
    )
    write(
        directory / "final_test_targets.json",
        {
            "dataset_variant": scale.upper(),
            "selected_ids": selected,
            "count": quota,
            "seed": 42,
            "randomized_selection_order": randomized,
            "selected_ids_in_randomized_order": randomized[:quota],
            "records": selected_records,
            "confidence_is_descriptive_only": True,
        },
    )
    for name, split, classes in (
        ("pg_train_pool", "train", train_classes),
        ("pg_validation_pool", "val", val_classes),
    ):
        write(
            directory / (name + ".json"),
            {
                "dataset_variant": scale.upper(),
                "split": split,
                "node_ids": pools[split],
                "count": len(pools[split]),
                "class_composition": classes,
                "eligibility": split + " AND motif AND structurally_clean AND common_correct",
                "class1_restriction": False,
                "shared_across_backbones": True,
                "subsampled": False,
            },
        )
    manifest = {
        "dataset_variant": scale.upper(),
        "dataset_path": record(input_path(scale, "dataset"))["path"],
        "dataset_sha256": INPUTS[scale]["dataset"],
        "checkpoints": {kind: record(input_path(scale, kind)) for kind in ("gcn", "graphsage")},
        "seed": 42,
        "selection_rule": "Sorted eligible IDs; dedicated CPU torch.Generator "
        "seed42; randperm; first quota; selected IDs sorted ascending",
        "eligibility_rule": "test AND motif AND structurally_clean AND common_correct AND y==1",
        "randomized_selection_order_semantics": "Full eligible permutation; "
        "selected prefix stored separately",
        "final_quota": quota,
        "eligible_test_count": len(eligible),
        "selected_count": quota,
        "selection_fraction": 0.75,
        "selected_ids": selected,
        "randomized_selection_order": randomized,
        "selected_ids_in_randomized_order": randomized[:quota],
        "selected_target_records": selected_records,
        "pg_train_pool_ids": pools["train"],
        "pg_validation_pool_ids": pools["val"],
        "pg_train_count": len(pools["train"]),
        "pg_validation_count": len(pools["val"]),
        "class_composition": {
            "final_targets": composition(selected, data.y),
            "pg_train": train_classes,
            "pg_validation": val_classes,
        },
        "structural_clean_conditions": "Own instance alone contributes GT; "
        "globalGT=ownGT=12, nonOwnGT=0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": environment,
        "git_commit": environment["git_commit"],
        "prediction_hashes": prediction_hashes,
        "predictive_readiness": diagnostic_rows,
        "test_nodes": int(data.test_mask.sum()),
        "motif_test": int((data.test_mask & motif).sum()),
        "confidence_used_for_selection": False,
        "no_training_or_explainer_execution": True,
        "track": "TRACK A ONLY - CLEAN SCALE ROBUSTNESS",
        "primary_gt": "GLOBAL edge mask restricted to induced 3-hop candidate edges",
        "determinism_verdict": f"BA{500 + 500 * index}_TARGET_SELECTION_DETERMINISM_PASS",
        "selection_verdict": f"BA{500 + 500 * index}_TARGET_SELECTION_PASS",
        "selection_determinism": {
            "repetitions": 2,
            "eligible_pool_equal": True,
            "randomized_order_equal": True,
            "selected_ids_equal": True,
            "selected_metadata_equal": True,
        },
        "pool_files": [record(directory / name) for name in FILES[:-1]],
    }
    write(directory / "target_manifest.json", manifest)
    manifest["manifest_file"] = record(directory / "target_manifest.json")
    emit(
        {
            "scale": scale,
            "eligible": len(eligible),
            "selected": quota,
            "selected_ids": selected,
            "pg_train": len(pools["train"]),
            "pg_validation": len(pools["val"]),
            "determinism": manifest["determinism_verdict"],
        }
    )
    return manifest


def tab(name: str, columns: list[str], rows: list[list[Any]]) -> dict:
    return {"name": name, "columns": columns, "rows": rows}


def reports(scales: list[dict]) -> None:
    tables = [
        tab(
            "BA-T1 Eligibility Counts",
            [
                "Scale",
                "TestNodes",
                "MotifTest",
                "CommonCorrectTest",
                "CleanCommonCorrectMotifTest",
                "Class1EligibleFinal",
                "Quota",
                "SelectionFraction",
            ],
            [
                [
                    s["dataset_variant"],
                    s["test_nodes"],
                    s["motif_test"],
                    s["predictive_readiness"][2]["common_correct"],
                    s["predictive_readiness"][2]["common_correct_motif_structurally_clean"],
                    s["eligible_test_count"],
                    s["final_quota"],
                    0.75,
                ]
                for s in scales
            ],
        ),
        tab(
            "BA-T2 Final Target Composition",
            [
                "Scale",
                "FinalTargets",
                "Class1",
                "Class2",
                "Class3",
                "MeanCandidateEdges",
                "MedianCandidateEdges",
                "MeanGTPrevalence",
            ],
            [
                [s["dataset_variant"], s["selected_count"]]
                + [s["class_composition"]["final_targets"][str(c)] for c in (1, 2, 3)]
                + [
                    statistics.mean(
                        r["candidate_directed_edge_count"] for r in s["selected_target_records"]
                    ),
                    statistics.median(
                        r["candidate_directed_edge_count"] for r in s["selected_target_records"]
                    ),
                    statistics.mean(
                        r["global_gt_prevalence"] for r in s["selected_target_records"]
                    ),
                ]
                for s in scales
            ],
        ),
        tab(
            "BA-T3 PG Train Validation Pools",
            ["Scale", "Split", "PoolSize", "Class1", "Class2", "Class3"],
            [
                [
                    s["dataset_variant"],
                    split,
                    s["pg_train_count" if split == "train" else "pg_validation_count"],
                ]
                + [
                    s["class_composition"]["pg_train" if split == "train" else "pg_validation"][
                        str(c)
                    ]
                    for c in (1, 2, 3)
                ]
                for s in scales
                for split in ("train", "validation")
            ],
        ),
        tab(
            "BA-T4 Final Target Validation",
            [
                "Scale",
                "TargetID",
                "Label",
                "GCNPred",
                "SAGEPred",
                "StructurallyClean",
                "GlobalGT",
                "OwnGT",
                "NonOwnGT",
                "CandidateEdges",
                "GTPrevalence",
            ],
            [
                [
                    s["dataset_variant"],
                    r["node_id"],
                    r["label"],
                    r["gcn_predicted_class"],
                    r["graphsage_predicted_class"],
                    r["structurally_clean"],
                    r["global_gt_directed_count"],
                    r["own_house_gt_directed_count"],
                    r["non_own_house_gt_directed_count"],
                    r["candidate_directed_edge_count"],
                    r["global_gt_prevalence"],
                ]
                for s in scales
                for r in s["selected_target_records"]
            ],
        ),
    ]
    for stem, chosen, payload in (
        (REPORTS[0], [tables[0], tables[1], tables[3]], {"scales": scales}),
        (
            REPORTS[1],
            [tables[2]],
            {
                "common_pools_not_backbone_specific": True,
                "scales": {s["dataset_variant"]: s["class_composition"] for s in scales},
            },
        ),
        (
            REPORTS[2],
            [],
            {
                "checks": {s["dataset_variant"]: s["selection_determinism"] for s in scales},
                "verdicts": [s["determinism_verdict"] for s in scales],
            },
        ),
    ):
        write(DOC / (stem + ".json"), {**payload, "tables": chosen})
        lines = [
            "# BA-House Scale Target Freeze",
            "",
            "TRACK A ONLY. No training or explainer runs. "
            "BA500 historical 12 targets is context only; extension quotas24/36 were fixed "
            "before explainer execution. Selection uses no confidence/degree/candidate/GT ranking. "
            "Primary global GT unchanged. Final verifier seal: 04_verifier_receipt.json.",
        ]
        for table in chosen:
            lines += [
                "",
                "## " + table["name"],
                "",
                "| " + " | ".join(table["columns"]) + " |",
                "| " + " | ".join(["---"] * len(table["columns"])) + " |",
            ]
            lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in table["rows"]]
        if not chosen:
            lines += [json.dumps(payload, indent=2)]
        with (DOC / (stem + ".md")).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")


def main() -> None:
    collisions = [record(p) for p in outputs() if p.exists()]
    if collisions:
        emit(collisions)
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_OUTPUT_ALREADY_EXISTS")
    for scale, hashes in INPUTS.items():
        for kind, checksum in hashes.items():
            if sha(input_path(scale, kind)) != checksum:
                raise RuntimeError("BA_HOUSE_SCALE_TARGET_INPUT_HASH_MISMATCH")
    protect()
    torch.set_num_threads(1)
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "pyg": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "evaluation_device": "cuda" if torch.cuda.is_available() else "cpu",
        "selection_device": "cpu",
        "cpu_threads": 1,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    write(
        DOC / "command_receipt.json",
        {
            "argv": sys.argv,
            "executable": sys.executable,
            "cwd": str(ROOT),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "mode": "Read-only frozen-backbone inference plus seeded target/pool freeze; "
            "no training",
        },
    )
    scales = [
        freeze_scale(scale, index, environment) for index, scale in enumerate(INPUTS, start=1)
    ]
    reports(scales)
    emit({"selection": "COMPLETE_AWAITING_INDEPENDENT_VERIFIER", "final_counts": [24, 36]})
    write(
        DOC / "selection_log.json",
        {
            "stdout": "\n".join(EVENTS) + "\n",
            "stderr": "",
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "training_or_explainer_runs": 0,
        },
    )
    write(
        ART / "target_freeze_master_manifest.json",
        {
            "scales": scales,
            "environment": environment,
            "file_hashes": [record(p) for p in outputs() if p.exists()],
            "source_hashes": [record(Path(__file__)), record(DOC / "verify_targets.py")],
            "independent_verifier_receipt": (DOC / "04_verifier_receipt.json")
            .relative_to(ROOT)
            .as_posix(),
            "self_after_and_verifier_receipt_excluded_to_avoid_hash_cycle": True,
        },
    )


if __name__ == "__main__":
    main()
