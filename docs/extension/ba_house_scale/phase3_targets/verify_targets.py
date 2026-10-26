from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parent
ART = ROOT / "artifacts/extensions/ba_house_scale/targets"
BA1 = ROOT / "docs/extension/ba_house_scale/phase1_generation"
BA2 = ROOT / "docs/extension/ba_house_scale/phase2_backbones"
HASHES = [
    (
        "7680fbaba95bcfecb96d5f8de0d4935b469e16d31be75d803ed583e5ca8e5b9a",
        "812e123a2ce3c3ed0d9b04eaca78cbba8a4f72d2fa61abf09cf336dde01d75ef",
        "2c51117461c3eb2ff1cb29b291296f5202d114ba4ac4e8543754b6ceeeb607ca",
    ),
    (
        "a4d227ac3b56cf40966c19a5afeef8edbd6a2ad65a9d4a36e34ae5a3dd135b95",
        "be0ffcacc4031eb0dec6195120c608791082475a3d498c59bee1acbb89451bd7",
        "01dcc7c4818e34a91d5d19c36acbefec818b095c29adb13dff208cb4a2a1b56e",
    ),
]
CHECKS: list[dict] = []


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def record(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def check(name: str, valid: bool, detail: Any = None) -> None:
    if not valid:
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_FREEZE_INVALID: " + name + "; " + str(detail))
    CHECKS.append({"check": name, "status": "PASS", "detail": detail})


def rehash(before: dict) -> list[dict]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(lambda r: record(ROOT / r["path"]), before["files"]))
    if actual != before["files"]:
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_INPUT_PRESERVATION_FAILURE")
    return actual


class Backbone(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                GCNConv(a, b, cached=False)
                if kind == "gcn"
                else SAGEConv(a, b, aggr="sum", normalize=False, root_weight=True)
                for a, b in ((10, 32), (32, 32), (32, 4))
            ]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = nn.functional.dropout(self.convs[0](x, edge_index).relu(), 0.2, self.training)
        x = nn.functional.dropout(self.convs[1](x, edge_index).relu(), 0.2, self.training)
        return self.convs[2](x, edge_index)


def classes(ids: list[int], labels: torch.Tensor) -> dict:
    chosen = labels[torch.tensor(ids, dtype=torch.long)]
    return {str(c): int((chosen == c).sum()) for c in (1, 2, 3)}


def main() -> None:
    receipt_path, after_path = DOC / "04_verifier_receipt.json", DOC / "protected_inputs_after.json"
    existing = [record(p) for p in (receipt_path, after_path) if p.exists()]
    if existing:
        print(json.dumps(existing, indent=2))
        raise RuntimeError("BA_HOUSE_SCALE_TARGET_OUTPUT_ALREADY_EXISTS")
    torch.set_num_threads(1)
    started = datetime.now(UTC).isoformat()
    before = read(DOC / "protected_inputs_before.json")
    rehash(before)
    protected = {r["path"]: r for r in before["files"]}
    previous = read(BA2 / "protected_inputs_after.json")
    check(
        "full_previous_input_scope_protected",
        all(protected.get(r["path"]) == r for r in previous["files"]),
    )
    prior_run = read(BA2 / "run_manifest.json")
    prior_receipt = read(BA2 / "04_verifier_receipt.json")
    prior_outputs = (
        prior_run["source_hashes"]
        + prior_run["output_hashes"]
        + [
            prior_receipt["run_manifest"],
            prior_receipt["protected_inputs_after"],
            record(BA2 / "04_verifier_receipt.json"),
        ]
    )
    check(
        "all_30_BA2_outputs_protected",
        len(prior_outputs) == 30 and all(protected.get(r["path"]) == r for r in prior_outputs),
    )
    check(
        "BA500_firewall",
        all(
            p in protected
            for p in (
                "artifacts/data/ba_house.pt",
                "artifacts/checkpoints/ba_house_gcn_best.pt",
                "artifacts/checkpoints/ba_house_graphsage_best.pt",
                "configs/targets/ba_house_2x2_targets.csv",
            )
        ),
    )
    master_path = ART / "target_freeze_master_manifest.json"
    master = read(master_path)
    check(
        "two_scales_only",
        [s["dataset_variant"] for s in master["scales"]] == ["BA1000_EXT", "BA1500_EXT"],
    )
    for row in master["file_hashes"] + master["source_hashes"]:
        check("new_file_hash_" + row["path"], record(ROOT / row["path"]) == row)
    structural = read(BA1 / "03_structural_clean_pool.json")["records"]
    frozen_ready = read(BA2 / "03_common_correct_readiness.json")["scales"]
    ba1_master = read(
        ROOT / "artifacts/extensions/ba_house_scale/ba_house_scale_dataset_manifest.json"
    )
    verified, expected_t1, expected_t2, expected_t3, expected_t4 = [], [], [], [], []
    for index, manifest in enumerate(master["scales"], start=1):
        scale = manifest["dataset_variant"].lower()
        directory = ART / scale
        dataset_path = ROOT / manifest["dataset_path"]
        if sha(dataset_path) != HASHES[index - 1][0]:
            raise RuntimeError("BA_HOUSE_SCALE_TARGET_INPUT_HASH_MISMATCH")
        check(scale + "_dataset_binding", manifest["dataset_sha256"] == HASHES[index - 1][0])
        data = torch.load(dataset_path, map_location="cpu", weights_only=False)
        device = torch.device(master["environment"]["evaluation_device"])
        inference = data.clone().to(device)
        pred, prob = {}, {}
        for position, kind in enumerate(("gcn", "graphsage"), start=1):
            path = ROOT / manifest["checkpoints"][kind]["path"]
            if sha(path) != HASHES[index - 1][position]:
                raise RuntimeError("BA_HOUSE_SCALE_TARGET_INPUT_HASH_MISMATCH")
            check(
                scale + "_checkpoint_binding_" + kind,
                manifest["checkpoints"][kind]["sha256"] == HASHES[index - 1][position],
            )
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = Backbone(kind).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            parameters_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.eval()
            with torch.no_grad():
                logits = model(inference.x, inference.edge_index)
            check(scale + "_finite_logits_" + kind, bool(torch.isfinite(logits).all()))
            check(
                scale + "_no_model_mutation_" + kind,
                all(torch.equal(t, model.state_dict()[k]) for k, t in parameters_before.items()),
            )
            pred[kind] = logits.argmax(-1).cpu()
            prob[kind] = logits.softmax(-1)[:, 1].cpu()
            check(
                scale + "_prediction_vector_" + kind,
                hashlib.sha256(pred[kind].numpy().tobytes()).hexdigest()
                == manifest["prediction_hashes"][kind],
            )
            old = next(
                c for c in prior_run["cells"] if c["scale"] == scale and c["backbone"] == kind
            )
            for split in ("train", "val", "test"):
                mask = getattr(data, split + "_mask")
                check(
                    scale + "_BA2_predictions_" + kind + "_" + split,
                    int((pred[kind][mask] == data.y[mask]).sum())
                    == old["metrics"][split]["correct"],
                )
        rows = {r["node_id"]: r for r in structural if r["dataset_variant"] == scale.upper()}
        ba1 = next(v for v in ba1_master["variants"] if v["dataset_variant"] == scale.upper())
        diagnostic = torch.load(
            ROOT / ba1["diagnostic_masks"]["path"], map_location="cpu", weights_only=False
        )
        clean = torch.zeros(data.num_nodes, dtype=torch.bool)
        source, target = data.edge_index
        check(
            scale + "_all_structural_records",
            list(rows) == diagnostic["node_ids"].tolist() == torch.where(data.y > 0)[0].tolist(),
        )
        for row_index, node in enumerate(diagnostic["node_ids"].tolist()):
            row = rows[node]
            candidate = diagnostic["candidate_edge_mask"][row_index]
            global_gt = candidate & data.edge_mask.bool()
            instance = int(data.motif_instance_id[node])
            own = (
                global_gt
                & (data.motif_instance_id[source] == instance)
                & (data.motif_instance_id[target] == instance)
            )
            other = global_gt & ~own
            instances = set(data.motif_instance_id[source[global_gt]].tolist())
            counts = (int(global_gt.sum()), int(own.sum()), int(other.sum()))
            is_clean = (
                bool(data.node_mask[node])
                and int(data.y[node]) > 0
                and instances == {instance}
                and counts == (12, 12, 0)
            )
            clean[node] = is_clean
            check(
                scale + "_structural_gt_" + str(node),
                row["structurally_clean"] == is_clean
                and counts
                == (
                    row["global_gt_directed_count"],
                    row["own_house_gt_directed_count"],
                    row["non_own_house_gt_directed_count"],
                )
                and row["motif_instance_count_3hop"] == len(instances)
                and row["candidate_directed_edge_count"] == int(candidate.sum())
                and row["candidate_node_count"]
                == int(data.edge_index[:, candidate].unique().numel())
                and row["global_gt_prevalence"] == counts[0] / int(candidate.sum())
                and torch.equal(global_gt, diagnostic["target_global_gt_mask"][row_index])
                and torch.equal(own, diagnostic["own_house_gt_mask"][row_index])
                and torch.equal(other, diagnostic["non_own_house_gt_mask"][row_index]),
            )
        motif = data.node_mask.bool() & (data.y > 0) & (data.motif_instance_id > 0)
        common = torch.stack([pred["gcn"] == data.y, pred["graphsage"] == data.y]).all(0)
        pg = {
            split: torch.where(getattr(data, split + "_mask") & motif & clean & common)[0].tolist()
            for split in ("train", "val", "test")
        }
        diagnostic_rows = [
            {
                "split": s,
                "common_correct": int((getattr(data, s + "_mask") & common).sum()),
                "common_correct_motif": int((getattr(data, s + "_mask") & common & motif).sum()),
                "common_correct_motif_structurally_clean": len(pg[s]),
            }
            for s in ("train", "val", "test")
        ]
        if (
            diagnostic_rows != frozen_ready[scale]["splits"]
            or {s: len(ids) for s, ids in pg.items()}
            != {
                "train": 47 * (data.num_nodes // 500),
                "val": 16 * (data.num_nodes // 500),
                "test": 16 * (data.num_nodes // 500),
            }
            or classes(pg["test"], data.y) != {"1": 16 * (data.num_nodes // 500), "2": 0, "3": 0}
        ):
            raise RuntimeError("BA_HOUSE_SCALE_TARGET_POOL_RECOMPUTE_MISMATCH")
        eligible = sorted(
            torch.where(data.test_mask & motif & clean & common & (data.y == 1))[0].tolist()
        )
        eligible_file = read(directory / "eligible_test_pool.json")
        check(
            scale + "_eligible_pool",
            eligible_file["node_ids"] == eligible
            and eligible_file["count"] == len(eligible) == 16 * (data.num_nodes // 500),
        )
        quota = int(12 * data.num_nodes / 500)
        check(
            scale + "_quota_fraction",
            quota
            == 12 * (data.num_nodes // 500)
            == manifest["final_quota"]
            == manifest["selected_count"]
            and quota / len(eligible) == manifest["selection_fraction"] == 0.75,
        )
        final_file = read(directory / "final_test_targets.json")
        repeated = []
        for repetition in range(2):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(42)
            permutation = torch.randperm(len(eligible), generator=generator)
            randomized = torch.tensor(eligible, dtype=torch.long)[permutation].tolist()
            selected = sorted(randomized[:quota])
            check(
                scale + "_seeded_selection_" + str(repetition),
                randomized
                == final_file["randomized_selection_order"]
                == manifest["randomized_selection_order"]
                and selected == final_file["selected_ids"] == manifest["selected_ids"]
                and randomized[:quota]
                == final_file["selected_ids_in_randomized_order"]
                == manifest["selected_ids_in_randomized_order"],
            )
            records = []
            for node in selected:
                structural_row = rows[node]
                computed = {
                    "node_id": node,
                    "label": int(data.y[node]),
                    "motif_instance_id": int(data.motif_instance_id[node]),
                    "motif_node": bool(motif[node]),
                    "test_mask": bool(data.test_mask[node]),
                    "structurally_clean": bool(clean[node]),
                    **{
                        k: structural_row[k]
                        for k in (
                            "global_gt_directed_count",
                            "own_house_gt_directed_count",
                            "non_own_house_gt_directed_count",
                            "candidate_directed_edge_count",
                            "candidate_node_count",
                            "global_gt_prevalence",
                        )
                    },
                    "gcn_predicted_class": int(pred["gcn"][node]),
                    "graphsage_predicted_class": int(pred["graphsage"][node]),
                    "gcn_probability_class1": float(prob["gcn"][node]),
                    "graphsage_probability_class1": float(prob["graphsage"][node]),
                }
                saved = next(r for r in final_file["records"] if r["node_id"] == node)
                check(
                    scale + "_target_metadata_" + str(repetition) + "_" + str(node),
                    set(computed) == set(saved)
                    and all(
                        math.isclose(value, saved[k], rel_tol=1e-6, abs_tol=1e-6)
                        if k.endswith("probability_class1")
                        else value == saved[k]
                        for k, value in computed.items()
                    ),
                )
                check(
                    scale + "_selected_gate_" + str(node),
                    bool(data.test_mask[node])
                    and int(data.y[node])
                    == int(pred["gcn"][node])
                    == int(pred["graphsage"][node])
                    == 1
                    and bool(motif[node])
                    and bool(clean[node])
                    and (
                        computed["global_gt_directed_count"],
                        computed["own_house_gt_directed_count"],
                        computed["non_own_house_gt_directed_count"],
                    )
                    == (12, 12, 0),
                )
                records.append(computed)
            repeated.append(
                {
                    "eligible": eligible,
                    "randomized": randomized,
                    "selected": selected,
                    "records": records,
                }
            )
        check(scale + "_deterministic_all_metadata", repeated[0] == repeated[1])
        check(
            scale + "_selected_record_manifest",
            final_file["records"] == manifest["selected_target_records"]
            and len(final_file["records"]) == len(set(manifest["selected_ids"])) == quota,
        )
        for name, split in (("pg_train_pool", "train"), ("pg_validation_pool", "val")):
            payload = read(directory / (name + ".json"))
            key = "pg_train" if split == "train" else "pg_validation"
            check(
                scale + "_" + name,
                payload["node_ids"] == manifest[key + "_pool_ids"] == pg[split]
                and payload["count"] == manifest[key + "_count"] == len(pg[split])
                and payload["class_composition"]
                == manifest["class_composition"][key]
                == classes(pg[split], data.y)
                and not payload["class1_restriction"]
                and payload["shared_across_backbones"]
                and not payload["subsampled"],
            )
            check(
                scale + "_split_disjoint_" + split,
                not set(pg[split]) & set(manifest["selected_ids"]),
            )
            expected_t3.append(
                [scale.upper(), "train" if split == "train" else "validation", len(pg[split])]
                + [classes(pg[split], data.y)[str(c)] for c in (1, 2, 3)]
            )
        check(scale + "_PG_train_val_disjoint", not set(pg["train"]) & set(pg["val"]))
        check(
            scale + "_final_composition",
            manifest["class_composition"]["final_targets"] == {"1": quota, "2": 0, "3": 0},
        )
        expected_t1.append(
            [
                scale.upper(),
                int(data.test_mask.sum()),
                int((data.test_mask & motif).sum()),
                diagnostic_rows[2]["common_correct"],
                len(pg["test"]),
                len(eligible),
                quota,
                0.75,
            ]
        )
        records = repeated[0]["records"]
        expected_t2.append(
            [
                scale.upper(),
                quota,
                quota,
                0,
                0,
                statistics.mean(r["candidate_directed_edge_count"] for r in records),
                statistics.median(r["candidate_directed_edge_count"] for r in records),
                statistics.mean(r["global_gt_prevalence"] for r in records),
            ]
        )
        expected_t4 += [
            [
                scale.upper(),
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
            for r in records
        ]
        verified.append(
            {
                "scale": scale.upper(),
                "eligible": len(eligible),
                "selected": quota,
                "selected_ids": manifest["selected_ids"],
                "pg_train": len(pg["train"]),
                "pg_validation": len(pg["val"]),
                "selection_verdict": f"BA{data.num_nodes}_TARGET_SELECTION_PASS",
                "determinism_verdict": f"BA{data.num_nodes}_TARGET_SELECTION_DETERMINISM_PASS",
            }
        )
        print(scale + " INDEPENDENT_TARGET_AND_PG_POOL_VERIFICATION_PASS", flush=True)
    report = read(DOC / "01_target_freeze_report.json")
    pool_report = read(DOC / "02_pg_pool_report.json")
    tables = {t["name"].split()[0]: t for t in report["tables"] + pool_report["tables"]}
    for name, expected in zip(
        ("BA-T1", "BA-T2", "BA-T3", "BA-T4"),
        (expected_t1, expected_t2, expected_t3, expected_t4),
        strict=True,
    ):
        check(name + "_table", tables[name]["rows"] == expected)
    check("target_report_manifest_parity", report["scales"] == master["scales"])
    current_inventory = {
        p.relative_to(ROOT).as_posix()
        for name in before["inventory_roots"]
        for p in (ROOT / name).rglob("*")
        if p.is_file()
    }
    old_inventory = set(before["artifact_inventory_before"])
    allowed_new = {
        r["path"]
        for r in master["file_hashes"]
        if r["path"].startswith("artifacts/extensions/ba_house_scale/targets/")
    } | {master_path.relative_to(ROOT).as_posix()}
    check(
        "no_explainer_or_other_artifact_created",
        current_inventory - old_inventory == allowed_new and not old_inventory - current_inventory,
        {"new_artifacts": sorted(current_inventory - old_inventory)},
    )
    check(
        "no_training_or_explainer_calls_recorded",
        read(DOC / "selection_log.json")["training_or_explainer_runs"] == 0,
    )
    after_rows = rehash(before)
    after = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(after_rows),
        "files": after_rows,
        "exact_equality": True,
        "verdict": "BA_HOUSE_SCALE_TARGET_INPUT_PRESERVATION_PASS",
    }
    with after_path.open("x", encoding="utf-8") as stream:
        json.dump(after, stream, indent=2)
        stream.write("\n")
    for row in master["file_hashes"] + master["source_hashes"]:
        check("final_hash_" + row["path"], record(ROOT / row["path"]) == row)
    result = {
        "status": "BA_HOUSE_SCALE_TARGET_INDEPENDENT_VERIFIER_PASS",
        "phase_verdict": "BA_HOUSE_SCALE_TARGET_FREEZE_COMPLETE",
        "pg_pools_verdict": "BA_HOUSE_SCALE_PG_POOLS_FROZEN",
        "preservation_verdict": after["verdict"],
        "protected_file_count": len(after_rows),
        "pass_gates": {f"G{i}": "PASS" for i in range(1, 21)},
        "scales": verified,
        "checks": CHECKS,
        "check_count": len(CHECKS),
        "independence": "Own backbone reconstruction, frozen-GT diagnostic decomposition, "
        "pool arithmetic and twice-repeated CPU generator selection; no selector/report imports",
        "confidence_comparison": "Inference metadata tolerance1e-6; "
        "repeated selection metadata exact",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "master_manifest": record(master_path),
        "protected_inputs_after": record(after_path),
        "source": record(Path(__file__)),
        "training_performed": False,
        "explainer_runs": 0,
        "command": {"argv": sys.argv, "executable": sys.executable, "cwd": str(ROOT)},
        "stdout": "\n".join(
            v["scale"].lower() + " INDEPENDENT_TARGET_AND_PG_POOL_VERIFICATION_PASS"
            for v in verified
        )
        + "\nBA_HOUSE_SCALE_TARGET_INDEPENDENT_VERIFIER_PASS\n",
        "stderr": "",
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
                    "pg_pools_verdict",
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
