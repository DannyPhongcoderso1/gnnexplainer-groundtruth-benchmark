from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/extension/ba_house_scale/phase1_generation"
MASTER = ROOT / "artifacts/extensions/ba_house_scale/ba_house_scale_dataset_manifest.json"
SCIENTIFIC = (
    "x",
    "y",
    "edge_index",
    "node_mask",
    "edge_mask",
    "train_mask",
    "val_mask",
    "test_mask",
)
CHECKS: list[dict[str, Any]] = []


def check(name: str, condition: bool, detail: Any = None) -> None:
    if not condition:
        raise RuntimeError("BA_HOUSE_SCALE_DATASET_PHASE_INVALID: " + name + "; " + str(detail))
    CHECKS.append({"check": name, "status": "PASS", "detail": detail})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def independent_splits(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    masks = [torch.zeros(len(labels), dtype=torch.bool) for _ in range(3)]
    for label in sorted(set(labels.tolist())):
        ids = torch.where(labels == label)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        ntrain, nval = int(0.6 * len(ids)), int(0.2 * len(ids))
        for mask, nodes in zip(
            masks, (ids[:ntrain], ids[ntrain : ntrain + nval], ids[ntrain + nval :]), strict=True
        ):
            mask[nodes] = True
    return tuple(masks)


def independent_regeneration(background: int, houses: int, seed: int) -> Any:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=background, num_edges=5),
        motif_generator=HouseMotif(),
        num_motifs=houses,
    )
    result = dataset[0]
    result.x = torch.ones((result.num_nodes, 10), dtype=torch.float32)
    result.train_mask, result.val_mask, result.test_mask = independent_splits(result.y, seed)
    result.motif_instance_id = torch.tensor(
        [0] * background + [i for i in range(1, houses + 1) for _ in range(5)]
    )
    result.motif_role_in_house = torch.tensor([-1] * background + list(range(5)) * houses)
    return result


def bfs(node: int, adjacency: list[set[int]]) -> set[int]:
    seen, frontier = {node}, {node}
    for _ in range(3):
        frontier = {v for u in frontier for v in adjacency[u]} - seen
        seen |= frontier
    return seen


def describe(values: list[int | float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=float)
    quantiles = np.quantile(a, [0.25, 0.5, 0.75], method="linear")
    return {
        "count": len(values),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "min": float(min(values)),
        "Q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "Q75": float(quantiles[2]),
        "max": float(max(values)),
    }


def verify_baseline(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before["files"] != after["files"]:
        raise RuntimeError("BA_HOUSE_SCALE_BASELINE_PRESERVATION_FAILURE: before/after")
    for record in before["files"]:
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise RuntimeError("BA_HOUSE_SCALE_BASELINE_PRESERVATION_FAILURE: " + str(path))
    reference = read_json(
        ROOT / "docs/extension/phase1_pipeline_audit/protected_baseline_manifest_after.json"
    )
    protected = {row["path"]: row for row in before["files"]}
    check(
        "phase1_scope_complete",
        all(protected.get(row["path"]) == row for row in reference["files"]),
    )
    check(
        "phase1_audit_files_protected",
        all(
            path.relative_to(ROOT).as_posix() in protected
            for path in (ROOT / "docs/extension/phase1_pipeline_audit").iterdir()
            if path.is_file()
        ),
    )
    check("baseline_independent_live_rehash", True, len(before["files"]))


def main() -> None:
    output = DOC / "04_verifier_receipt.json"
    if output.exists():
        print(
            json.dumps({"path": str(output), "bytes": output.stat().st_size, "sha256": sha(output)})
        )
        raise RuntimeError("BA_HOUSE_SCALE_PHASE1_OUTPUT_ALREADY_EXISTS")
    torch.set_num_threads(1)
    started = datetime.now(UTC).isoformat()
    before = read_json(DOC / "protected_baseline_before.json")
    after = read_json(DOC / "protected_baseline_after.json")
    verify_baseline(before, after)
    master = read_json(MASTER)
    check(
        "three_exact_variants",
        [v["dataset_variant"] for v in master["variants"]]
        == ["BA500_EXT", "BA1000_EXT", "BA1500_EXT"],
    )
    for record in master["file_hashes"] + master["source_hashes"]:
        path = ROOT / record["path"]
        check(
            "immutable_hash_" + record["path"],
            path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
        )
    structural = read_json(DOC / "03_structural_clean_pool.json")
    with (DOC / "03_structural_clean_pool.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    check("csv_json_same_row_count", len(csv_rows) == len(structural["records"]) == 1200)
    for raw, record in zip(csv_rows, structural["records"], strict=True):
        for key, value in record.items():
            parsed = raw[key] == "True" if isinstance(value, bool) else type(value)(raw[key])
            check(
                "csv_json_" + record["dataset_variant"] + "_" + str(record["node_id"]) + "_" + key,
                parsed == value,
            )
    checked_variants = []
    house_edges = {(0, 1), (0, 3), (0, 4), (1, 4), (1, 2), (2, 3)}
    frozen = torch.load(ROOT / "artifacts/data/ba_house.pt", map_location="cpu", weights_only=False)
    for index, variant in enumerate(master["variants"], start=1):
        name = variant["dataset_variant"]
        dataset_file = ROOT / variant["dataset"]["path"]
        data = torch.load(dataset_file, map_location="cpu", weights_only=False)
        background, houses, expected_nodes = 300 * index, 40 * index, 500 * index
        check(name + "_dataset_hash", sha(dataset_file) == variant["dataset"]["sha256"])
        parameters = variant["generation_config"]["parameters"]
        check(
            name + "_exact_config",
            parameters
            == dict(
                background_nodes=background,
                num_house_motifs=houses,
                ba_m=5,
                feature_dim=10,
                seed=42,
            ),
        )
        config = read_json(ROOT / variant["config_file"]["path"])
        check(name + "_config_manifest_parity", config == variant["generation_config"])
        check(name + "_nodes", data.num_nodes == expected_nodes)
        expected_labels = torch.tensor([0] * background + [1, 1, 2, 2, 3] * houses)
        check(name + "_label_semantics", same(data.y, expected_labels))
        check(
            name + "_feature_semantics",
            same(data.x, torch.ones((expected_nodes, 10), dtype=torch.float32)),
        )
        instance = torch.tensor(
            [0] * background + [h for h in range(1, houses + 1) for _ in range(5)]
        )
        role = torch.tensor([-1] * background + list(range(5)) * houses)
        check(name + "_instance_mapping", same(data.motif_instance_id, instance))
        check(name + "_role_mapping", same(data.motif_role_in_house, role))
        check(
            name + "_instance_count_size",
            Counter(instance.tolist()) == {0: background, **dict.fromkeys(range(1, houses + 1), 5)},
        )
        check(
            name + "_motif_fraction",
            int((instance > 0).sum()) == 5 * houses and 5 * houses / expected_nodes == 0.4,
        )
        edge_list = list(map(tuple, data.edge_index.t().tolist()))
        edge_set = set(edge_list)
        check(
            name + "_reciprocal_simple_graph",
            len(edge_list) == len(edge_set)
            and all(u != v and (v, u) in edge_set for u, v in edge_set),
        )
        actual_house_pairs = {
            tuple(sorted((u, v))) for u, v in edge_set if u >= background and v >= background
        }
        expected_house_pairs = {
            tuple(sorted((background + 5 * h + u, background + 5 * h + v)))
            for h in range(houses)
            for u, v in house_edges
        }
        check(name + "_house_connectivity", actual_house_pairs == expected_house_pairs)
        global_gt = torch.tensor(
            [tuple(sorted((u, v))) in expected_house_pairs for u, v in edge_list]
        )
        check(
            name + "_global_node_gt",
            torch.equal(data.node_mask.bool(), instance > 0)
            and bool(((data.node_mask == 0) | (data.node_mask == 1)).all()),
        )
        check(
            name + "_global_edge_gt",
            torch.equal(data.edge_mask.bool(), global_gt)
            and bool(((data.edge_mask == 0) | (data.edge_mask == 1)).all()),
        )
        check(name + "_positive_gt_count", int(global_gt.sum()) == 12 * houses)
        bridges = [(u, v) for u, v in edge_list if (u >= background) != (v >= background)]
        check(
            name + "_bridge_count_negative",
            len(bridges) == 2 * houses
            and all(
                not bool(data.edge_mask[i])
                for i, (u, v) in enumerate(edge_list)
                if (u >= background) != (v >= background)
            ),
        )
        bridge_house_ids = Counter(int(instance[u if u >= background else v]) for u, v in bridges)
        check(
            name + "_one_bridge_per_house",
            bridge_house_ids == dict.fromkeys(range(1, houses + 1), 2),
        )
        masks = (data.train_mask, data.val_mask, data.test_mask)
        regenerated_splits = independent_splits(data.y, 42)
        check(
            name + "_split_determinism",
            all(same(a, b) for a, b in zip(masks, regenerated_splits, strict=True)),
        )
        check(name + "_split_coverage", bool(torch.all(sum(m.int() for m in masks) == 1)))
        check(
            name + "_split_totals",
            [int(m.sum()) for m in masks] == [300 * index, 100 * index, 100 * index],
        )
        expected_class_splits = [[180, 48, 48, 24], [60, 16, 16, 8], [60, 16, 16, 8]]
        for mask, counts in zip(masks, expected_class_splits, strict=True):
            check(
                name + "_split_class_counts_" + str(counts),
                torch.bincount(data.y[mask], minlength=4).tolist() == [n * index for n in counts],
            )
        adjacency = [set() for _ in range(expected_nodes)]
        for u, v in edge_list:
            adjacency[u].add(v)
        degrees = [len(neighbors) for neighbors in adjacency]
        independently_counted = {
            "num_nodes": expected_nodes,
            "undirected_edges": len(edge_set) // 2,
            "directed_entries": len(edge_list),
            "average_degree": sum(degrees) / expected_nodes,
            "median_degree": float(np.median(degrees)),
            "min_degree": min(degrees),
            "max_degree": max(degrees),
            "class_counts": dict(sorted(Counter(map(str, data.y.tolist())).items())),
            "background_undirected_edges": sum(u < v < background for u, v in edge_set),
        }
        check(
            name + "_reported_structure",
            all(variant["structure"][k] == value for k, value in independently_counted.items()),
        )
        diagnostic = torch.load(
            ROOT / variant["diagnostic_masks"]["path"], map_location="cpu", weights_only=False
        )
        motif_ids = list(range(background, expected_nodes))
        records = [r for r in structural["records"] if r["dataset_variant"] == name]
        check(
            name + "_all_motif_records",
            [r["node_id"] for r in records] == motif_ids
            and diagnostic["node_ids"].tolist() == motif_ids,
        )
        for row_index, record in enumerate(records):
            node = record["node_id"]
            own_id = int(instance[node])
            subset = bfs(node, adjacency)
            candidate = torch.tensor([u in subset and v in subset for u, v in edge_list])
            primary_gt = candidate & global_gt
            own_gt = torch.tensor(
                [
                    bool(primary_gt[i])
                    and int(instance[u]) == own_id
                    and int(instance[v]) == own_id
                    for i, (u, v) in enumerate(edge_list)
                ]
            )
            non_own = primary_gt & ~own_gt
            present_ids = {
                int(instance[edge_list[i][0]]) for i in torch.where(primary_gt)[0].tolist()
            }
            global_count, own_count, non_count = (
                int(primary_gt.sum()),
                int(own_gt.sum()),
                int(non_own.sum()),
            )
            clean = present_ids == {own_id} and global_count == own_count == 12 and non_count == 0
            independently_computed = {
                "dataset_variant": name,
                "node_id": node,
                "class": int(data.y[node]),
                "own_motif_instance": own_id,
                "candidate_directed_edge_count": int(candidate.sum()),
                "candidate_node_count": len(subset),
                "motif_instance_count_3hop": len(present_ids),
                "global_gt_directed_count": global_count,
                "own_house_gt_directed_count": own_count,
                "non_own_house_gt_directed_count": non_count,
                "global_gt_prevalence": global_count / int(candidate.sum()),
                "structurally_clean": clean,
            }
            check(name + "_bfs_structural_record_" + str(node), record == independently_computed)
            check(name + "_decomposition_" + str(node), global_count == own_count + non_count)
            for key, value in [
                ("candidate_edge_mask", candidate),
                ("target_global_gt_mask", primary_gt),
                ("own_house_gt_mask", own_gt),
                ("non_own_house_gt_mask", non_own),
            ]:
                check(
                    name + "_saved_mask_" + str(node) + "_" + key,
                    same(diagnostic[key][row_index], value),
                )
            check(
                name + "_finite_record_" + str(node),
                all(
                    math.isfinite(float(v)) for v in record.values() if isinstance(v, (int, float))
                ),
            )
        clean_records = [r for r in records if r["structurally_clean"]]
        check(name + "_clean_exists", len(clean_records) > 0)
        clean_summary = variant["structural_pool"]
        check(name + "_clean_count", len(clean_records) == clean_summary["structurally_clean"])
        for field, values in clean_summary["clean_distributions"].items():
            recomputed = describe([r[field] for r in clean_records])
            check(
                name + "_clean_distribution_" + field,
                all(
                    math.isclose(float(values[k]), float(v), rel_tol=1e-12, abs_tol=1e-12)
                    for k, v in recomputed.items()
                ),
            )
        for field, values in clean_summary["all_motif_distributions"].items():
            recomputed = describe([r[field] for r in records])
            check(
                name + "_all_motif_distribution_" + field,
                all(
                    math.isclose(float(values[k]), float(v), rel_tol=1e-12, abs_tol=1e-12)
                    for k, v in recomputed.items()
                ),
            )
        regenerated = independent_regeneration(background, houses, 42)
        parity = {
            key: same(getattr(data, key), getattr(regenerated, key))
            for key in (*SCIENTIFIC, "motif_instance_id", "motif_role_in_house")
        }
        check(name + "_independent_regeneration", all(parity.values()), parity)
        if name == "BA500_EXT":
            baseline_parity = {
                key: same(getattr(data, key), getattr(frozen, key)) for key in SCIENTIFIC
            }
            check(
                "ba500_control_tensor_parity",
                baseline_parity == master["ba500_control"]["tensor_parity"],
            )
            check(
                "ba500_control_verdict",
                master["ba500_control"]["verdict"]
                == (
                    "BA500_EXT_EXACT_DATA_REPRODUCTION"
                    if all(baseline_parity.values())
                    else "BA500_EXT_SEMANTIC_DATA_REPRODUCTION"
                ),
            )
        checked_variants.append(
            {
                "variant": name,
                "dataset_sha256": sha(dataset_file),
                "nodes": expected_nodes,
                "directed_entries": len(edge_list),
                "motif_nodes": len(records),
                "structurally_clean": len(clean_records),
                "determinism_tensor_parity": parity,
                "determinism_verdict": name + "_DETERMINISM_PASS",
            }
        )
        print(name + " INDEPENDENT_STRUCTURAL_AND_DETERMINISM_PASS", flush=True)
    verify_baseline(before, after)
    check(
        "required_scale_tables",
        all(
            "BA-S" + str(i) in (DOC / "02_scale_parity_report.md").read_text() for i in range(1, 6)
        ),
    )
    gates = {f"G{i}": "PASS" for i in range(1, 17)}
    result = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "status": "BA_HOUSE_SCALE_INDEPENDENT_VERIFIER_PASS",
        "phase_verdict": "BA_HOUSE_SCALE_DATASET_PHASE_COMPLETE",
        "ba500_control_verdict": master["ba500_control"]["verdict"],
        "structural_protocol_verdict": "BA_HOUSE_TRACK_A_STRUCTURAL_PROTOCOL_FEASIBLE",
        "baseline_preservation_verdict": "BA_HOUSE_SCALE_BASELINE_PRESERVATION_PASS",
        "protected_files_rehashed": len(before["files"]),
        "variants": checked_variants,
        "pass_gates": gates,
        "check_count": len(CHECKS),
        "checks": CHECKS,
        "independence": "Separate process; no extension wrapper/report-builder imports; "
        "direct PyG regeneration, independent class split algorithm and adjacency BFS.",
        "dataset_only": True,
        "no_model_load_or_model_related_activity": True,
        "verifier_source_sha256": sha(Path(__file__)),
        "master_manifest_sha256": sha(MASTER),
        "command": {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "working_directory": str(ROOT),
        },
        "stdout": "\n".join(
            v["variant"] + " INDEPENDENT_STRUCTURAL_AND_DETERMINISM_PASS" for v in checked_variants
        )
        + "\nBA_HOUSE_SCALE_INDEPENDENT_VERIFIER_PASS\n",
        "stderr": "",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("BA_HOUSE_SCALE_INDEPENDENT_VERIFIER_PASS")
    print(
        json.dumps(
            {
                "phase_verdict": result["phase_verdict"],
                "checks": len(CHECKS),
                "variants": checked_variants,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
