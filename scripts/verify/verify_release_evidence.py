# ruff: noqa: E402
"""Verify saved release evidence using the standard library; never run a model."""

from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from gnnexplainer_benchmark.utils.paths import resolve_repository_path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10), (a, b)


def verify(root):
    manifest = load(root / "docs/provenance/release_manifest.json")
    for entry in manifest["files"]:
        path = resolve_repository_path(entry["path"], root)
        assert path.is_file(), entry["path"]
        assert digest(path) == entry["sha256"], entry["path"]
    trace = load(root / "results/final/thesis_traceability.json")
    assert len(trace["records"]) == 217 and trace["withheld"]["count"] == 9
    assert len({x["thesis_identifier"] for x in trace["records"]}) == 22
    for row in trace["records"]:
        assert resolve_repository_path(row["release_evidence"], root).is_file(), row[
            "release_evidence"
        ]
        reported = row["extracted_numeric_value"]
        source = row["local_source_value"]
        # An inequality/qualitative display is kept for manual audit, not rounded equality.
        try:
            a, b = float(reported), float(source)
        except ValueError:
            continue
        decimals = len(reported.partition(".")[2])
        assert abs(a - b) <= 0.500001 * 10 ** (-decimals), row

    case = load(root / "results/elliptic/final200/historical_raw_final200.json")["measurement"]
    historical_nodes = {}
    for bb, rows in case["per_target"].items():
        assert len(rows) == 200 and len({x["node_id"] for x in rows}) == 200
        historical_nodes[bb.lower()] = {x["node_id"] for x in rows}
        for x in rows:
            n, k = x["candidate_edge_count"], x["selected_k"]
            assert k == math.ceil(0.2 * n)
            ids, raw = x["candidate_global_edge_indices"], x["raw_logits"]
            assert len(ids) == len(raw) == n and all(math.isfinite(v) for v in raw)
            ranked = sorted(range(n), key=lambda i: (-raw[i], ids[i]))
            assert set(x["selected_global_edge_indices"]) == {ids[i] for i in ranked[:k]}
            close(x["delta_remove"], x["p_original"] - x["p_remove"])
            close(x["retain_delta"], x["p_original"] - x["p_retain"])
            close(x["FidelityPlus"], x["original_predicted_class"] != x["remove_predicted_class"])
            close(x["FidelityMinus"], x["original_predicted_class"] != x["retain_predicted_class"])
        for key in ["FidelityPlus", "FidelityMinus", "delta_remove", "retain_delta", "sparsity"]:
            close(statistics.mean(x[key] for x in rows), case["aggregate"][bb][key])

    prefix = root / "docs/audit/elliptic_pg_stability/task29_final_evaluation_tie_v1"

    def table(name):
        with (prefix / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    pairs = {
        (x["backbone"], int(x["node_id"]), int(x["seed_a"]), int(x["seed_b"])): x
        for x in table("TASK29_FINAL_STABILITY_PAIRWISE.csv")
    }
    targets = {
        (x["backbone"], int(x["node_id"])): x
        for x in table("TASK29_FINAL_STABILITY_PER_TARGET.csv")
    }
    assert len(pairs) == 4000 and len(targets) == 400
    means = {"gcn": [], "graphsage": []}
    records = load(root / "results/elliptic/stability/fresh5_saved_selections.json")["records"]
    assert len(records) == 400
    for record in records:
        x = record["measurement"]
        bb, node = x["backbone"], x["node_id"]
        assert node in historical_nodes[bb]
        runs = {run["seed"]: run for run in x["runs"]}
        assert set(runs) == set(range(5))
        ids = x["candidate_global_edge_indices"]
        selected = {}
        for seed, run in runs.items():
            raw = run["raw_logits"]
            assert len(raw) == len(ids) and all(math.isfinite(v) for v in raw)
            k = math.ceil(0.2 * len(ids))
            ranked = sorted(range(len(ids)), key=lambda i: (-raw[i], ids[i]))
            selected[seed] = set(run["selected_global_edge_indices"])
            assert selected[seed] == {ids[i] for i in ranked[:k]}
        js = []
        for sa, sb in itertools.combinations(range(5), 2):
            intersection = len(selected[sa] & selected[sb])
            union = len(selected[sa] | selected[sb])
            j = intersection / union if union else 1.0
            expected = pairs[(bb, node, sa, sb)]
            assert intersection == int(expected["intersection"]) and union == int(expected["union"])
            close(j, expected["jaccard"])
            js.append(j)
        mean = statistics.mean(js)
        close(mean, targets[(bb, node)]["mean_pairwise_jaccard"])
        means[bb].append(mean)
    for row in table("TASK29_FINAL_STABILITY_SUMMARY.csv"):
        values = means[row["backbone"]]
        assert len(values) == int(row["targets"]) == 200
        close(statistics.mean(values), row["mean_jaccard"])
        close(statistics.stdev(values), row["sample_sd"])

    masks = load(root / "results/shapeggen/v1/diagnostics/thesis_diagnostic_counts.json")[
        "measurement"
    ]
    assert masks["thresholds"] == {"below_01": 0.01, "below_001": 0.001}
    assert masks["masks"]["gcn"]["auroc_below_half"] == 138
    assert masks["masks"]["gcn"]["negative_gt_gap"] == 153
    assert masks["masks"]["graphsage"]["below_001"] == 19590
    assert masks["masks"]["graphsage"]["n"] == 20572

    probes = load(root / "results/shapeggen/v1/diagnostics/loss_gradient_probes.json")[
        "measurement"
    ]
    for backbone in probes["backbones"].values():
        for split, count in [("train", 539), ("test", 167)]:
            cell = backbone["counterfactuals"][split]
            for condition in ["LEARNED", "ZERO", "ONE"]:
                rows = cell["per_target"][condition]
                assert len(rows) == count and len({x["node_id"] for x in rows}) == count
                for key in [
                    "prediction_ce",
                    "size",
                    "entropy",
                    "total",
                    "original_probability",
                    "masked_probability",
                    "probability_delta",
                ]:
                    close(
                        statistics.mean(x[key] for x in rows),
                        cell["conditions"][condition]["all"][key]["mean"],
                    )
                for x in rows:
                    # Original probe subtraction was float32, before JSON serialization.
                    assert math.isclose(
                        x["probability_delta"],
                        x["original_probability"] - x["masked_probability"],
                        rel_tol=0,
                        abs_tol=1e-7,
                    )

    ba_rows = 0
    for filename in [
        "phase4a_gnn_random/02_gnnexplainer_primary_results.csv",
        "phase4b1_pg_primary/05_pg_primary_test_results.csv",
    ]:
        with resolve_repository_path(root / "docs/extension/ba_house_scale" / filename, root).open(
            encoding="utf-8-sig"
        ) as stream:
            rows = list(csv.DictReader(stream))
        for scale, count in [("ba1000_ext", 24), ("ba1500_ext", 36)]:
            for bb in ["gcn", "graphsage"]:
                cell = [x for x in rows if x["scale"] == scale and x["backbone"] == bb]
                assert len(cell) == count and {x["seed"] for x in cell} == {"42"}
                for x in cell:
                    close(
                        x["remove_delta"],
                        float(x["original_probability"]) - float(x["remove_probability"]),
                    )
                    close(
                        x["retain_delta"],
                        float(x["original_probability"]) - float(x["retain_probability"]),
                    )
        ba_rows += len(rows)
    return {
        "status": "PASS",
        "manifest_files": len(manifest["files"]),
        "trace_cells_checked": 217,
        "withheld_cells": 9,
        "historical_elliptic_nodes": 400,
        "fresh5_records": 400,
        "fresh5_pairwise_comparisons": 4000,
        "ba_seed42_records": ba_rows,
        "scope": "Saved evidence arithmetic and hashes; no model reproduction or training",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
