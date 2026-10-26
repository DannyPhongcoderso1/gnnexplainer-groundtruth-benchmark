"""Future offline aggregation on explicitly supplied extraction artifacts, no import-time work."""

import argparse
import itertools
import math
import statistics
from pathlib import Path

from common import BENCHMARK, HERE, append_csv, atomic_json, identity, read_json, require, sha


def calculate(runs, nodes):
    require(sorted(r["seed"] for r in runs) == list(range(5)), "Fresh5 seeds0..4 required")
    pairs, targets = [], []
    for run in runs:
        require([r["node_id"] for r in run["records"]] == nodes, "Target identity/order mismatch")
    for index, node in enumerate(nodes):
        rows = [run["records"][index] for run in runs]
        candidates = rows[0]["candidate_global_edge_indices"]
        require(
            candidates == sorted(set(candidates))
            and candidates
            and all(type(e) is int and 0 <= e < 468710 for e in candidates),
            "Invalid directed edge IDs",
        )
        k = max(1, math.ceil(0.2 * len(candidates)))
        sets = {}
        for run, row in zip(runs, rows, strict=True):
            selected = row["selected_global_edge_indices"]
            require(
                row["candidate_global_edge_indices"] == candidates
                and row["k"] == k
                and selected == sorted(set(selected))
                and len(selected) == k
                and set(selected) <= set(candidates),
                "Candidate/budget/selected-set mismatch",
            )
            raw = row["raw_logits"]
            require(
                len(raw) == len(candidates) and all(math.isfinite(z) for z in raw),
                "Invalid raw scores",
            )
            selected_values = [
                z for e, z in zip(candidates, raw, strict=True) if e in set(selected)
            ]
            other = [z for e, z in zip(candidates, raw, strict=True) if e not in set(selected)]
            require(other and min(selected_values) > max(other), "RAW topk boundary mismatch/tie")
            sets[run["seed"]] = set(selected)
        values = []
        for a, b in itertools.combinations(sorted(sets), 2):
            intersection, union = len(sets[a] & sets[b]), len(sets[a] | sets[b])
            value = intersection / union
            values.append(value)
            pairs.append(
                {
                    "node_id": node,
                    "seed_a": a,
                    "seed_b": b,
                    "intersection_count": intersection,
                    "union_count": union,
                    "jaccard": value,
                }
            )
        targets.append(
            {"node_id": node, "mean_pairwise_jaccard": statistics.mean(values), "num_pairs": 10}
        )
    numbers = [r["mean_pairwise_jaccard"] for r in targets]
    return (
        pairs,
        targets,
        {
            "mean": statistics.mean(numbers),
            "sample_sd": statistics.stdev(numbers),
            "ddof": 1,
            "targets": len(nodes),
            "seeds": 5,
            "pairs_per_target": 10,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-stability", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(args.execute_stability, "Stability requires separate explicit authorization")
    data = read_json(args.input)
    require(
        data["protocol_identity"] == identity(read_json(HERE / "protocol.json")),
        "Protocol identity mismatch",
    )
    require(
        data["backbone"] in ("gcn", "graphsage") and data["hops"] == 2 and data["top_ratio"] == 0.2,
        "Backbone/candidate/budget protocol mismatch",
    )
    require(
        data["inference"] == "RAW_FLOAT32_CPU_TOPK_CANONICAL_DIRECTED_GLOBAL_IDS",
        "Wrong inference rule",
    )
    require(
        data["target_sha256"] == "e414941e6943b06484ecd5028c51daf80ff2bf9f1bbb06e6a5fd816de8dfb9c0",
        "Wrong final200 identity",
    )
    nodes = [r["node_id"] for r in data["runs"][0]["records"]]
    from state_verify import pool_ids

    frozen_targets = BENCHMARK / "configs/targets/elliptic_xai_common_test_targets.csv"
    require(
        sha(frozen_targets) == data["target_sha256"] and nodes == pool_ids(frozen_targets),
        "Exact frozen final200 IDs/order mismatch",
    )
    require(len(nodes) == len(set(nodes)) == 200, "Final200 count/uniqueness mismatch")
    pairs, targets, summary = calculate(data["runs"], nodes)
    require(len(pairs) == 2000 and len(targets) == 200, "Stability row count mismatch")
    require(not args.output_directory.exists(), "Refusing stability output overwrite")
    args.output_directory.mkdir(parents=True)
    for row in pairs:
        append_csv(args.output_directory / "stability_pairwise_jaccard.csv", row)
    for row in targets:
        append_csv(args.output_directory / "stability_per_target.csv", row)
    atomic_json(
        args.output_directory / "stability_summary.json",
        {
            **summary,
            "backbone": data["backbone"],
            "reporting_role": data["reporting_role"],
            "protocol_identity": data["protocol_identity"],
        },
        overwrite=False,
    )
