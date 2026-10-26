"""Independent standard-library readback: Python exact-score sort and two-pointer overlap."""

import csv
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/audit/elliptic_pg_stability/task29_final_evaluation_tie_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def intersection_sorted(a, b):
    i = j = count = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            count += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return count


def verify():
    evidence = json.loads((OUT / "TASK29_FINAL_EVALUATION_INTEGRITY.json").read_text())
    results = json.loads((OUT / "TASK29_FINAL_STABILITY_RESULTS.json").read_text())
    policy = evidence["evaluation_policy_identity"]
    with (OUT / "TASK29_FINAL_STABILITY_SUMMARY.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == 2
    for row in summaries:
        headline = next(r for r in results["results"] if r["backbone"] == row["backbone"])
        assert row["evaluation_policy_identity"] == policy
        assert float(row["mean_jaccard"]) == headline["mean_jaccard"]
        assert float(row["sample_sd"]) == headline["sample_sd"]
    with (OUT / "TASK29_FINAL_STABILITY_PAIRWISE.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    with (OUT / "TASK29_FINAL_STABILITY_PER_TARGET.csv").open() as stream:
        targets = list(csv.DictReader(stream))
    assert len(pairs) == 4000 and len(targets) == 400
    pmap = {
        (r["backbone"], int(r["node_id"]), int(r["seed_a"]), int(r["seed_b"])): r for r in pairs
    }
    tmap = {(r["backbone"], int(r["node_id"])): r for r in targets}
    assert len(pmap) == 4000 and len(tmap) == 400
    assert all(r["evaluation_policy_identity"] == policy for r in pairs + targets)
    max_delta = 0.0
    recomputed = {}
    for backbone in ("gcn", "graphsage"):
        means = []
        for node in evidence["target_ids"]:
            path = OUT / "records" / backbone / f"{node}.json"
            record = json.loads(path.read_text())
            receipt = json.loads(path.with_suffix(".receipt.json").read_text())
            assert sha(path) == receipt["sha256"]
            assert (
                record["evaluation_policy_identity"]
                == receipt["evaluation_policy_identity"]
                == policy
            )
            ids = record["candidate_global_edge_indices"]
            assert ids == sorted(set(ids))
            assert all(type(edge) is int and 0 <= edge < 468710 for edge in ids)
            k = max(1, math.ceil(0.2 * len(ids)))
            rows = {r["seed"]: r for r in record["runs"]}
            assert sorted(rows) == list(range(5)) and len(record["runs"]) == 5
            for row in rows.values():
                raw = row["raw_logits"]
                assert len(raw) == len(ids)
                assert all(
                    math.isfinite(x) and struct.unpack("<f", struct.pack("<f", x))[0] == x
                    for x in raw
                )
                primary = sorted(
                    ids[i] for i in sorted(range(len(ids)), key=lambda i: (-raw[i], ids[i]))[:k]
                )
                alternative = sorted(
                    ids[i] for i in sorted(range(len(ids)), key=lambda i: (-raw[i], -ids[i]))[:k]
                )
                assert primary == row["selected_global_edge_indices"]
                assert alternative == row["alternative_selected_global_edge_indices"]
                assert len(primary) == len(set(primary)) == k
            values = []
            for a in range(5):
                for b in range(a + 1, 5):
                    left, right = (
                        rows[a]["selected_global_edge_indices"],
                        rows[b]["selected_global_edge_indices"],
                    )
                    intersection = intersection_sorted(left, right)
                    union = len(left) + len(right) - intersection
                    value = intersection / union
                    official = pmap[backbone, node, a, b]
                    assert (
                        int(official["intersection"]) == intersection
                        and int(official["union"]) == union
                    )
                    max_delta = max(max_delta, abs(value - float(official["jaccard"])))
                    assert abs(value - float(official["jaccard"])) <= 1e-14
                    values.append(value)
            mean = math.fsum(values) / 10
            assert abs(mean - float(tmap[backbone, node]["mean_pairwise_jaccard"])) <= 1e-14
            means.append(mean)
        overall = math.fsum(means) / 200
        sd = math.sqrt(math.fsum((x - overall) ** 2 for x in means) / 199)
        headline = next(r for r in results["results"] if r["backbone"] == backbone)
        assert abs(overall - headline["mean_jaccard"]) <= 1e-14
        assert abs(sd - headline["sample_sd"]) <= 1e-14
        recomputed[backbone] = {"mean_jaccard": overall, "sample_sd": sd}
    expected = {**evidence["old_evaluation_snapshot_before"], **evidence["protected_files_before"]}
    assert all(sha(path) == h for path, h in expected.items())
    return {
        "evaluation_policy_identity": policy,
        "status": "PASS",
        "pair_rows": len(pairs),
        "target_means": len(targets),
        "ranked_vectors": 2000,
        "method": "Python exact stored-float32 scalar tuple sort; sorted-ID two-pointer intersection; fsum mean/two-pass SD",
        "tolerance_absolute": 1e-14,
        "max_pair_difference": max_delta,
        "results": recomputed,
        "original_inputs_and_evidence_unchanged": len(expected),
        "result_file_sha256": sha(OUT / "TASK29_FINAL_STABILITY_RESULTS.json"),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
