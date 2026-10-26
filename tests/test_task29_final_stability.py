"""Evaluation-only unit tests; no model training."""

import importlib.util
import io
import json
import statistics
import unittest
import zipfile
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location(
    "task29_final",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_task29_final_stability.py",
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def archive(member="x", corrupt=False):
    content = b"fixture"
    manifest = {
        "files": {
            member: {
                "size_bytes": len(content),
                "sha256": "0" * 64 if corrupt else m.digest(content),
            }
        }
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr(member, content)
        z.writestr("ARCHIVE_MANIFEST.json", json.dumps(manifest))
    return stream.getvalue()


class FinalStabilityTests(unittest.TestCase):
    def test_archive_safe(self):
        self.assertEqual(m.checked_archive(archive())[0].read("x"), b"fixture")

    def test_archive_traversal(self):
        with self.assertRaises(ValueError):
            m.checked_archive(archive("../x"))

    def test_archive_member_hash(self):
        with self.assertRaises(ValueError):
            m.checked_archive(archive(corrupt=True))

    def test_archive_outer_hash(self):
        with self.assertRaises(ValueError):
            m.checked_archive(archive(), "0" * 64)

    def test_jaccard(self):
        self.assertEqual(m.jaccard([1, 2], [2, 3]), 1 / 3)
        self.assertEqual(m.jaccard([1], [1]), 1)

    def test_empty_jaccard_rejected(self):
        with self.assertRaises(ValueError):
            m.jaccard([], [1])

    def test_topk_cardinality(self):
        ids, d = m.selection(torch.arange(11, dtype=torch.float32), torch)
        self.assertEqual(len(set(ids.tolist())), 3)
        self.assertFalse(d["boundary_tie_crosses_k"])

    def test_exact_boundary_tie_diagnosed_without_new_policy(self):
        _, d = m.selection(torch.ones(10, dtype=torch.float32), torch)
        self.assertTrue(d["boundary_tie_crosses_k"])
        self.assertEqual(d["unique_raw_float32"], 1)

    def test_tie_away_from_boundary(self):
        _, d = m.selection(torch.tensor([9.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), torch)
        self.assertFalse(d["boundary_tie_crosses_k"])

    def test_aggregation_statistical_unit(self):
        result = m.aggregate([{"mean_pairwise_jaccard": v} for v in (0.0, 0.5, 1.0)])
        self.assertEqual(result["mean_jaccard"], 0.5)
        self.assertEqual(result["sample_sd"], statistics.stdev((0.0, 0.5, 1.0)))

    def test_cpu_replay(self):
        raw = torch.arange(21, dtype=torch.float32)
        self.assertTrue(torch.equal(m.selection(raw, torch)[0], m.selection(raw, torch)[0]))

    def test_canonical_ids_and_independent_membership(self):
        candidates = [2, 5, 9, 10]
        a, b = [2, 9], [5, 9]
        x = [e in a for e in candidates]
        y = [e in b for e in candidates]
        self.assertEqual(
            sum(p and q for p, q in zip(x, y)) / sum(p or q for p, q in zip(x, y)), m.jaccard(a, b)
        )

    def test_cached_seed_matrix_rejects_duplicate(self):
        with self.assertRaises(ValueError):
            m.validate_cached_record({"identity": "x", "runs": [{"seed": 0}] * 5}, "x")

    def test_cache_never_admits_boundary_tie(self):
        record = {
            "identity": "x",
            "runs": [
                {"seed": s, "diagnostics": {"boundary_tie_crosses_k": s == 3}} for s in range(5)
            ],
        }
        with self.assertRaisesRegex(ValueError, "BOUNDARY_TIE"):
            m.validate_cached_record(record, "x")

    def test_cached_configuration_mismatch(self):
        with self.assertRaisesRegex(ValueError, "identity changed"):
            m.validate_cached_record({"identity": "x"}, "y")


if __name__ == "__main__":
    unittest.main()
