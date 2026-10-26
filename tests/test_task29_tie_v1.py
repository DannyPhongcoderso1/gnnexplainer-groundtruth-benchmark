"""Approved exact-float32 lexicographic ranking, no model construction."""

import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = importlib.import_module("evaluate_task29_tie_v1")


class TieV1Tests(unittest.TestCase):
    def rank(self, scores, ids=None, alternative=False):
        scores = np.asarray(scores, dtype=np.float32)
        return m.rank(scores, list(range(len(scores))) if ids is None else ids, alternative)[0]

    def test_strict_ranking(self):
        self.assertEqual(self.rank([3, 1, 2, 5, 4]), [3])

    def test_ties_inside_topk(self):
        self.assertEqual(self.rank([9, 9, 7, 6, 5, 4, 3, 2, 1, 0]), [0, 1])

    def test_ties_outside_topk(self):
        self.assertEqual(self.rank([10, 9, 7, 7, 5, 4, 3, 2, 1, 0]), [0, 1])

    def test_crossing_boundary(self):
        scores = [10, 9, 9, 6, 5, 4, 3, 2, 1, 0]
        self.assertEqual(self.rank(scores, [90, 17, 13, 4, 5, 6, 7, 8, 9, 10]), [13, 90])

    def test_all_equal_nonconsecutive_ids(self):
        self.assertEqual(self.rank([1] * 10, [99, 77, 55, 33, 11, 9, 7, 5, 3, 1]), [1, 3])

    def test_alternative_not_official(self):
        self.assertEqual(self.rank([1] * 10, alternative=True), [8, 9])

    def test_shuffled_candidate_enumeration(self):
        raw = np.array([4, 3, 3, 2, 1, 0, -1, -2, -3, -4], dtype=np.float32)
        ids = np.array([20, 15, 13, 100, 7, 3, 51, 8, 41, 0])
        shuffle = np.array([8, 4, 2, 9, 1, 7, 3, 0, 6, 5])
        self.assertEqual(m.rank(raw, ids)[0], m.rank(raw[shuffle], ids[shuffle])[0])

    def test_cardinality_unique(self):
        for n in (1, 2, 5, 6, 17, 180):
            chosen = self.rank(np.ones(n))
            self.assertEqual(len(chosen), max(1, math.ceil(0.2 * n)))
            self.assertEqual(len(chosen), len(set(chosen)))

    def test_repeated_ranking(self):
        self.assertEqual(self.rank([2, 1, 2, 1, 0]), self.rank([2, 1, 2, 1, 0]))

    def test_strict_boundary_original_topk_preserved(self):
        import torch

        raw = torch.arange(23, dtype=torch.float32)
        self.assertEqual(self.rank(raw.numpy()), sorted(torch.topk(raw, 5).indices.tolist()))

    def test_no_float64_input(self):
        with self.assertRaises(ValueError):
            m.rank(np.ones(5, dtype=np.float64), list(range(5)))

    def test_raw_values_not_modified(self):
        raw = np.array([-223.89364624023438, -223.89364624023438, 1, 2, 3], dtype=np.float32)
        before = raw.tobytes()
        m.rank(raw, [417344, 417302, 1, 2, 3])
        self.assertEqual(before, raw.tobytes())

    def test_scores_always_dominate_ids(self):
        next_value = np.nextafter(np.float32(1), np.float32(2))
        self.assertEqual(self.rank([1, next_value, 0, 0, 0], [0, 999, 1, 2, 3]), [999])

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            m.rank(np.ones(5, dtype=np.float32), [1] * 5)

    def test_known_blocker(self):
        scores = [0] * 35 + [-223.89364624023438] * 2 + [-500] * 143
        ids = list(range(35)) + [417344, 417302] + list(range(100, 243))
        chosen = self.rank(scores, ids)
        self.assertIn(417302, chosen)
        self.assertNotIn(417344, chosen)


if __name__ == "__main__":
    unittest.main()
