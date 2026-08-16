from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.run_pgexplainer_gcn_stability import compute_stability_rows, summarize_values


def write_selected(path: Path, edges: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["global_edge_index"])
        writer.writeheader()
        for edge in edges:
            writer.writerow({"global_edge_index": edge})


def test_pgexplainer_stability_jaccard_uses_ten_seed_pairs_per_node(tmp_path: Path) -> None:
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        write_selected(tmp_path / f"seed_{seed}" / "node_10" / "selected_edges.csv", [1, 2, seed])

    pairwise_rows, per_node_rows, summary = compute_stability_rows(tmp_path, seeds)

    assert len(pairwise_rows) == 10
    assert per_node_rows[0]["pair_count"] == 10
    assert summary["pairwise_comparisons_per_node"] == 10
    assert summary["node_count"] == 1


def test_pgexplainer_stability_runtime_summary_is_distributional() -> None:
    summary = summarize_values([1.0, 2.0, 3.0])

    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(3.0)
