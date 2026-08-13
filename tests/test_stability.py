from __future__ import annotations

import pytest

from gnnexplainer_benchmark.evaluation.stability import (
    expected_pair_count,
    jaccard_similarity,
    pairwise_jaccard,
    pairwise_jaccard_rows,
    summarize_node_stability,
    summarize_pairwise_values,
    summarize_stability_rows,
)


def test_jaccard_similarity_uses_selected_edge_sets() -> None:
    assert jaccard_similarity({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert jaccard_similarity(set(), set()) == pytest.approx(1.0)
    assert jaccard_similarity({1, 2}, {1, 2}) == pytest.approx(1.0)
    assert jaccard_similarity({1, 2}, {3, 4}) == pytest.approx(0.0)


def test_pairwise_stability_across_three_seeds() -> None:
    values = pairwise_jaccard([{1, 2}, {2, 3}, {1, 2}])

    assert values == pytest.approx([1 / 3, 1.0, 1 / 3])


def test_node_stability_summary_reports_seed_and_pair_counts() -> None:
    row = summarize_node_stability(465, [{1, 2}, {2, 3}, {1, 2}])

    assert row["node_id"] == 465
    assert row["num_seeds"] == 3
    assert row["num_pairs"] == 3
    assert row["mean_pairwise_jaccard"] == pytest.approx(5 / 9)


def test_ten_seeds_produce_45_pairwise_jaccard_rows() -> None:
    seed_to_edges = {seed: {seed, seed + 1} for seed in range(10)}

    rows = pairwise_jaccard_rows(
        node_id=306,
        seed_to_edges=seed_to_edges,
        selected_edge_count=2,
    )

    assert expected_pair_count(10) == 45
    assert len(rows) == 45
    assert rows[0]["selected_edge_count"] == 2


def test_per_node_stability_equals_mean_of_pairwise_values() -> None:
    seed_to_edges = {
        0: {1, 2},
        1: {2, 3},
        2: {1, 2},
    }
    pair_rows = pairwise_jaccard_rows(
        node_id=306,
        seed_to_edges=seed_to_edges,
        selected_edge_count=2,
    )
    pair_values = [float(row["jaccard"]) for row in pair_rows]
    node_summary = summarize_node_stability(306, list(seed_to_edges.values()))

    assert node_summary["mean_pairwise_jaccard"] == pytest.approx(sum(pair_values) / 3)


def test_overall_stability_is_equal_weighted_mean_of_per_node_stability() -> None:
    rows = [
        {"status": "success", "mean_pairwise_jaccard": 0.2},
        {"status": "success", "mean_pairwise_jaccard": 0.6},
        {"status": "failed", "mean_pairwise_jaccard": ""},
    ]

    summary = summarize_stability_rows(rows)

    assert summary[0]["n"] == 2
    assert summary[0]["mean"] == pytest.approx(0.4)


def test_summary_values_report_distribution_statistics() -> None:
    summary = summarize_pairwise_values([0.2, 0.4, 0.6])

    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(0.4)
    assert summary["median"] == pytest.approx(0.4)
    assert summary["min"] == pytest.approx(0.2)
    assert summary["max"] == pytest.approx(0.6)
