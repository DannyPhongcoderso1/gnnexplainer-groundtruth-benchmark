from __future__ import annotations

from itertools import combinations
from statistics import mean, median, stdev


def expected_pair_count(num_seeds: int) -> int:
    if num_seeds < 0:
        raise ValueError("num_seeds must be non-negative")
    return num_seeds * (num_seeds - 1) // 2


def jaccard_similarity(first: set[int], second: set[int]) -> float:
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def pairwise_jaccard(edge_sets: list[set[int]]) -> list[float]:
    if len(edge_sets) < 2:
        return []
    return [jaccard_similarity(first, second) for first, second in combinations(edge_sets, 2)]


def pairwise_jaccard_rows(
    *,
    node_id: int,
    seed_to_edges: dict[int, set[int]],
    selected_edge_count: int,
) -> list[dict[str, float | int]]:
    rows = []
    for seed_a, seed_b in combinations(sorted(seed_to_edges), 2):
        first = seed_to_edges[seed_a]
        second = seed_to_edges[seed_b]
        intersection_count = len(first & second)
        union_count = len(first | second)
        rows.append(
            {
                "node_id": node_id,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "selected_edge_count": selected_edge_count,
                "intersection_count": intersection_count,
                "union_count": union_count,
                "jaccard": jaccard_similarity(first, second),
            }
        )
    return rows


def summarize_node_stability(node_id: int, edge_sets: list[set[int]]) -> dict[str, float | int]:
    values = pairwise_jaccard(edge_sets)
    return {
        "node_id": node_id,
        "mean_pairwise_jaccard": mean(values) if values else None,
        "std_pairwise_jaccard": stdev(values) if len(values) > 1 else 0.0 if values else None,
        "median_pairwise_jaccard": median(values) if values else None,
        "min_pairwise_jaccard": min(values) if values else None,
        "max_pairwise_jaccard": max(values) if values else None,
        "num_seeds": len(edge_sets),
        "num_pairs": len(values),
    }


def summarize_pairwise_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_stability_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = [
        float(row["mean_pairwise_jaccard"])
        for row in rows
        if row.get("status") == "success" and row.get("mean_pairwise_jaccard") not in (None, "")
    ]
    summary = summarize_pairwise_values(values)
    return [
        {
            "metric": "mean_pairwise_jaccard",
            **summary,
        }
    ]
