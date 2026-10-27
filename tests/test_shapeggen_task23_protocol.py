from __future__ import annotations

import csv
import json
from math import ceil
from pathlib import Path

import pytest
import yaml

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.data.shapeggen import (
    get_target_candidate_edge_mask,
    get_target_edge_ground_truth,
)
from gnnexplainer_benchmark.evaluation.budget import selected_edge_count_for_ratio

TASK23_CONFIG = Path("configs/shapeggen/v1/shapeggen_task23.yaml")
FREEZE_PATH = Path("docs/shapeggen_gnnexplainer_random_freeze.json")
REPORT_PATH = Path("docs/provenance/historical_reports/shapeggen_gnnexplainer_random_report.md")


def read_task23_config() -> dict:
    config = yaml.safe_load(TASK23_CONFIG.read_text(encoding="utf-8"))
    return config["task23"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_task23_config_locks_frozen_inputs_and_budget() -> None:
    config = read_task23_config()

    assert config["target_count"] == 167
    assert config["selected_edge_ratio"] == 0.2
    assert config["primary_seed"] == 42
    assert config["gnnexplainer_stability_seeds"] == [0, 1, 2, 3, 4]
    assert config["random_baseline_seeds"] == list(range(10))
    assert (
        config["edge_ranking_metric_scope"]
        == "continuous_scores_over_full_candidate_edge_set"
    )
    assert sha256_file(Path(config["target_file"])) == config["target_sha256"]
    assert sha256_file(Path(config["dataset_path"])) == config["dataset_sha256"]
    assert sha256_file(Path(config["gcn_checkpoint"])) == config["gcn_checkpoint_sha256"]
    assert (
        sha256_file(Path(config["graphsage_checkpoint"]))
        == config["graphsage_checkpoint_sha256"]
    )


def test_task23_target_file_matches_shapeggen_candidate_masks() -> None:
    config = read_task23_config()
    data = load_data_artifact(Path(config["dataset_path"]))
    rows = read_csv_rows(Path(config["target_file"]))
    node_ids = [int(row["node_id"]) for row in rows]

    assert len(rows) == config["target_count"]
    assert node_ids == sorted(node_ids)
    assert len(node_ids) == len(set(node_ids))

    for row in rows:
        node_id = int(row["node_id"])
        candidate_mask = get_target_candidate_edge_mask(data, node_id)
        gt_mask = get_target_edge_ground_truth(data, node_id)
        candidate_count = int(candidate_mask.sum().item())
        positive_count = int(gt_mask[candidate_mask].sum().item())
        negative_count = candidate_count - positive_count
        selected_count = selected_edge_count_for_ratio(
            candidate_count,
            config["selected_edge_ratio"],
        )

        assert row["split"] == "test"
        assert row["candidate_edge_mapping_valid"] == "True"
        assert candidate_mask.dtype == gt_mask.dtype
        assert not bool((gt_mask & ~candidate_mask).any())
        assert candidate_count == int(row["candidate_edge_count"])
        assert positive_count == int(row["positive_gt_edges"])
        assert negative_count == int(row["negative_candidate_edges"])
        assert selected_count == max(1, ceil(candidate_count * 0.2))


def test_task23_outputs_lock_schema_after_freeze() -> None:
    if not FREEZE_PATH.exists() or not REPORT_PATH.exists():
        pytest.skip("Task23 outputs have not been generated yet")

    manifest = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert manifest["status"] == "success"
    assert all(manifest["acceptance"].values())
    assert manifest["protocol"]["selected_edge_ratio"] == 0.2
    assert "continuous_scores_over_full_candidate_edge_set" in report
    assert "GNNExplainer runtime is per-target explanation optimization" in report
    assert "Random runtime is random edge selection plus metric evaluation" in report

    rows = manifest["comparison_summary"]["rows"]
    assert {(row["backbone"], row["method"]) for row in rows} == {
        ("GCN", "GNNExplainer"),
        ("GCN", "Random"),
        ("GraphSAGE", "GNNExplainer"),
        ("GraphSAGE", "Random"),
    }


def test_task23_metric_files_preserve_node_and_seed_units_after_freeze() -> None:
    if not FREEZE_PATH.exists():
        pytest.skip("Task23 outputs have not been generated yet")

    config = read_task23_config()
    target_count = config["target_count"]

    for backbone in ("gcn", "graphsage"):
        gnn_rows = read_csv_rows(
            Path(f"results/metrics/shapeggen/{backbone}/gnnexplainer/per_node_metrics.csv")
        )
        random_seed_rows = read_csv_rows(
            Path(f"results/metrics/shapeggen/{backbone}/random/random_seed_level_metrics.csv")
        )
        random_node_rows = read_csv_rows(
            Path(f"results/metrics/shapeggen/{backbone}/random/random_node_level_metrics.csv")
        )

        assert len(gnn_rows) == target_count
        assert len(random_seed_rows) == target_count * 10
        assert len(random_node_rows) == target_count
        assert {row["seed"] for row in random_node_rows} == {"node_mean"}

        for row in gnn_rows + random_node_rows:
            candidate_count = int(float(row["candidate_edge_count"]))
            selected_count = int(float(row["selected_edge_count"]))
            assert selected_count == selected_edge_count_for_ratio(candidate_count, 0.2)
            assert row["status"] == "success"
