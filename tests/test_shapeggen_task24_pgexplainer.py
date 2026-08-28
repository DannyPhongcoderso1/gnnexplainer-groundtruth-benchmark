from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from gnnexplainer_benchmark.data.serialization import sha256_file

CONFIG_PATH = Path("configs/evaluation/shapeggen_task24_pgexplainer.yaml")
FREEZE_PATH = Path("docs/shapeggen_pgexplainer_freeze.json")
REPORT_PATH = Path("docs/shapeggen_pgexplainer_report.md")


def read_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["task24"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_task24_config_locks_common_pool_and_protocol() -> None:
    config = read_config()

    assert config["common_train_count"] == 539
    assert config["common_val_count"] == 182
    assert config["final_target_count"] == 167
    assert sha256_file(Path(config["common_train_file"])) == config["common_train_sha256"]
    assert sha256_file(Path(config["common_val_file"])) == config["common_val_sha256"]
    assert sha256_file(Path(config["final_target_file"])) == config["final_target_sha256"]
    assert config["primary_seed"] == 42
    assert config["stability_seeds"] == [0, 1, 2, 3, 4]
    assert config["selected_edge_ratio"] == 0.2
    assert (
        config["edge_ranking_metric_scope"]
        == "continuous_scores_over_full_candidate_edge_set"
    )
    assert config["post_test_tuning"] is False


def test_task24_outputs_freeze_pgexplainer_common_pool() -> None:
    if not FREEZE_PATH.exists():
        pytest.skip("Task24 has not been generated yet")

    manifest = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    assert manifest["status"] == "success"
    assert all(manifest["acceptance"].values())
    assert manifest["fairness_audit"]["common_train_count"] == 539
    assert manifest["fairness_audit"]["common_val_count"] == 182
    assert manifest["fairness_audit"]["final_target_count"] == 167
    assert manifest["fairness_audit"]["train_final_disjoint"] is True
    assert manifest["fairness_audit"]["val_final_disjoint"] is True
    assert manifest["protocol"]["primary_seed"] == 42
    assert manifest["protocol"]["stability_seeds"] == [0, 1, 2, 3, 4]
    assert manifest["protocol"]["post_test_tuning"] is False

    for backbone in ("GCN", "GraphSAGE"):
        result = manifest["results"][backbone]
        assert result["primary"]["success_count"] == 167
        assert result["primary"]["failure_count"] == 0
        assert result["stability"]["seed_success_count"] == 835
        assert result["stability"]["seed_failure_count"] == 0
        assert result["stability"]["stability"]["pairwise_rows"] == 1670
        assert result["checkpoint_file_unchanged"] is True
        assert result["primary"]["training_manifest"]["epochs_completed"] == 30
        assert result["primary"]["training_manifest"]["checkpoint_reload_ok"] is True
        assert result["primary"]["training_manifest"]["model_weights_unchanged"] is True
        assert len(result["stability"]["training_manifests"]) == 5


def test_task24_metric_files_preserve_seed42_and_stability_units() -> None:
    if not FREEZE_PATH.exists():
        pytest.skip("Task24 has not been generated yet")

    for backbone in ("gcn", "graphsage"):
        primary = read_rows(
            Path(f"results/metrics/shapeggen/{backbone}/pgexplainer/per_node_metrics.csv")
        )
        stability = read_rows(
            Path(
                f"results/metrics/shapeggen/{backbone}/pgexplainer/"
                "stability_seed_metrics.csv"
            )
        )
        pairs = read_rows(
            Path(
                f"results/metrics/shapeggen/{backbone}/pgexplainer/"
                "stability_pairwise_jaccard.csv"
            )
        )

        assert len(primary) == 167
        assert {row["seed"] for row in primary} == {"42"}
        assert {row["status"] for row in primary} == {"success"}
        assert len(stability) == 835
        assert {row["seed"] for row in stability} == {"0", "1", "2", "3", "4"}
        assert {row["status"] for row in stability} == {"success"}
        assert len(pairs) == 1670


def test_task24_report_records_runtime_and_stop_boundary() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("Task24 report has not been generated yet")

    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "PGExplainer runtime is separated into upfront explainer training" in report
    assert "Task25 is not started automatically" in report
    assert "continuous scores over the full candidate-edge set" in report
