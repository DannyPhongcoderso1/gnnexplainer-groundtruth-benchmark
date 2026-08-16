from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def node_ids(path: str) -> set[int]:
    return {int(row["node_id"]) for row in read_rows(path)}


def test_common_pool_is_deterministic_intersection() -> None:
    gcn_train = node_ids("configs/targets/pgexplainer_train_targets.csv")
    sage_train = node_ids("configs/targets/pgexplainer_graphsage_train_targets.csv")
    common_train = node_ids("configs/targets/ba_house_pg_common_train_targets.csv")
    gcn_val = node_ids("configs/targets/pgexplainer_val_targets.csv")
    sage_val = node_ids("configs/targets/pgexplainer_graphsage_val_targets.csv")
    common_val = node_ids("configs/targets/ba_house_pg_common_val_targets.csv")

    assert common_train == gcn_train & sage_train
    assert common_val == gcn_val & sage_val
    assert len(common_train) == 48
    assert len(common_val) == 16


def test_common_pool_disjoint_from_final_targets_and_reuses_gcn_pool() -> None:
    common_train = node_ids("configs/targets/ba_house_pg_common_train_targets.csv")
    common_val = node_ids("configs/targets/ba_house_pg_common_val_targets.csv")
    final_targets = node_ids("configs/targets/ba_house_2x2_targets.csv")

    assert not (common_train & common_val)
    assert not (common_train & final_targets)
    assert not (common_val & final_targets)
    assert common_train == node_ids("configs/targets/pgexplainer_train_targets.csv")
    assert common_val == node_ids("configs/targets/pgexplainer_val_targets.csv")


def test_graphsage_common_config_uses_common_files_and_isolated_checkpoint() -> None:
    config = yaml.safe_load(
        Path("configs/evaluation/pgexplainer_graphsage_common_pool.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert (
        config["evaluation"]["train_targets"]
        == "configs/targets/ba_house_pg_common_train_targets.csv"
    )
    assert (
        config["evaluation"]["validation_targets"]
        == "configs/targets/ba_house_pg_common_val_targets.csv"
    )

    natural_path = Path("artifacts/explainers/pgexplainer_graphsage_ba_house_seed42.pt")
    common_path = Path("artifacts/explainers/pgexplainer_graphsage_ba_house_common_seed42.pt")
    assert natural_path != common_path
    assert natural_path.exists()
    assert common_path.exists()


def test_graphsage_common_training_preserved_backbone_weights() -> None:
    manifest = json.loads(
        Path(
            "artifacts/explainers/"
            "pgexplainer_graphsage_ba_house_common_seed42_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["model_weights_unchanged"] is True
    assert manifest["optimizer_excludes_gcn_parameters"] is True
    assert manifest["final_test_targets_used_for_training"] == 0
    assert manifest["epochs_completed"] == 30
