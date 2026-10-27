from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml
from torch_geometric.data import Data

from gnnexplainer_benchmark.data.pgexplainer_targets import (
    PGExplainerTarget,
    select_pgexplainer_targets,
    split_mask,
    validate_pgexplainer_target_splits,
)
from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.evaluation.budget import selected_edge_count_for_ratio
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)
from scripts.select_pgexplainer_targets import target_rows_match


def make_chain_data() -> Data:
    return Data(
        x=torch.ones((6, 2)),
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
        y=torch.tensor([0, 1, 1, 2, 2, 1]),
        node_mask=torch.tensor([False, True, True, True, True, False]),
        train_mask=torch.tensor([True, True, True, False, False, False]),
        val_mask=torch.tensor([False, False, False, True, True, False]),
        test_mask=torch.tensor([False, False, False, False, False, True]),
        edge_mask=torch.tensor([False, True, True, True, False]),
    )


def test_split_mask_maps_validation_to_val_mask() -> None:
    data = make_chain_data()

    assert torch.equal(split_mask(data, "validation"), data.val_mask)
    assert torch.equal(split_mask(data, "train"), data.train_mask)


def test_pgexplainer_targets_use_frozen_prediction_as_target_not_label_default() -> None:
    data = make_chain_data()
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ]
    )

    targets = select_pgexplainer_targets(logits=logits, data=data, split="train", num_hops=1)

    assert [target.node_id for target in targets] == [1, 2]
    assert all(
        target.target_for_pgexplainer == target.frozen_model_prediction for target in targets
    )
    assert all(target.ground_truth_label == target.frozen_model_prediction for target in targets)


def test_pgexplainer_target_selection_excludes_frozen_test_ids() -> None:
    data = make_chain_data()
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
        ]
    )

    targets = select_pgexplainer_targets(
        logits=logits,
        data=data,
        split="validation",
        num_hops=1,
        excluded_node_ids={4},
    )

    assert [target.node_id for target in targets] == [3]


def test_pgexplainer_target_splits_reject_test_leakage() -> None:
    target = PGExplainerTarget(
        node_id=10,
        ground_truth_label=1,
        frozen_model_prediction=1,
        target_for_pgexplainer=1,
        confidence=0.8,
        base_model_split="train",
        is_ground_truth_motif=True,
        gcn_prediction_correct=True,
        candidate_edge_count=5,
        candidate_edge_mapping_valid=True,
    )

    with pytest.raises(ValueError, match="train/test leakage"):
        validate_pgexplainer_target_splits([target], [], {10})


def test_pgexplainer_protocol_keeps_frozen_twenty_percent_budget() -> None:
    assert selected_edge_count_for_ratio(24, 0.2) == 5
    assert selected_edge_count_for_ratio(866, 0.2) == 174


def read_target_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def node_ids(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["node_id"]) for row in rows]


def assert_deterministic_unique_order(rows: list[dict[str, str]]) -> None:
    ids = node_ids(rows)
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_canonical_pgexplainer_target_files_are_frozen_and_disjoint() -> None:
    train_rows = read_target_rows(Path("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv"))
    val_rows = read_target_rows(Path("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv"))
    test_rows = read_target_rows(Path("results/metrics/explainer_test_targets.csv"))

    assert len(train_rows) == 48
    assert len(val_rows) == 16
    assert len(test_rows) == 12
    assert_deterministic_unique_order(train_rows)
    assert_deterministic_unique_order(val_rows)
    assert_deterministic_unique_order(test_rows)

    train_ids = set(node_ids(train_rows))
    val_ids = set(node_ids(val_rows))
    test_ids = set(node_ids(test_rows))
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_pgexplainer_config_points_to_canonical_train_and_validation_targets() -> None:
    config = yaml.safe_load(Path("configs/ba_house/ba500/pgexplainer_gcn.yaml").read_text())
    evaluation = config["evaluation"]

    assert evaluation["train_targets"] == "configs/ba_house/ba500/targets/pgexplainer_train_targets.csv"
    assert evaluation["validation_targets"] == "configs/ba_house/ba500/targets/pgexplainer_val_targets.csv"
    assert evaluation["test_targets"] == "results/metrics/explainer_test_targets.csv"


def test_selector_reproduces_canonical_pgexplainer_targets() -> None:
    data = load_data_artifact(Path("data/ba_house/ba500/graph.pt"))
    checkpoint = load_checkpoint(Path("artifacts/checkpoints/ba_house/ba500/backbones/gcn/seed42.pt"))
    model = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    test_ids = set(node_ids(read_target_rows(Path("results/metrics/explainer_test_targets.csv"))))
    num_hops = int(checkpoint["resolved_config"]["model"]["num_layers"])
    regenerated_train = select_pgexplainer_targets(
        logits=logits,
        data=data,
        split="train",
        num_hops=num_hops,
        excluded_node_ids=test_ids,
    )
    regenerated_val = select_pgexplainer_targets(
        logits=logits,
        data=data,
        split="validation",
        num_hops=num_hops,
        excluded_node_ids=test_ids,
    )

    assert [target.node_id for target in regenerated_train] == node_ids(
        read_target_rows(Path("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv"))
    )
    assert [target.node_id for target in regenerated_val] == node_ids(
        read_target_rows(Path("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv"))
    )


def test_target_row_match_allows_small_confidence_float_drift(tmp_path: Path) -> None:
    header = (
        "node_id,ground_truth_label,frozen_model_prediction,target_for_pgexplainer,"
        "confidence,base_model_split,is_ground_truth_motif,gcn_prediction_correct,"
        "candidate_edge_count,candidate_edge_mapping_valid\n"
    )
    canonical = tmp_path / "canonical.csv"
    regenerated = tmp_path / "regenerated.csv"
    canonical.write_text(header + "1,1,1,1,0.2722652554512024,train,True,True,28,True\n")
    regenerated.write_text(header + "1,1,1,1,0.2722652852535248,train,True,True,28,True\n")

    assert target_rows_match(canonical, regenerated)
