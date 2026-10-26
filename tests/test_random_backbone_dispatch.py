from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml

from gnnexplainer_benchmark.data.serialization import load_data_artifact, sha256_file
from gnnexplainer_benchmark.evaluation.budget import select_top_ratio_edges
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.explainers.random_baseline import random_candidate_edge_scores
from gnnexplainer_benchmark.models import GCN, GraphSAGE
from gnnexplainer_benchmark.training.checkpointing import build_gcn_from_checkpoint, load_checkpoint
from scripts.run_random_baseline import build_backbone_from_checkpoint, parse_args, seed_for_node

TARGETS = [306, 315, 316, 355, 365, 375, 376, 411, 445, 465, 486, 491]


def checkpoint(backbone):
    return load_checkpoint(Path(f"artifacts/checkpoints/ba_house_{backbone}_best.pt"))


@pytest.mark.parametrize("backbone,cls", [("gcn", GCN), ("graphsage", GraphSAGE)])
def test_dispatch_and_exact_checkpoint_state(backbone, cls):
    cp = checkpoint(backbone)
    model = build_backbone_from_checkpoint(cp, backbone=backbone, in_channels=10, out_channels=4)
    assert isinstance(model, cls)
    assert all(torch.equal(v, cp["model_state_dict"][k]) for k, v in model.state_dict().items())
    if backbone == "graphsage":
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("backbone", ["unknown", "gcn"])
def test_invalid_or_mismatched_backbone_rejected(backbone):
    with pytest.raises(ValueError):
        build_backbone_from_checkpoint(
            checkpoint("graphsage"), backbone=backbone, in_channels=10, out_channels=4
        )


def test_gcn_constructor_and_defaults_unchanged():
    cp = checkpoint("gcn")
    old = build_gcn_from_checkpoint(cp, 10, 4).eval()
    new = build_backbone_from_checkpoint(cp, backbone="gcn", in_channels=10, out_channels=4).eval()
    data = load_data_artifact(Path("artifacts/data/ba_house.pt"))
    with torch.no_grad():
        assert torch.equal(old(data.x, data.edge_index), new(data.x, data.edge_index))
    args = parse_args([])
    assert args.backbone == "gcn"
    assert args.output_root == Path("results/explanations/phase4d_random_baseline")


def test_namespace_separation_and_invalid_cli():
    gcn, sage = parse_args([]), parse_args(["--backbone", "graphsage"])
    for key in ("checkpoint", "output_root", "metrics_output_dir", "comparison_output"):
        assert getattr(gcn, key) != getattr(sage, key)
    with pytest.raises(SystemExit):
        parse_args(["--backbone", "invalid"])
    with pytest.raises(ValueError):
        parse_args(["--backbone", "graphsage", "--output-root", "results/explanations"])
    with pytest.raises(ValueError):
        parse_args(["--backbone", "graphsage", "--output-root", str(gcn.output_root)])
    explicit = parse_args(["--backbone", "graphsage", "--checkpoint", str(gcn.checkpoint)])
    assert explicit.checkpoint == gcn.checkpoint


def test_frozen_protocol_parity_and_model_immutability():
    args = parse_args(["--backbone", "graphsage"])
    assert sha256_file(args.checkpoint) == (
        "09992c5ac4d2cd80c10792c12be5f945b3a20d749d8433008eba9cf3694135eb"
    )
    assert sha256_file(args.target_file) == (
        "2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f"
    )
    with args.target_file.open() as handle:
        targets = list(csv.DictReader(handle))
    assert [int(row["node_id"]) for row in targets] == TARGETS
    cfg = yaml.safe_load(args.phase4d_config.read_text())["phase4d"]
    assert cfg["random_baseline_seeds"] == list(range(10))
    assert cfg["selected_edge_ratio"] == 0.2
    data = load_data_artifact(args.artifact)
    cp = checkpoint("graphsage")
    model = build_backbone_from_checkpoint(cp, backbone="graphsage", in_channels=10, out_channels=4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).argmax(-1)
    with Path("results/metrics/model_target_parity_gcn_graphsage.csv").open() as handle:
        parity = {int(row["node_id"]): row for row in csv.DictReader(handle)}
    for row in targets:
        node = int(row["node_id"])
        assert int(predictions[node]) == int(row["label"]) == int(row["prediction"])
        assert int(predictions[node]) == int(parity[node]["graphsage_prediction"])
        assert bool(data.test_mask[node])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_random_generator_matches_saved_gcn_scores_without_gcn_evaluation():
    data = load_data_artifact(Path("artifacts/data/ba_house.pt"))
    candidate = computation_subgraph_edge_mask(data, 306, 3)
    scores = random_candidate_edge_scores(
        edge_count=data.edge_index.size(1),
        candidate_edge_mask=candidate,
        seed=seed_for_node(0, 306),
    )
    saved = torch.load(
        "results/explanations/phase4d_random_baseline/seed_0/node_306/edge_mask.pt",
        weights_only=True,
    )
    assert torch.equal(scores, saved)
    selected, budget = select_top_ratio_edges(scores, candidate, 0.2)
    assert budget.selected_edge_count == 66
    assert int(selected.sum()) == 66
    assert not bool((selected & ~candidate).any())
