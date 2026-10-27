"""Model-only publication checks; synthetic unit inputs, no Elliptic dataset."""

import importlib.util
from pathlib import Path

import pytest
import torch

from scripts.elliptic_frozen_models import (
    checked_path,
    load_models,
    read_json,
    safe_checkpoint,
    target_ids,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def loaded():
    config = read_json(ROOT / "configs/evaluation/elliptic_frozen_publication.json")
    models, algorithms = load_models(ROOT, config)
    return config, models, algorithms


def test_all_14_frozen_identities_and_loads(loaded):
    config, models, algorithms = loaded
    assert len(config["models"]) == 14 and len(models) == 2 and len(algorithms) == 12
    assert {e["seed"] for e in config["models"] if e["family"] == "historical_seed42"} == {42}
    assert {e["seed"] for e in config["models"] if e["family"] == "fresh5"} == set(range(5))
    for model in models.values():
        assert not model.training and all(not p.requires_grad for p in model.parameters())
    for (backbone, _, _), algorithm in algorithms.items():
        assert not algorithm.mlp.training
        assert all(not p.requires_grad for p in algorithm.parameters())
        channels = 270 if backbone == "gcn" else 309
        before = {k: v.clone() for k, v in algorithm.state_dict().items()}
        with torch.no_grad():
            result = algorithm.mlp(torch.zeros(3, channels))
        assert result.shape == (3, 1) and torch.isfinite(result).all()
        assert all(torch.equal(v, before[k]) for k, v in algorithm.state_dict().items())


def test_exact_mit_architectures_match_existing_inference_wrappers(loaded):
    config, models, _ = loaded
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    features = torch.arange(4 * 165, dtype=torch.float32).reshape(4, 165) / 1000
    for backbone, filename, name in [("gcn", "GCN.py", "GCN"), ("graphsage", "SAGE.py", "SAGE")]:
        spec = importlib.util.spec_from_file_location(
            "licensed_" + backbone, ROOT / "external/elliptic_baseline/model" / filename
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        params = {
            k: v
            for k, v in config["architectures"][backbone].items()
            if k not in ["class", "cached"]
        }
        reference = getattr(module, name)(edges, **params, graphnorm=False, init_mode="default")
        state = next(
            e for e in config["models"] if e["family"] == "backbone" and e["backbone"] == backbone
        )
        reference.load_state_dict(safe_checkpoint(ROOT, state), strict=True)
        reference.eval().requires_grad_(False)
        with torch.no_grad():
            torch.testing.assert_close(
                reference(features, edges), models[backbone](features, edges), rtol=0, atol=0
            )


def test_checkpoint_metadata_and_targets_match_original_training(loaded):
    config, _, _ = loaded
    train = target_ids(ROOT, config["targets"]["pg_train"])
    validation = target_ids(ROOT, config["targets"]["pg_validation"])
    final = target_ids(ROOT, config["targets"]["final200"])
    assert len(train) == 2000 and len(validation) == 500 and len(final) == 200
    assert not (set(train) & set(final)) and not (set(validation) & set(final))
    for entry in config["models"]:
        if entry["family"] != "backbone":
            payload = safe_checkpoint(ROOT, entry)
            assert payload["train_node_ids"] == train
            assert payload["validation_node_ids"] == validation
            assert payload["epochs_completed"] == 30


def test_checkpoint_substitution_fails_before_unpickling(loaded):
    config, _, _ = loaded
    wrong = dict(config["models"][0], sha256="0" * 64)
    with pytest.raises(ValueError, match="mismatch"):
        checked_path(ROOT, wrong)


def test_existing_preprocessing_temporal_split_on_synthetic_fixture(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "licensed_dataset", ROOT / "external/elliptic_baseline/data/dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        ",".join(map(str, [i, step] + [0] * 165))
        for i, step in [(1, 29), (2, 30), (3, 40), (4, 41)]
    ]
    (tmp_path / "elliptic_txs_features.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "elliptic_txs_edgelist.csv").write_text("txId1,txId2\n1,2\n2,3\n3,4\n")
    (tmp_path / "elliptic_txs_classes.csv").write_text("txId,class\n1,2\n2,1\n3,2\n4,unknown\n")
    data = module.BCDataset(type="elliptic", path=str(tmp_path)).to_torch_data()
    assert data.x.shape == (4, 165) and data.edge_index.shape == (2, 6)
    assert data.y.tolist() == [0, 1, 0, 2]
    assert data.train_mask.tolist() == [True, False, False, False]
    assert data.val_mask.tolist() == [False, True, False, False]
    assert data.test_mask.tolist() == [False, False, True, False]
