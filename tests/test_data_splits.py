from __future__ import annotations

from pathlib import Path

from gnnexplainer_benchmark.data.ba_house import generate_ba_house_data, load_ba_house_config
from gnnexplainer_benchmark.data.validation import validate_splits


def test_data_splits_do_not_overlap_and_cover_all_nodes() -> None:
    config = load_ba_house_config(Path("configs/dataset/ba_house.yaml"))

    data = generate_ba_house_data(config)

    validate_splits(data.train_mask, data.val_mask, data.test_mask, data.num_nodes)
    assert int((data.train_mask | data.val_mask | data.test_mask).sum().item()) == data.num_nodes
    assert not bool((data.train_mask & data.val_mask).any())
    assert not bool((data.train_mask & data.test_mask).any())
    assert not bool((data.val_mask & data.test_mask).any())
