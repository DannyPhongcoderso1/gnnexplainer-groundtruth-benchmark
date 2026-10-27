from __future__ import annotations

import csv
from pathlib import Path

from gnnexplainer_benchmark.explainers.pgexplainer_smoke import run_pgexplainer_smoke


def test_canonical_targets_store_frozen_prediction_as_pgexplainer_target() -> None:
    for path in [
        Path("configs/ba_house/ba500/targets/pgexplainer_train_targets.csv"),
        Path("configs/ba_house/ba500/targets/pgexplainer_val_targets.csv"),
    ]:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            assert row["target_for_pgexplainer"] == row["frozen_model_prediction"]
            assert "ground_truth_label" in row


def test_pgexplainer_smoke_target_is_frozen_prediction_without_grad() -> None:
    result = run_pgexplainer_smoke(output_path=None, device="cpu")

    assert result.target_matches_frozen_prediction is True
    assert result.target_requires_grad is False
    assert result.target_has_grad_fn is False
