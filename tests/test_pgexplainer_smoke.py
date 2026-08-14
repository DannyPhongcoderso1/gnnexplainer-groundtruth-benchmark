from __future__ import annotations

import math

from gnnexplainer_benchmark.explainers.pgexplainer_smoke import run_pgexplainer_smoke


def test_pgexplainer_minimal_smoke_has_finite_loss_and_shared_metrics() -> None:
    result = run_pgexplainer_smoke(output_path=None, device="cpu")

    assert len(result.losses) == 4
    assert all(math.isfinite(loss) for loss in result.losses)
    assert 0.0 <= result.edge_auroc <= 1.0
    assert 0.0 <= result.edge_ap <= 1.0
    assert result.explainer_trainable_parameter_count > 0
    assert result.explainer_parameters_changed is True
    assert result.explainer_gradients_present is True
    assert result.explainer_gradients_finite is True
    assert result.target_matches_frozen_prediction is True
