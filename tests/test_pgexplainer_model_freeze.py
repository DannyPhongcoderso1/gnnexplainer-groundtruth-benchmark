from __future__ import annotations

from gnnexplainer_benchmark.explainers.pgexplainer_smoke import run_pgexplainer_smoke


def test_pgexplainer_smoke_does_not_mutate_frozen_gcn() -> None:
    result = run_pgexplainer_smoke(output_path=None, device="cpu")

    assert result.model_parameters_unchanged is True
    assert result.sentinel_logits_unchanged is True
    assert result.model_parameter_hash_before == result.model_parameter_hash_after
    assert result.explainer_optimizer_excludes_gcn_parameters is True
