from __future__ import annotations

from gnnexplainer_benchmark.explainers.pgexplainer_smoke import run_pgexplainer_smoke


def test_pgexplainer_smoke_returns_full_edge_aligned_mask_and_candidate_mapping() -> None:
    result = run_pgexplainer_smoke(output_path=None, device="cpu")

    assert result.edge_mask_shape == (3398,)
    assert result.candidate_edge_count > 0
    assert result.selected_edge_count > 0
    assert result.edge_mask_min <= result.edge_mask_max
