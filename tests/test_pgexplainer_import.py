from __future__ import annotations

from torch_geometric.explain.algorithm import PGExplainer


def test_pyg_pgexplainer_imports_from_installed_environment() -> None:
    assert PGExplainer.__name__ == "PGExplainer"
    assert PGExplainer.coeffs["edge_size"] == 0.05
    assert PGExplainer.coeffs["edge_ent"] == 1.0
    assert PGExplainer.coeffs["temp"] == [5.0, 2.0]
