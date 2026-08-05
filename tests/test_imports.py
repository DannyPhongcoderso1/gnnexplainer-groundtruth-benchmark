def test_imports() -> None:
    import torch
    import torch_geometric
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.nn import GCNConv

    import gnnexplainer_benchmark

    assert torch is not None
    assert torch_geometric is not None
    assert GCNConv is not None
    assert Explainer is not None
    assert GNNExplainer is not None
    assert gnnexplainer_benchmark is not None
