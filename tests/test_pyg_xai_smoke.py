import torch

from scripts.smoke_test_pyg_xai import TinyGCN, build_toy_graph, run_smoke


def test_toy_gcn_forward_cpu() -> None:
    data = build_toy_graph(torch.device("cpu"))
    model = TinyGCN(in_channels=3, hidden_channels=8, out_channels=2)
    logits = model(data.x, data.edge_index)
    assert logits.shape == (data.num_nodes, 2)
    assert torch.isfinite(logits).all()


def test_gnnexplainer_smoke_cpu() -> None:
    result = run_smoke(device_name="cpu")
    assert result.edge_mask_shape == (12,)
    assert result.node_mask_shape == (1, 3)
    assert result.edge_mask_min <= result.edge_mask_max
