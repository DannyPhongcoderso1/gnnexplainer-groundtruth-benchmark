from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv


class TinyGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


@dataclass(frozen=True)
class SmokeResult:
    device: str
    predicted_class: int
    edge_mask_shape: tuple[int, ...]
    node_mask_shape: tuple[int, ...]
    edge_mask_min: float
    edge_mask_max: float
    runtime_seconds: float


def build_toy_graph(device: torch.device) -> Data:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
        ],
        dtype=torch.long,
        device=device,
    )
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.8, 0.2, 0.4],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.2],
            [0.7, 0.1, 0.6],
            [0.2, 0.8, 0.3],
        ],
        dtype=torch.float,
        device=device,
    )
    y = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long, device=device)
    return Data(x=x, edge_index=edge_index, y=y)


def train_tiny_model(model: nn.Module, data: Data, steps: int = 20) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits, data.y)
        loss.backward()
        optimizer.step()
    model.eval()


def run_smoke(device_name: str = "auto") -> SmokeResult:
    torch.manual_seed(42)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name
        if device_name != "auto"
        else "cpu"
    )
    data = build_toy_graph(device)
    model = TinyGCN(in_channels=3, hidden_channels=8, out_channels=2).to(device)
    train_tiny_model(model, data)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    assert logits.shape == (data.num_nodes, 2)
    assert torch.isfinite(logits).all()
    predicted_class = int(logits[0].argmax().item())

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=5, lr=0.01),
        explanation_type="model",
        node_mask_type="common_attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    start = time.perf_counter()
    explanation = explainer(data.x, data.edge_index, index=0)
    runtime = time.perf_counter() - start

    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask
    assert edge_mask is not None
    assert node_mask is not None
    assert edge_mask.numel() == data.edge_index.size(1)
    assert torch.isfinite(edge_mask).all()
    assert torch.isfinite(node_mask).all()
    for name, param in model.named_parameters():
        assert torch.equal(before[name], param.detach())

    result = SmokeResult(
        device=str(device),
        predicted_class=predicted_class,
        edge_mask_shape=tuple(edge_mask.shape),
        node_mask_shape=tuple(node_mask.shape),
        edge_mask_min=float(edge_mask.min().item()),
        edge_mask_max=float(edge_mask.max().item()),
        runtime_seconds=runtime,
    )
    print(
        "PyG XAI smoke passed | "
        f"device={result.device} | "
        f"predicted_class={result.predicted_class} | "
        f"edge_mask_shape={result.edge_mask_shape} | "
        f"node_mask_shape={result.node_mask_shape} | "
        f"edge_mask_min={result.edge_mask_min:.6f} | "
        f"edge_mask_max={result.edge_mask_max:.6f} | "
        f"runtime_seconds={result.runtime_seconds:.3f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny PyG GNNExplainer smoke test.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(device_name=args.device)
