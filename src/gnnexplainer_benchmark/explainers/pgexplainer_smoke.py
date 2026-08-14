from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import PGExplainer

from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import compute_ratio_metrics
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.training.checkpointing import (
    build_gcn_from_checkpoint,
    load_checkpoint,
)


@dataclass(frozen=True)
class PGExplainerSmokeResult:
    train_nodes: list[int]
    validation_node: int
    losses: list[float]
    edge_mask_shape: tuple[int, ...]
    edge_mask_min: float
    edge_mask_max: float
    candidate_edge_count: int
    selected_edge_count: int
    edge_auroc: float
    edge_ap: float
    model_parameter_hash_before: str
    model_parameter_hash_after: str
    explainer_trainable_parameter_count: int
    explainer_parameters_changed: bool
    explainer_gradients_present: bool
    explainer_gradients_finite: bool
    explainer_optimizer_excludes_gcn_parameters: bool
    sentinel_logits_unchanged: bool
    model_parameters_unchanged: bool
    target_matches_frozen_prediction: bool
    target_requires_grad: bool
    target_has_grad_fn: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_target_node_ids(path: Path, limit: int | None = None) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        node_ids = [int(row["node_id"]) for row in csv.DictReader(handle)]
    if limit is not None:
        return node_ids[:limit]
    return node_ids


def run_pgexplainer_smoke(
    *,
    artifact_path: Path = Path("artifacts/data/ba_house.pt"),
    checkpoint_path: Path = Path("artifacts/checkpoints/ba_house_gcn_best.pt"),
    train_targets_path: Path = Path("configs/targets/pgexplainer_train_targets.csv"),
    validation_targets_path: Path = Path("configs/targets/pgexplainer_val_targets.csv"),
    output_path: Path | None = Path("docs/pgexplainer_smoke_report.json"),
    device: str = "cpu",
    seed: int = 0,
    smoke_epochs: int = 2,
    train_node_count: int = 2,
) -> PGExplainerSmokeResult:
    seed_everything(seed)
    torch_device = torch.device(device)
    data_cpu = load_data_artifact(artifact_path)
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_gcn_from_checkpoint(
        checkpoint,
        in_channels=int(data_cpu.x.size(1)),
        out_channels=int(data_cpu.y.max().item()) + 1,
    ).to(torch_device)
    model.eval()
    data = data_cpu.clone().to(torch_device)

    train_nodes = read_target_node_ids(train_targets_path, limit=train_node_count)
    validation_node = read_target_node_ids(validation_targets_path, limit=1)[0]
    sentinel_nodes = torch.tensor([*train_nodes, validation_node], device=torch_device)

    with torch.no_grad():
        logits_before = model(data.x, data.edge_index)
        frozen_predictions = logits_before.argmax(dim=-1)
        sentinel_logits_before = logits_before[sentinel_nodes].detach().cpu().clone()
    target_matches_frozen_prediction = all(
        int(frozen_predictions[node_id].item()) == int(data.y[node_id].item())
        for node_id in [*train_nodes, validation_node]
    )
    target_requires_grad = bool(frozen_predictions.requires_grad)
    target_has_grad_fn = frozen_predictions.grad_fn is not None

    parameter_hash_before = model_parameter_hash(model)
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=smoke_epochs, lr=0.003),
        explanation_type="phenomenon",
        edge_mask_type="object",
        node_mask_type=None,
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    _materialize_pgexplainer_parameters(
        explainer.algorithm,
        model,
        data.x,
        data.edge_index,
        train_nodes[0],
    )
    explainer_parameters_before = _trainable_parameter_snapshot(explainer.algorithm)
    explainer_trainable_parameter_count = sum(
        parameter.numel() for parameter in explainer_parameters_before.values()
    )
    if explainer_trainable_parameter_count <= 0:
        raise RuntimeError("PGExplainer smoke has no trainable parameters")
    gcn_parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for group in explainer.algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_excludes_gcn_parameters = not (optimizer_parameter_ids & gcn_parameter_ids)

    losses: list[float] = []
    gradients_present = False
    gradients_finite = True
    for epoch in range(smoke_epochs):
        for node_id in train_nodes:
            loss = explainer.algorithm.train(
                epoch,
                model,
                data.x,
                data.edge_index,
                target=frozen_predictions,
                index=node_id,
            )
            if not np.isfinite(loss):
                raise RuntimeError(f"Non-finite PGExplainer smoke loss: {loss}")
            observed_gradients = [
                parameter.grad
                for parameter in explainer.algorithm.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradients_present = gradients_present or bool(observed_gradients)
            gradients_finite = gradients_finite and all(
                bool(torch.isfinite(gradient).all().item())
                for gradient in observed_gradients
            )
            losses.append(float(loss))
    explainer_parameters_after = _trainable_parameter_snapshot(explainer.algorithm)
    explainer_parameters_changed = _parameters_changed(
        explainer_parameters_before,
        explainer_parameters_after,
    )

    explanation = explainer(
        data.x,
        data.edge_index,
        target=frozen_predictions,
        index=validation_node,
    )
    edge_mask = explanation.edge_mask.detach().cpu()
    if edge_mask.numel() != data_cpu.edge_index.size(1):
        raise RuntimeError("PGExplainer smoke edge mask is not full-edge aligned")
    candidate_edge_mask = computation_subgraph_edge_mask(
        data_cpu,
        validation_node,
        int(checkpoint["resolved_config"]["model"]["num_layers"]),
    )
    target = _target_row_for_metrics(
        validation_targets_path,
        validation_node,
    )
    metrics = compute_ratio_metrics(
        model=model,
        data=data,
        node_id=validation_node,
        target=target,
        edge_scores=edge_mask,
        candidate_edge_mask=candidate_edge_mask,
        selected_edge_ratio=0.2,
        runtime_seconds=0.0,
        explainer="PyG PGExplainer smoke",
        candidate_edge_scope="computation_subgraph",
    )
    selected_edge_count = int(metrics["sparsity"]["selected_edge_count"])

    with torch.no_grad():
        logits_after = model(data.x, data.edge_index)
        sentinel_logits_after = logits_after[sentinel_nodes].detach().cpu().clone()
    parameter_hash_after = model_parameter_hash(model)
    result = PGExplainerSmokeResult(
        train_nodes=train_nodes,
        validation_node=validation_node,
        losses=losses,
        edge_mask_shape=tuple(edge_mask.shape),
        edge_mask_min=float(edge_mask.min().item()),
        edge_mask_max=float(edge_mask.max().item()),
        candidate_edge_count=int(candidate_edge_mask.sum().item()),
        selected_edge_count=selected_edge_count,
        edge_auroc=float(metrics["edge_ground_truth"]["edge_auroc"]),
        edge_ap=float(metrics["edge_ground_truth"]["edge_ap"]),
        model_parameter_hash_before=parameter_hash_before,
        model_parameter_hash_after=parameter_hash_after,
        explainer_trainable_parameter_count=explainer_trainable_parameter_count,
        explainer_parameters_changed=explainer_parameters_changed,
        explainer_gradients_present=gradients_present,
        explainer_gradients_finite=gradients_finite,
        explainer_optimizer_excludes_gcn_parameters=optimizer_excludes_gcn_parameters,
        sentinel_logits_unchanged=torch.allclose(
            sentinel_logits_before,
            sentinel_logits_after,
            atol=1e-8,
            rtol=1e-8,
        ),
        model_parameters_unchanged=parameter_hash_before == parameter_hash_after,
        target_matches_frozen_prediction=target_matches_frozen_prediction,
        target_requires_grad=target_requires_grad,
        target_has_grad_fn=target_has_grad_fn,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_result_to_json(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _target_row_for_metrics(path: Path, node_id: int) -> Any:
    from gnnexplainer_benchmark.data.target_selection import ExplainerTarget

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["node_id"]) == node_id:
                return ExplainerTarget(
                    node_id=node_id,
                    label=int(row["ground_truth_label"]),
                    prediction=int(row["frozen_model_prediction"]),
                    confidence=float(row["confidence"]),
                    split=row["base_model_split"],
                    is_ground_truth_motif=str(row["is_ground_truth_motif"]).lower() == "true",
                )
    raise ValueError(f"Node {node_id} not found in {path}")


def _materialize_pgexplainer_parameters(
    algorithm: PGExplainer,
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_id: int,
) -> None:
    with torch.no_grad():
        node_embeddings = algorithm._get_embeddings(model, x, edge_index)
        inputs = algorithm._get_inputs(node_embeddings, edge_index, node_id)
        algorithm.mlp(inputs[:1])


def _trainable_parameter_snapshot(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _parameters_changed(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> bool:
    if before.keys() != after.keys():
        return True
    return any(not torch.equal(before[name], after[name]) for name in before)


def _result_to_json(result: PGExplainerSmokeResult) -> dict[str, Any]:
    return {
        "train_nodes": result.train_nodes,
        "validation_node": result.validation_node,
        "losses": result.losses,
        "losses_finite": all(np.isfinite(loss) for loss in result.losses),
        "edge_mask_shape": list(result.edge_mask_shape),
        "edge_mask_min": result.edge_mask_min,
        "edge_mask_max": result.edge_mask_max,
        "candidate_edge_count": result.candidate_edge_count,
        "selected_edge_count": result.selected_edge_count,
        "edge_auroc": result.edge_auroc,
        "edge_ap": result.edge_ap,
        "model_parameter_hash_before": result.model_parameter_hash_before,
        "model_parameter_hash_after": result.model_parameter_hash_after,
        "explainer_trainable_parameter_count": result.explainer_trainable_parameter_count,
        "explainer_parameters_changed": result.explainer_parameters_changed,
        "explainer_gradients_present": result.explainer_gradients_present,
        "explainer_gradients_finite": result.explainer_gradients_finite,
        "explainer_optimizer_excludes_gcn_parameters": (
            result.explainer_optimizer_excludes_gcn_parameters
        ),
        "sentinel_logits_unchanged": result.sentinel_logits_unchanged,
        "model_parameters_unchanged": result.model_parameters_unchanged,
        "target_matches_frozen_prediction": result.target_matches_frozen_prediction,
        "target_requires_grad": result.target_requires_grad,
        "target_has_grad_fn": result.target_has_grad_fn,
        "note": "Smoke only: 2 epochs on 2 train nodes, explanation on one validation node.",
    }
