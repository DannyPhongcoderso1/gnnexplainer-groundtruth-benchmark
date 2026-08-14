from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch_geometric
from torch_geometric.explain import ModelConfig
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import ExplainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit installed PyG PGExplainer API.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pgexplainer_installed_api_audit.json"),
    )
    return parser.parse_args()


def support_check(explanation_type: str, node_mask_type: str | None) -> dict[str, Any]:
    algorithm = PGExplainer(epochs=30, lr=0.003)
    try:
        algorithm.connect(
            ExplainerConfig(
                explanation_type=explanation_type,
                node_mask_type=node_mask_type,
                edge_mask_type="object",
            ),
            ModelConfig(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )
        return {"supported": bool(algorithm.supports()), "error": None}
    except Exception as exc:
        return {"supported": False, "error": str(exc)}


def lazy_mlp_audit() -> dict[str, Any]:
    algorithm = PGExplainer(epochs=30, lr=0.003)
    before = repr(algorithm.mlp)
    _ = algorithm.mlp(torch.randn(2, 96))
    after = repr(algorithm.mlp)
    parameters = {
        name: list(parameter.shape)
        for name, parameter in algorithm.mlp.named_parameters()
    }
    return {
        "before_lazy_initialization": before,
        "after_lazy_initialization": after,
        "parameter_shapes": parameters,
    }


def main() -> None:
    args = parse_args()
    audit = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "pgexplainer_class": f"{PGExplainer.__module__}.{PGExplainer.__name__}",
        "source_path": inspect.getfile(PGExplainer),
        "class_signature": str(inspect.signature(PGExplainer)),
        "train_signature": str(inspect.signature(PGExplainer.train)),
        "forward_signature": str(inspect.signature(PGExplainer.forward)),
        "coeffs": PGExplainer.coeffs,
        "documented_lr_default": inspect.signature(PGExplainer).parameters["lr"].default,
        "epochs_required_argument": (
            inspect.signature(PGExplainer).parameters["epochs"].default is inspect._empty
        ),
        "mlp": lazy_mlp_audit(),
        "support_checks": {
            "model_object_edge_mask_node_task": support_check("model", None),
            "phenomenon_object_edge_mask_node_task": support_check("phenomenon", None),
            "phenomenon_with_node_feature_mask": support_check("phenomenon", "object"),
        },
        "target_and_index_requirements": {
            "node_level_index_required": True,
            "explicit_target_required_for_phenomenon": True,
            "thesis_target_semantics": "target_for_pgexplainer = frozen_model_prediction",
        },
        "paper_aligned_defaults_check": {
            "epochs": 30,
            "lr": 0.003,
            "edge_size": 0.05,
            "edge_ent": 1.0,
            "temp": [5.0, 2.0],
            "matches_installed_pyg_coefficients": (
                PGExplainer.coeffs["edge_size"] == 0.05
                and PGExplainer.coeffs["edge_ent"] == 1.0
                and PGExplainer.coeffs["temp"] == [5.0, 2.0]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PGExplainer API audit saved: {args.output}")


if __name__ == "__main__":
    main()
