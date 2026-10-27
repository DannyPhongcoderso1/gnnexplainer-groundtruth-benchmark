"""Verify frozen model hashes and strict CPU state loading; no forward/training."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def verify(root: Path, check_load: bool = False) -> dict:
    catalog = json.loads((root / "docs/provenance/external_assets.json").read_text("utf-8"))
    models = graphs = strict = 0
    for entry in catalog["required_artifacts"]:
        relative = entry["published_relative_path"]
        if relative is None:
            assert entry["dataset"] == "Elliptic" and entry["role"] == "DATA_GRAPH"
            continue
        path = root / relative
        content = path.read_bytes()
        assert len(content) == entry["size_bytes"], relative
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], relative
        if entry["role"] == "DATA_GRAPH":
            graphs += 1
            continue
        models += 1
        if not check_load or entry["dataset"] == "Elliptic":
            continue
        import torch
        from torch.torch_version import TorchVersion

        from gnnexplainer_benchmark.explainers.pgexplainer import build_pgexplainer_from_config
        from gnnexplainer_benchmark.training.checkpointing import build_model_from_checkpoint

        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
        assert int(payload["seed"]) == int(entry["seed"]), relative
        if entry["role"] == "BACKBONE_CHECKPOINT":
            state = payload["model_state_dict"]
            backbone = entry["backbone"].lower()
            key = "convs.0.lin.weight" if backbone == "gcn" else "convs.0.lin_l.weight"
            last = max(int(k.split(".")[1]) for k in state if k.startswith("convs."))
            output = f"convs.{last}.bias" if backbone == "gcn" else f"convs.{last}.lin_l.bias"
            model = build_model_from_checkpoint(payload, state[key].shape[1], state[output].numel())
            assert model.__class__.__module__.startswith("gnnexplainer_benchmark.models.synthetic.")
            assert model.__class__.__name__.lower() == backbone
        else:
            algorithm = build_pgexplainer_from_config(payload["explainer_config"])
            algorithm.load_state_dict(payload["algorithm_state_dict"], strict=True)
        strict += 1
    assert models == 71 and graphs == 4
    if check_load:
        from scripts.evaluate.elliptic_frozen_models import load_models, read_json

        config = read_json(root / "configs/elliptic/elliptic_frozen_publication.json")
        classifiers, explainers = load_models(root, config)
        assert len(classifiers) == 2 and len(explainers) == 12
        strict += len(classifiers) + len(explainers)
    return {
        "status": "PASS",
        "model_hashes": models,
        "synthetic_graph_hashes": graphs,
        "strict_cpu_state_loads": strict,
        "forward_calls": 0,
        "training_calls": 0,
        "elliptic_dataset_distributed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-load", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.check_load), indent=2))


if __name__ == "__main__":
    main()
