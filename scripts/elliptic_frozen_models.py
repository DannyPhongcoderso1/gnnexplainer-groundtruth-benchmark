"""Hash-check/load final Elliptic models, or replay their existing inference policies.

No dataset is bundled, no optimizer is constructed, and no training occurs.
Graph replay requires an independently acquired local graph with frozen tensors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_path(root, entry):
    root = Path(root).resolve()
    path = (root / entry["published_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint path escapes root")
    if digest(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Frozen checkpoint mismatch: {entry['published_path']}")
    return path


def safe_checkpoint(root, entry):
    import torch
    from torch.torch_version import TorchVersion

    path = checked_path(root, entry)
    # TorchVersion is a known PyTorch string subclass, not a payload-selected global.
    # Unknown pickle globals fail; there is no weights_only=False fallback.
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_models(root, config):
    from scripts.prepare_elliptic_task26 import EllipticGCN, EllipticSAGE

    backbones = {}
    pg = {}
    for name, cls in [("gcn", EllipticGCN), ("graphsage", EllipticSAGE)]:
        parameters = {k: v for k, v in config["architectures"][name].items() if k != "class"}
        model = cls(**parameters)
        entry = next(
            e for e in config["models"] if e["family"] == "backbone" and e["backbone"] == name
        )
        model.load_state_dict(safe_checkpoint(root, entry), strict=True)
        backbones[name] = model.eval().requires_grad_(False)
    from gnnexplainer_benchmark.explainers.pgexplainer_training import build_pgexplainer_from_config

    for entry in config["models"]:
        if entry["family"] == "backbone":
            continue
        payload = safe_checkpoint(root, entry)
        if payload["seed"] != entry["seed"] or payload["backbone"].lower() != entry["backbone"]:
            raise ValueError("Backbone/seed differs from original checkpoint metadata")
        if (
            payload["epochs_completed"] != 30
            or payload["hashes"]["dataset"] != config["dataset"]["file_sha256"]
        ):
            raise ValueError("Unexpected frozen training/dataset identity")
        algorithm = build_pgexplainer_from_config(payload["explainer_config"])
        algorithm.load_state_dict(payload["algorithm_state_dict"], strict=True)
        algorithm._curr_epoch = payload["algorithm_current_epoch"]
        # PGExplainer.train is its optimization API, not nn.Module.train(mode).
        # Replay uses the original deterministic MLP and never invokes that API.
        algorithm.mlp.eval().requires_grad_(False)
        algorithm.training = False
        pg[(entry["backbone"], entry["family"], entry["seed"])] = algorithm
    return backbones, pg


def tensor_fingerprint(data):
    result = {}
    for name in ["x", "edge_index", "y", "train_mask", "val_mask", "test_mask", "time_step"]:
        tensor = getattr(data, name).detach().cpu().contiguous()
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    return result


def load_graph(path, expected):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    # Only known PyG container classes are allowed; input never selects new globals.
    with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        data = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_fingerprint(data) != expected["tensor_fingerprint"]:
        raise ValueError("Graph tensors/edge order/labels/splits differ from frozen Elliptic data")
    return data


def target_ids(root, entry):
    path = Path(root) / entry["path"]
    # Historical Linux and Windows CSVs share this frozen normalization policy.
    canonical = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if hashlib.sha256(canonical).hexdigest() != entry["sha256"]:
        raise ValueError("Frozen target CSV hash mismatch")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        ids = [int(row["node_id"]) for row in csv.DictReader(stream)]
    if len(ids) != entry["count"] or len(set(ids)) != len(ids):
        raise ValueError("Target identity/count mismatch")
    return ids


def replay(root, config, graph_path, backbone, family, seed, device, limit=None):
    import torch
    from torch_geometric.utils import get_embeddings

    from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity
    from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
    from scripts.evaluate_task29_final_stability import jaccard, selection
    from scripts.evaluate_task29_tie_v1 import rank

    cpu = load_graph(graph_path, config["dataset"])
    ids = target_ids(root, config["targets"]["final200"])
    if limit is not None:
        if limit < 1:
            raise ValueError("--max-targets must be positive")
        ids = ids[:limit]
    models, algorithms = load_models(root, config)
    model = models[backbone].to(device)
    data = cpu.clone().to(device)
    with torch.no_grad():
        embeddings = get_embeddings(model, data.x, data.edge_index)[-1].detach()
    seeds = [42] if family == "historical_seed42" else list(range(5))
    if seed is not None:
        if seed not in seeds:
            raise ValueError("Historical seed42 and FRESH5 seeds0–4 must remain distinct")
        seeds = [seed]
    rows = []
    for node in ids:
        candidates = torch.nonzero(computation_subgraph_edge_mask(cpu, node, 2)).flatten()
        inputs = torch.cat(
            [
                embeddings[data.edge_index[0]],
                embeddings[data.edge_index[1]],
                embeddings[node].view(1, -1).repeat(data.edge_index.size(1), 1),
            ],
            dim=-1,
        )
        selected_sets = []
        for current in seeds:
            mlp = algorithms[(backbone, family, current)].mlp.to(device).eval()
            with torch.no_grad():
                raw = mlp(inputs).view(-1)[candidates.to(device)].cpu()
            if family == "fresh5":
                selected, _ = rank(raw.numpy(), candidates.numpy())
            else:
                local, _ = selection(raw, torch)
                selected = sorted(candidates[local].tolist())
            selected_sets.append(set(selected))
            row = {
                "node_id": node,
                "seed": current,
                "backbone": backbone,
                "family": family,
                "selected_global_edge_indices": selected,
                "sparsity": 1 - len(selected) / len(candidates),
            }
            if family == "historical_seed42":
                mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=device)
                mask[selected] = True
                row.update(asdict(compute_fidelity(model, data, node, mask)))
            rows.append(row)
        if family == "fresh5" and len(seeds) == 5:
            values = [
                jaccard(left, right) for left, right in itertools.combinations(selected_sets, 2)
            ]
            rows.append(
                {
                    "node_id": node,
                    "family": family,
                    "backbone": backbone,
                    "mean_pairwise_jaccard": statistics.mean(values),
                }
            )
    return {
        "scope": "Frozen-checkpoint replay; compare to saved reference before citing",
        "targets": len(ids),
        "device": device,
        "graph_file_sha256": digest(graph_path),
        "graph_tensor_identity": "MATCH_FROZEN",
        "family": family,
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-load", action="store_true")
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--backbone", choices=["gcn", "graphsage"], default="gcn")
    parser.add_argument(
        "--family", choices=["historical_seed42", "fresh5"], default="historical_seed42"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = read_json(args.root / "configs/evaluation/elliptic_frozen_publication.json")
    for entry in config["models"]:
        checked_path(args.root, entry)
    for entry in config["targets"].values():
        target_ids(args.root, entry)
    result = {"status": "PASS", "checkpoint_hashes": 14, "dataset_required_for_hash_check": False}
    if args.graph:
        if not args.output:
            parser.error("Graph replay requires a new --output path")
        if args.output.exists():
            parser.error("Refusing to overwrite an existing output")
        result = replay(
            args.root,
            config,
            args.graph,
            args.backbone,
            args.family,
            args.seed,
            args.device,
            args.max_targets,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    elif args.check_load:
        models, algorithms = load_models(args.root, config)
        result.update(
            backbone_loads=len(models),
            pg_loads=len(algorithms),
            loading="weights_only=True",
            dataset_loaded=False,
        )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
