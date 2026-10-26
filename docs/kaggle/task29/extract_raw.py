"""Future final200 extraction only. Requires explicit CLI flag; never called by training."""

import argparse
import math
import os
import sys
from pathlib import Path

from common import BASELINE, BENCHMARK, HERE, atomic_json, identity, read_json, require, sha
from state_verify import pool_ids


def execute(backbone, checkpoints, output):
    require(
        len(checkpoints) == 5 and not Path(output).exists(),
        "Five fresh checkpoints and new output required",
    )
    protocol = read_json(HERE / "protocol.json")
    target = BENCHMARK / "configs/targets/elliptic_xai_common_test_targets.csv"
    require(
        sha(target)
        == protocol["frozen_files"]["benchmark"][target.relative_to(BENCHMARK).as_posix()],
        "Frozen final200 hash mismatch",
    )
    nodes = pool_ids(target)
    require(len(nodes) == 200, "Final target count must equal200")
    # Match the archived inference helper, separately from immutable training defaults.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["ELLIPTIC_BASELINE_REPO"] = str(BASELINE)
    sys.path[:0] = [str(BENCHMARK), str(BENCHMARK / "src"), str(BENCHMARK / "scripts")]
    import torch
    from torch_geometric.nn import Linear
    from torch_geometric.utils import get_embeddings
    from prepare_elliptic_task26 import build_models, load_data
    from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
    from gnnexplainer_benchmark.evaluation.budget import select_top_ratio_edges
    from preflight import file_gate
    from state_verify import check_tensors

    file_gate(protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cpu = load_data()
    data = cpu.clone().to("cuda:0")
    model = build_models(cpu)["gcn_compatible" if backbone == "gcn" else "graphsage"]
    model = model.to("cuda:0").eval().requires_grad_(False)
    with torch.no_grad():
        embeddings = get_embeddings(model, data.x, data.edge_index)[-1].detach()
    results = []
    seen = set()
    for path in checkpoints:
        path = Path(path)
        before = sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        seed = saved["seed"]
        require(
            seed in range(5) and seed not in seen and saved["epochs_completed"] == 30,
            "Wrong/duplicate seed or horizon",
        )
        require(
            saved["backbone"] == ("GCN" if backbone == "gcn" else "GraphSAGE"), "Wrong backbone"
        )
        require(
            saved["training_config"]["task29_execution_identity"] == protocol["execution_identity"],
            "Checkpoint is not from reviewed Task29 fresh cohort",
        )
        settings = saved["explainer_config"]["explainer"]
        require(
            settings["regularization"] == {"edge_size": 0.05, "edge_ent": 1.0}
            and settings["learning_rate"] == 0.003
            and settings["epochs"] == 30
            and settings["implementation"] == "torch_geometric",
            "Wrong explainer training configuration",
        )
        for key, expected in protocol["runner_input_hashes"].items():
            require(saved["hashes"][key] == expected, "Extraction checkpoint frozen-input mismatch")
        check_tensors(saved, backbone, 60000, protocol["optimizer_group"], torch)
        seen.add(seed)
        dim = 90 if backbone == "gcn" else 103
        mlp = torch.nn.Sequential(Linear(3 * dim, 64), torch.nn.ReLU(), Linear(64, 1)).to("cuda:0")
        mlp.load_state_dict(
            {k.removeprefix("mlp."): v for k, v in saved["algorithm_state_dict"].items()},
            strict=True,
        )
        mlp.eval().requires_grad_(False)
        records = []
        for node in nodes:
            mask = computation_subgraph_edge_mask(cpu, node, 2)
            ids = torch.nonzero(mask).flatten()
            with torch.no_grad():
                inputs = torch.cat(
                    [
                        embeddings[data.edge_index[0]],
                        embeddings[data.edge_index[1]],
                        embeddings[node].view(1, -1).repeat(data.edge_index.size(1), 1),
                    ],
                    dim=-1,
                )
                raw = mlp(inputs).view(-1)[mask.to("cuda:0")].cpu()
            require(
                raw.dtype == torch.float32 and bool(torch.isfinite(raw).all()),
                "Invalid RAW float32",
            )
            full = torch.zeros(data.edge_index.size(1))
            full[mask] = raw
            selected, budget = select_top_ratio_edges(full, mask, 0.2)
            k = max(1, math.ceil(0.2 * len(ids)))
            require(k == budget.selected_edge_count, "Budget mismatch")
            local = selected[mask]
            require(
                bool((~local).any()) and float(raw[local].min()) > float(raw[~local].max()),
                "RAW boundary tie/flat ranking: STOP; no post-hoc tie policy",
            )
            records.append(
                {
                    "node_id": node,
                    "candidate_global_edge_indices": ids.tolist(),
                    "selected_global_edge_indices": torch.nonzero(selected).flatten().tolist(),
                    "raw_logits": raw.tolist(),
                    "k": k,
                }
            )
        require(sha(path) == before, "Extraction checkpoint changed")
        results.append({"seed": seed, "checkpoint_sha256": before, "records": records})
    result = {
        "schema": 1,
        "backbone": backbone,
        "seeds": sorted(seen),
        "protocol_identity": identity(protocol),
        "target_sha256": sha(target),
        "inference": "RAW_FLOAT32_CPU_TOPK_CANONICAL_DIRECTED_GLOBAL_IDS",
        "hops": 2,
        "top_ratio": 0.2,
        "reporting_role": "DIAGNOSTIC_DESCRIPTIVE"
        if backbone == "gcn"
        else "FRESH_REPLICATION_EXTENSION",
        "runs": results,
    }
    atomic_json(output, result, overwrite=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-final200", action="store_true")
    parser.add_argument("--backbone", choices=["gcn", "graphsage"], required=True)
    parser.add_argument("--checkpoints", nargs=5, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.execute_final200, "Final200 extraction requires separate explicit authorization")
    execute(args.backbone, args.checkpoints, args.output)
