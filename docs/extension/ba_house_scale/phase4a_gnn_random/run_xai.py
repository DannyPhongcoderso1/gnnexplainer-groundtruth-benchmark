from __future__ import annotations

import csv
import hashlib
import json
import statistics as st
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch_geometric
import yaml
from torch_geometric.explain import GNNExplainer

from gnnexplainer_benchmark.evaluation.budget import select_top_ratio_edges
from gnnexplainer_benchmark.evaluation.edge_ground_truth import compute_edge_ground_truth_metrics
from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity
from gnnexplainer_benchmark.evaluation.stability import summarize_node_stability
from gnnexplainer_benchmark.explain.gnnexplainer_runner import run_gnnexplainer_for_node
from gnnexplainer_benchmark.explain.node_selection import TargetNode
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.explainers.random_baseline import random_candidate_edge_scores
from gnnexplainer_benchmark.training.checkpointing import build_model_from_checkpoint

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parent
ART = ROOT / "artifacts/extensions/ba_house_scale/xai"
BA3 = ROOT / "docs/extension/ba_house_scale/phase3_targets"
TARGET_ROOT = ROOT / "artifacts/extensions/ba_house_scale/targets"
SCALES = ("ba1000_ext", "ba1500_ext")
CELLS = [(s, b) for s in SCALES for b in ("gcn", "graphsage")]
HASHES = {
    "ba1000_ext": [
        "7680fbaba95bcfecb96d5f8de0d4935b469e16d31be75d803ed583e5ca8e5b9a",
        "812e123a2ce3c3ed0d9b04eaca78cbba8a4f72d2fa61abf09cf336dde01d75ef",
        "2c51117461c3eb2ff1cb29b291296f5202d114ba4ac4e8543754b6ceeeb607ca",
    ],
    "ba1500_ext": [
        "a4d227ac3b56cf40966c19a5afeef8edbd6a2ad65a9d4a36e34ae5a3dd135b95",
        "be0ffcacc4031eb0dec6195120c608791082475a3d498c59bee1acbb89451bd7",
        "01dcc7c4818e34a91d5d19c36acbefec818b095c29adb13dff208cb4a2a1b56e",
    ],
}
IDS = {
    "ba1000_ext": [
        601,
        615,
        621,
        660,
        671,
        686,
        690,
        701,
        725,
        756,
        775,
        795,
        805,
        810,
        841,
        846,
        866,
        876,
        880,
        950,
        955,
        956,
        976,
        991,
    ],
    "ba1500_ext": [
        905,
        906,
        916,
        935,
        940,
        950,
        951,
        955,
        960,
        990,
        996,
        1016,
        1050,
        1071,
        1076,
        1080,
        1086,
        1105,
        1130,
        1156,
        1161,
        1195,
        1205,
        1206,
        1211,
        1235,
        1321,
        1381,
        1400,
        1405,
        1411,
        1446,
        1465,
        1481,
        1490,
        1491,
    ],
}
METRICS = (
    "edge_auroc",
    "edge_ap",
    "fid_plus",
    "fid_minus",
    "one_minus_fid_minus",
    "remove_delta",
    "retain_probability",
    "retain_delta",
    "retain_probability_ratio",
    "sparsity",
)
NAMES = (
    "01_gnnexplainer_protocol_receipt.md",
    "01_gnnexplainer_protocol_receipt.json",
    "02_gnnexplainer_primary_results.csv",
    "02_gnnexplainer_primary_results.json",
    "03_gnnexplainer_stability_results.csv",
    "03_gnnexplainer_stability_results.json",
    "04_random_results.csv",
    "04_random_results.json",
    "05_ba4a_aggregate_results.md",
    "05_ba4a_aggregate_results.json",
    "06_candidate_complexity_receipt.csv",
    "06_candidate_complexity_receipt.json",
    "07_verifier_receipt.json",
    "protected_inputs_before.json",
    "protected_inputs_after.json",
    "command_receipt.json",
    "execution_log.json",
    "run_manifest.json",
    "failure_receipt.json",
)
EVENTS: list[str] = []
CONTEXT: dict[str, Any] = {}


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rec(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def emit(payload: Any) -> None:
    line = json.dumps(payload)
    EVENTS.append(line)
    print(line, flush=True)


def artifact(method: str, scale: str, backbone: str, node: int, seed: int) -> Path:
    return ART / method / scale / backbone / f"seed_{seed}" / f"node_{node}.pt"


def planned() -> list[Path]:
    paths = [DOC / n for n in NAMES]
    paths += [
        artifact("gnnexplainer", s, b, n, seed)
        for s, b in CELLS
        for seed in (42, 0, 1, 2, 3, 4)
        for n in IDS[s]
    ]
    paths += [
        artifact("random", s, "shared", n, seed)
        for s in SCALES
        for seed in range(10)
        for n in IDS[s]
    ]
    return paths


def inventory() -> list[str]:
    return sorted(
        {
            p.relative_to(ROOT).as_posix()
            for root in ("artifacts", "results", "configs", "docs")
            for p in (ROOT / root).rglob("*")
            if p.is_file()
        }
    )


def protect() -> None:
    expected = {r["path"]: r for r in read(BA3 / "protected_inputs_after.json")["files"]}
    master = read(TARGET_ROOT / "target_freeze_master_manifest.json")
    verifier = read(BA3 / "04_verifier_receipt.json")
    if verifier["status"] != "BA_HOUSE_SCALE_TARGET_INDEPENDENT_VERIFIER_PASS":
        raise RuntimeError("BA_HOUSE_SCALE_BA4A_INVALID: BA3 verifier prerequisite")
    for row in (
        master["file_hashes"]
        + master["source_hashes"]
        + [
            verifier["master_manifest"],
            verifier["protected_inputs_after"],
            rec(BA3 / "04_verifier_receipt.json"),
        ]
    ):
        expected[row["path"]] = row
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(lambda p: rec(ROOT / p), sorted(expected)))
    if any(row != expected[row["path"]] for row in actual):
        raise RuntimeError("BA_HOUSE_SCALE_BA4A_INPUT_PRESERVATION_FAILURE")
    write(
        DOC / "protected_inputs_before.json",
        {
            "file_count": len(actual),
            "files": actual,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "artifact_inventory": inventory(),
        },
    )
    emit({"protected_inputs": len(actual), "identity": "PASS"})


def protocol() -> tuple[dict, dict]:
    cfg = yaml.safe_load((ROOT / "configs/explainer/gnnexplainer.yaml").read_text())
    expected = {
        "algorithm": "PyG GNNExplainer",
        "explanation_type": "model",
        "task_level": "node",
        "return_type": "raw",
        "edge_mask_type": "object",
        "node_mask_type": "common_attributes",
        "epochs": 5,
        "learning_rate": 0.01,
        "seed": 42,
        "top_k_edges": 25,
    }
    coefficients = {
        "edge_size": 0.005,
        "edge_reduction": "sum",
        "edge_ent": 1.0,
        "node_feat_size": 1.0,
        "node_feat_reduction": "mean",
        "node_feat_ent": 0.1,
        "EPS": 1e-15,
    }
    phase4d = yaml.safe_load((ROOT / "configs/evaluation/phase4d.yaml").read_text())["phase4d"]
    metadata_paths = [
        ROOT / "results/explanations/frozen_phase4c_v2/node_306/metadata.json",
        ROOT / "results/explanations/graphsage_gnnexplainer/seed42/node_306/metadata.json",
    ]
    originals = [read(p) for p in metadata_paths]
    if (
        cfg["explainer"] != expected
        or GNNExplainer(epochs=5, lr=0.01).coeffs != coefficients
        or phase4d["selected_edge_ratio"] != 0.2
        or phase4d["stability_seeds"] != list(range(5))
        or phase4d["random_baseline_seeds"] != list(range(10))
        or any(
            m["explainer_config"] != cfg or m["num_hops"] != 3 or m["selected_edge_ratio"] != 0.2
            for m in originals
        )
    ):
        raise RuntimeError(
            "BA_HOUSE_SCALE_GNNEXPLAINER_PROTOCOL_MISMATCH: "
            + json.dumps(
                {"config": cfg, "coefficients": GNNExplainer().coeffs, "historical": originals}
            )
        )
    source_paths = [
        "configs/explainer/gnnexplainer.yaml",
        "configs/evaluation/phase4d.yaml",
        "configs/evaluation/core_metrics.yaml",
        "src/gnnexplainer_benchmark/explain/gnnexplainer_runner.py",
        "src/gnnexplainer_benchmark/evaluation/budget.py",
        "src/gnnexplainer_benchmark/evaluation/fidelity.py",
        "src/gnnexplainer_benchmark/evaluation/edge_ground_truth.py",
        "src/gnnexplainer_benchmark/evaluation/stability.py",
        "src/gnnexplainer_benchmark/explainers/random_baseline.py",
        "scripts/run_random_baseline.py",
        "scripts/run_stability.py",
        "scripts/compute_random_stability.py",
        "src/gnnexplainer_benchmark/evaluation/batch_gnnexplainer.py",
        "docs/audit/ba_house_protocol_table.md",
        ".venv/Lib/site-packages/torch_geometric/explain/algorithm/gnn_explainer.py",
        "results/metrics/frozen_phase4c_v2/batch_gnnexplainer_manifest.json",
        "results/metrics/graphsage_gnnexplainer/seed42/batch_gnnexplainer_manifest.json",
        "results/metrics/phase4d_random_baseline/random_baseline_manifest.json",
    ]
    receipt = {
        "verdict": "BA_HOUSE_SCALE_GNNEXPLAINER_PROTOCOL_PARITY_PASS",
        "resolved_config": cfg,
        "coefficients": coefficients,
        "phase4d": phase4d,
        "optimizer": "Adam mask parameters only",
        "candidate": "induced3hop; full-graph GNN optimization, "
        "candidate-filtered downstream scores",
        "topk": "CPU torch.topk(float32 candidate scores); native frozen tie behavior unchanged",
        "ignored_legacy_top_k_edges": "25 is not used by ratio-budget path; actual K=ceil(.2*C)",
        "fidelity": "Physical edge-column remove and globally retain only selected columns; "
        "x unchanged; "
        "Fid+ and Fid- are original-class flip indicators; ratio None iff p_orig==0",
        "stability": "Mean of all unordered pair Jaccards per target; "
        "macro mean/sample SD over targets",
        "random": "CPU float32 Uniform; seed=s*1000003+node; scores generated once and shared",
        "random_aggregation": "Mean across10 seeds per target then macro targets. Historical SD is "
        "sample SD over pooled target-seed rows (batch summary/raw Random receipt). "
        "Also retain per-target-mean SD and across-seed macro SD separately, "
        "without altering mean.",
        "historical_metadata": [rec(p) for p in metadata_paths],
        "evidence": [rec(ROOT / p) for p in source_paths],
        "no_tuning": True,
    }
    return cfg, receipt


def csv_json(stem: str, rows: list[dict], payload: dict | None = None) -> None:
    write(DOC / (stem + ".json"), {"rows": rows, **(payload or {})})
    with (DOC / (stem + ".csv")).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
            for r in rows
        )


def score_hash(scores: torch.Tensor) -> str:
    return hashlib.sha256(scores.contiguous().numpy().tobytes()).hexdigest()


def save_scores(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        torch.save(payload, f)
    return rec(path)


def summarize(
    primary: list[dict], stability: list[dict], random: list[dict], complexity: list[dict]
) -> dict:
    def table(name: str, columns: list[str], rows: list[list]) -> dict:
        return {"name": name, "columns": columns, "rows": rows}

    pmeans, rmeans, stab, rand_stab, dispersion = {}, {}, {}, {}, []
    for scale, backbone in CELLS:
        cell = [r for r in primary if (r["scale"], r["backbone"]) == (scale, backbone)]
        pmeans[scale, backbone] = {m: st.mean(r[m] for r in cell) for m in METRICS}
        nodes = []
        for node in IDS[scale]:
            edge_sets = [
                set(r["selected_global_indices"])
                for r in stability
                if (r["scale"], r["backbone"], r["node_id"]) == (scale, backbone, node)
            ]
            nodes.append(summarize_node_stability(node, edge_sets)["mean_pairwise_jaccard"])
        stab[scale, backbone] = {"mean": st.mean(nodes), "std": st.stdev(nodes), "values": nodes}
        target_means = {m: [] for m in METRICS}
        random_cell = [r for r in random if (r["scale"], r["backbone"]) == (scale, backbone)]
        random_jaccards = []
        for node in IDS[scale]:
            runs = [r for r in random_cell if r["node_id"] == node]
            for metric in METRICS:
                target_means[metric].append(st.mean(r[metric] for r in runs))
            random_jaccards.append(
                summarize_node_stability(node, [set(r["selected_global_indices"]) for r in runs])[
                    "mean_pairwise_jaccard"
                ]
            )
        rmeans[scale, backbone] = {m: st.mean(v) for m, v in target_means.items()}
        rand_stab[scale, backbone] = st.mean(random_jaccards)
        for metric in METRICS:
            seed_means = [
                st.mean(r[metric] for r in random_cell if r["seed"] == seed) for seed in range(10)
            ]
            dispersion.append(
                {
                    "scale": scale,
                    "backbone": backbone,
                    "metric": metric,
                    "historical_pooled_target_seed_sd": st.stdev(r[metric] for r in random_cell),
                    "per_target_mean_sd": st.stdev(target_means[metric]),
                    "seed_macro_mean_sd": st.stdev(seed_means),
                    "per_target_means": target_means[metric],
                    "seed_macro_means": seed_means,
                }
            )
    metric_columns = [
        "EdgeAUROC",
        "EdgeAP",
        "FidelityPlus",
        "FidelityMinus",
        "OneMinusFidelityMinus",
        "RemoveDelta",
        "RetainProbability",
        "RetainDelta",
        "RetainRatio",
        "Sparsity",
    ]
    t1 = table(
        "BA-X1 GNNExplainer Primary",
        ["Scale", "Backbone", "N", *metric_columns],
        [[s, b, len(IDS[s])] + [pmeans[s, b][m] for m in METRICS] for s, b in CELLS],
    )
    t2 = table(
        "BA-X2 GNNExplainer Stability",
        [
            "Scale",
            "Backbone",
            "N",
            "JaccardMean",
            "JaccardSD",
            "SuccessfulTargets",
            "FailedTargets",
        ],
        [
            [s, b, len(IDS[s]), stab[s, b]["mean"], stab[s, b]["std"], len(IDS[s]), 0]
            for s, b in CELLS
        ],
    )
    t3 = table(
        "BA-X3 Random",
        [
            "Scale",
            "Backbone",
            "N",
            "RandomSeeds",
            "EdgeAUROC",
            "EdgeAUROC_SD",
            "EdgeAP",
            "EdgeAP_SD",
            *metric_columns[2:],
            "Jaccard",
        ],
        [
            [
                s,
                b,
                len(IDS[s]),
                10,
                rmeans[s, b]["edge_auroc"],
                st.stdev(r["edge_auroc"] for r in random if (r["scale"], r["backbone"]) == (s, b)),
                rmeans[s, b]["edge_ap"],
                st.stdev(r["edge_ap"] for r in random if (r["scale"], r["backbone"]) == (s, b)),
            ]
            + [rmeans[s, b][m] for m in METRICS[2:]]
            + [rand_stab[s, b]]
            for s, b in CELLS
        ],
    )
    t4rows = []
    for s, b in CELLS:
        for m in (
            "edge_auroc",
            "edge_ap",
            "fid_plus",
            "one_minus_fid_minus",
            "remove_delta",
            "stability",
        ):
            p = stab[s, b]["mean"] if m == "stability" else pmeans[s, b][m]
            r = rand_stab[s, b] if m == "stability" else rmeans[s, b][m]
            t4rows.append([s, b, m, p, r, p - r])
    t4 = table(
        "BA-X4 GNNExplainer vs Random",
        ["Scale", "Backbone", "Metric", "GNNExplainer", "Random", "Difference"],
        t4rows,
    )
    descriptive = table(
        "Scale Candidate Complexity",
        [
            "Scale",
            "Targets",
            "MeanCandidateEdges",
            "MedianCandidateEdges",
            "MinCandidateEdges",
            "MaxCandidateEdges",
            "MeanGTPrevalence",
            "MedianGTPrevalence",
        ],
        [
            [
                s,
                len(IDS[s]),
                st.mean(r["candidate_count"] for r in complexity if r["scale"] == s),
                st.median(r["candidate_count"] for r in complexity if r["scale"] == s),
                min(r["candidate_count"] for r in complexity if r["scale"] == s),
                max(r["candidate_count"] for r in complexity if r["scale"] == s),
                st.mean(r["gt_prevalence"] for r in complexity if r["scale"] == s),
                st.median(r["gt_prevalence"] for r in complexity if r["scale"] == s),
            ]
            for s in SCALES
        ],
    )
    return {
        "tables": [t1, t2, t3, t4, descriptive],
        "random_dispersion": dispersion,
        "gnn_target_stability": [
            {"scale": s, "backbone": b, "node_ids": IDS[s], **stab[s, b]} for s, b in CELLS
        ],
        "mean_aggregation": "Random seed mean within target then macro; "
        "other methods macro targets",
        "random_sd": "Historical pooled target-seed sample SD; alternative SDs explicitly labeled",
        "failure_count": 0,
        "significance_testing_performed": False,
    }


def main() -> None:
    collisions = [rec(p) for p in planned() if p.exists()]
    if collisions:
        emit(collisions)
        raise RuntimeError("BA_HOUSE_SCALE_BA4A_OUTPUT_ALREADY_EXISTS")
    cfg, protocol_receipt = protocol()
    for s in SCALES:
        manifest = read(TARGET_ROOT / s / "target_manifest.json")
        final = read(TARGET_ROOT / s / "final_test_targets.json")
        if manifest["selected_ids"] != final["selected_ids"] or final["selected_ids"] != IDS[s]:
            raise RuntimeError("BA_HOUSE_SCALE_BA4A_TARGET_MISMATCH")
        input_files = [
            ROOT / manifest["dataset_path"],
            *[ROOT / manifest["checkpoints"][b]["path"] for b in ("gcn", "graphsage")],
        ]
        if [sha(p) for p in input_files] != HASHES[s]:
            raise RuntimeError("BA_HOUSE_SCALE_BA4A_INVALID: frozen input hash mismatch")
    protect()
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, models, candidates, base_states, complexity = {}, {}, {}, {}, []
    for s in SCALES:
        manifest = read(TARGET_ROOT / s / "target_manifest.json")
        data[s] = torch.load(
            ROOT / manifest["dataset_path"], map_location="cpu", weights_only=False
        )
        for b in ("gcn", "graphsage"):
            checkpoint = torch.load(
                ROOT / manifest["checkpoints"][b]["path"], map_location=device, weights_only=False
            )
            models[s, b] = build_model_from_checkpoint(checkpoint, 10, 4).to(device).eval()
            base_states[s, b] = {
                k: v.detach().cpu().clone() for k, v in models[s, b].state_dict().items()
            }
        with torch.no_grad():
            preds = [
                models[s, b](data[s].x.to(device), data[s].edge_index.to(device)).argmax(-1).cpu()
                for b in ("gcn", "graphsage")
            ]
        adjacency = [set() for _ in range(data[s].num_nodes)]
        for u, v in data[s].edge_index.t().tolist():
            adjacency[u].add(v)
        for n in IDS[s]:
            candidate = computation_subgraph_edge_mask(data[s], n, 3)
            seen, frontier = {n}, {n}
            for _ in range(3):
                frontier = {v for u in frontier for v in adjacency[u]} - seen
                seen |= frontier
            independent_candidate = torch.tensor(
                [u in seen and v in seen for u, v in data[s].edge_index.t().tolist()]
            )
            if not torch.equal(candidate, independent_candidate):
                raise RuntimeError("BA_HOUSE_SCALE_BA4A_INVALID: independent candidate mismatch")
            indices = torch.where(candidate)[0]
            gt = data[s].edge_mask[candidate].bool()
            inst = data[s].motif_instance_id
            src, dst = data[s].edge_index[:, indices]
            own = (inst[src] == inst[n]) & (inst[dst] == inst[n]) & gt
            frozen = next(r for r in manifest["selected_target_records"] if r["node_id"] == n)
            if not (
                bool(data[s].test_mask[n])
                and bool(data[s].node_mask[n])
                and int(data[s].y[n]) == 1
                and all(int(p[n]) == 1 for p in preds)
                and frozen["structurally_clean"]
                and int(gt.sum()) == 12
                and int(own.sum()) == 12
                and not bool((gt & ~own).any())
                and len(indices) > 12
                and len(indices) == frozen["candidate_directed_edge_count"]
            ):
                raise RuntimeError("BA_HOUSE_SCALE_BA4A_INVALID: target invariants " + str(n))
            candidates[s, n] = candidate
            complexity.append(
                {
                    "scale": s,
                    "node_id": n,
                    "candidate_count": len(indices),
                    "gt_positive_count": 12,
                    "gt_prevalence": 12 / len(indices),
                }
            )
    write(DOC / "01_gnnexplainer_protocol_receipt.json", protocol_receipt)
    with (DOC / "01_gnnexplainer_protocol_receipt.md").open("x", encoding="utf-8") as f:
        f.write(
            "# GNNExplainer Protocol Parity\n\n```json\n"
            + json.dumps(protocol_receipt, indent=2)
            + "\n```\n"
        )
    write(
        DOC / "command_receipt.json",
        {
            "argv": sys.argv,
            "executable": sys.executable,
            "cwd": str(ROOT),
            "device": str(device),
            "python": sys.version,
            "torch": torch.__version__,
            "pyg": torch_geometric.__version__,
            "cuda": torch.version.cuda,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "primary_order": CELLS,
            "stability_seeds": list(range(5)),
            "random_seeds": list(range(10)),
            "no_PG": True,
        },
    )
    device_data = {s: d.clone().to(device) for s, d in data.items()}
    primary, stability, random = [], [], []

    def process(
        method: str, s: str, b: str, n: int, seed: int, scores: torch.Tensor, artifact_record: dict
    ) -> dict:
        candidate = candidates[s, n]
        indices = torch.where(candidate)[0]
        selected, budget = select_top_ratio_edges(scores, candidate, 0.2)
        candidate_scores, gt = scores[candidate].float(), data[s].edge_mask[candidate].bool()
        metrics = compute_edge_ground_truth_metrics(candidate_scores, gt)
        fidelity = asdict(compute_fidelity(models[s, b], device_data[s], n, selected.to(device)))
        row = {
            "scale": s,
            "backbone": b,
            "method": method,
            "node_id": n,
            "seed": seed,
            "label": 1,
            "candidate_count": len(indices),
            "gt_positive_count": 12,
            "gt_prevalence": 12 / len(indices),
            "top_k": budget.selected_edge_count,
            "selected_count": int(selected.sum()),
            "selected_local_indices": torch.where(selected[indices])[0].tolist(),
            "selected_global_indices": torch.where(selected)[0].tolist(),
            "scores_artifact": artifact_record,
            "score_sha256": score_hash(scores),
            "edge_auroc": metrics["edge_auroc"].value,
            "edge_ap": metrics["edge_ap"].value,
            **fidelity,
            "sparsity": budget.sparsity,
            "status": "success",
        }
        if row["original_prediction"] != 1 or not all(
            isinstance(row[m], (int, float)) and torch.isfinite(torch.tensor(row[m])).item()
            for m in METRICS
        ):
            raise RuntimeError("BA_HOUSE_SCALE_BA4A_INVALID: nonfinite metrics")
        return row

    for stable in (False, True):
        for s, b in CELLS:
            for seed in range(5) if stable else [42]:
                for n in IDS[s]:
                    CONTEXT.update(method="gnnexplainer", scale=s, backbone=b, seed=seed, node_id=n)
                    config = deepcopy(cfg)
                    config["explainer"]["seed"] = seed
                    explanation = run_gnnexplainer_for_node(
                        models[s, b], device_data[s], TargetNode(n, 1, 1, 0.0), config
                    )
                    scores = explanation.edge_mask.detach().cpu()
                    candidate = candidates[s, n]
                    if scores.numel() != candidate.numel() or not torch.isfinite(scores).all():
                        raise RuntimeError("BA_HOUSE_SCALE_BA4A_INVALID: invalid scores")
                    artifact_record = save_scores(
                        artifact("gnnexplainer", s, b, n, seed),
                        {
                            "edge_scores": scores,
                            "node_feature_mask": explanation.node_mask.detach().cpu(),
                            "candidate_global_indices": torch.where(candidate)[0],
                            "candidate_edge_scores": scores[candidate],
                            "restricted_global_gt": data[s].edge_mask[candidate].bool(),
                            "scale": s,
                            "backbone": b,
                            "node_id": n,
                            "seed": seed,
                            "resolved_config": config,
                        },
                    )
                    row = process("gnnexplainer", s, b, n, seed, scores, artifact_record)
                    (stability if stable else primary).append(row)
                emit(
                    {
                        "completed": "GNN stability" if stable else "GNN primary",
                        "scale": s,
                        "backbone": b,
                        "seed": seed,
                        "targets": len(IDS[s]),
                    }
                )
    for s in SCALES:
        for seed in range(10):
            for n in IDS[s]:
                CONTEXT.update(method="random", scale=s, backbone="shared", seed=seed, node_id=n)
                candidate = candidates[s, n]
                scores = random_candidate_edge_scores(
                    edge_count=candidate.numel(),
                    candidate_edge_mask=candidate,
                    seed=seed * 1000003 + n,
                )
                artifact_record = save_scores(
                    artifact("random", s, "shared", n, seed),
                    {
                        "edge_scores": scores,
                        "candidate_global_indices": torch.where(candidate)[0],
                        "candidate_edge_scores": scores[candidate],
                        "restricted_global_gt": data[s].edge_mask[candidate].bool(),
                        "scale": s,
                        "backbone": "shared",
                        "node_id": n,
                        "seed": seed,
                        "node_seed": seed * 1000003 + n,
                    },
                )
                pair = [
                    process("random", s, b, n, seed, scores, artifact_record)
                    for b in ("gcn", "graphsage")
                ]
                for key in ("score_sha256", "selected_global_indices", "edge_auroc", "edge_ap"):
                    if pair[0][key] != pair[1][key]:
                        raise RuntimeError("BA_HOUSE_SCALE_RANDOM_BACKBONE_INVARIANCE_FAILURE")
                random.extend(pair)
            emit(
                {
                    "completed": "Random shared scores",
                    "scale": s,
                    "seed": seed,
                    "targets": len(IDS[s]),
                }
            )
    for cell, model in models.items():
        if any(
            not torch.equal(v.detach().cpu(), base_states[cell][k])
            for k, v in model.state_dict().items()
        ):
            raise RuntimeError("BA_HOUSE_SCALE_BA4A_INVALID: backbone mutation")
    aggregate = summarize(primary, stability, random, complexity)
    csv_json("02_gnnexplainer_primary_results", primary)
    csv_json(
        "03_gnnexplainer_stability_results",
        stability,
        {"target_jaccard": aggregate["gnn_target_stability"]},
    )
    csv_json("04_random_results", random, {"dispersion": aggregate["random_dispersion"]})
    complexity_metrics = [
        {
            **row,
            "method": r["method"],
            "backbone": r["backbone"],
            "seed": r["seed"],
            **{m: r[m] for m in METRICS},
        }
        for row in complexity
        for r in primary + random
        if (row["scale"], row["node_id"]) == (r["scale"], r["node_id"])
    ]
    csv_json("06_candidate_complexity_receipt", complexity_metrics, {"unique_targets": complexity})
    write(DOC / "05_ba4a_aggregate_results.json", aggregate)
    with (DOC / "05_ba4a_aggregate_results.md").open("x", encoding="utf-8") as f:
        f.write(
            "# BA-4A Aggregate Results\n\nRandom means: seed average per target then macro. "
            "Random SD: historical pooled target-seed sample SD; alternatives retained in JSON.\n"
        )
        for table in aggregate["tables"]:
            f.write("\n## " + table["name"] + "\n\n| " + " | ".join(table["columns"]) + " |\n")
            f.write("| " + " | ".join(["---"] * len(table["columns"])) + " |\n")
            for row in table["rows"]:
                f.write("| " + " | ".join(str(x) for x in row) + " |\n")
    emit(
        {
            "completed": "BA4A execution awaiting independent verifier",
            "primary": len(primary),
            "stability": len(stability),
            "random_rows": len(random),
            "failures": 0,
        }
    )
    write(
        DOC / "execution_log.json",
        {
            "stdout": "\n".join(EVENTS) + "\n",
            "stderr": "",
            "failed_runs": 0,
            "retries": 0,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "backbone_training": False,
            "PG_execution": False,
        },
    )
    write(
        DOC / "run_manifest.json",
        {
            "input_hashes": HASHES,
            "targets": IDS,
            "protocol": protocol_receipt,
            "output_hashes": [rec(p) for p in planned() if p.exists()],
            "source_hashes": [rec(Path(__file__)), rec(DOC / "verify_xai.py")],
            "expected_counts": {
                "primary": 120,
                "stability": 600,
                "random_rows": 1200,
                "random_score_artifacts": 600,
            },
            "method_cells": 8,
            "raw_score_artifacts": 1320,
            "no_backbone_mutation": True,
            "PG_execution": False,
            "final_verifier_receipt": "07_verifier_receipt.json",
            "self_after_and_verifier_excluded": True,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if CONTEXT and not (DOC / "failure_receipt.json").exists():
            write(
                DOC / "failure_receipt.json",
                {
                    "status": "BA_HOUSE_SCALE_BA4A_INVALID",
                    "failed_run": CONTEXT,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "stdout": "\n".join(EVENTS),
                    "no_target_substitution": True,
                    "no_config_retry": True,
                },
            )
        raise
