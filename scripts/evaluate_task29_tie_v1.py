"""TASK29 evaluation-only exact-float32 tie addendum, never model training."""

import argparse
import contextlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np

import evaluate_task29_final_stability as frozen

ROOT = frozen.ROOT
OLD = ROOT / "docs/audit/elliptic_pg_stability/task29_final_evaluation"
OUT = ROOT / "docs/audit/elliptic_pg_stability/task29_final_evaluation_tie_v1"
POLICY = {
    "original_scientific_protocol_identity": frozen.PROTOCOL_ID,
    "addendum_version": "TASK29_EVALUATION_EXACT_TIE_V1",
    "canonical_tie_policy": {
        "primary": "RAW_FLOAT32_DESCENDING",
        "secondary_exact_equal_only": "CANONICAL_GLOBAL_DIRECTED_EDGE_ID_ASCENDING",
        "score_modification": "NONE",
        "budget": "max(1,ceil(0.20*candidate_count))",
    },
}
POLICY_ID = frozen.ident(POLICY)
VERDICT = "TASK29_FINAL_STABILITY_COMPLETE_WITH_INTERPRETIVE_LIMITATIONS"


def write(name, value):
    if isinstance(value, dict):
        value = {"evaluation_policy_identity": POLICY_ID, **value}
    frozen.write(OUT / name, value)


def rank(raw, global_ids, descending_ids=False):
    values = np.asarray(raw)
    ids = np.asarray(global_ids, dtype=np.int64)
    if values.dtype != np.float32 or values.ndim != 1 or values.shape != ids.shape:
        raise ValueError("Exact finite RAW float32 vector and aligned canonical IDs required")
    if not np.isfinite(values).all() or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid scores/duplicate candidate IDs")
    k = max(1, math.ceil(0.2 * len(ids)))
    # Negation is exact in float32; no epsilon, score recomputation or float64 inference.
    order = np.lexsort((-ids if descending_ids else ids, -values))
    selected = np.sort(ids[order[:k]])
    assert len(selected) == len(set(selected.tolist())) == k
    return selected.tolist(), order[:k]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    addendum = {
        **POLICY,
        "evaluation_policy_identity": POLICY_ID,
        "date": "2026-10-15",
        "post_observation": True,
        "original_execution_identity": "7d4706ff1342ae2a7445cd92a7ead06e6200480885fa4a4e6699c92ba2d3e11c",
        "user_approved": True,
        "approval_source": "PRIVATE_APPROVAL_RECORD",
        "uniform_scope": "all2 backbones x5 seeds x200 targets",
        "alternative_diagnostic_only": "RAW descending; exact-equal global IDs descending",
        "original_stop_policy_changed": False,
        "old_blocked_evidence_preserved": True,
        "scientific_inference_arithmetic_changed": False,
        "candidate_universe_changed": False,
        "randomized_ties": False,
        "epsilon": None,
        "score_normalization": None,
        "logit_float64_recomputation": False,
        "additional_training": False,
        "optimized_for_final200_jaccard": False,
        "known_blocker_expected_choice": {"chosen": 417302, "not_chosen": 417344},
    }
    addendum["approval_sha256"] = frozen.sha(addendum["approval_source"])
    path = OUT / "TASK29_TIE_RESOLUTION_ADDENDUM.json"
    if path.exists():
        assert json.loads(path.read_text()) == addendum
    else:
        frozen.write(path, addendum)
    return addendum


def snapshot():
    paths = set(OLD.rglob("*")) | set(frozen.TASK.rglob("*"))
    paths |= {
        ROOT / "scripts/evaluate_task29_final_stability.py",
        ROOT / "scripts/report_task29_final_blocker.py",
        ROOT / "tests/test_task29_final_stability.py",
    }
    return {
        str(p): frozen.sha(p)
        for p in paths
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def run_inference(context, evidence):
    import torch
    import torch_geometric
    from torch_geometric.nn import Linear
    from torch_geometric.utils import get_embeddings
    from prepare_elliptic_task26 import load_data, build_models
    from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask

    protocol, nodes, checkpoints, _ = context
    assert platform.python_version() == "3.11.9" and torch.__version__ == "2.11.0+cu128"
    assert torch_geometric.__version__ == "2.8.0.post1" and torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    evidence["runtime"] = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "pyg": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "deterministic_algorithms": True,
        "matmul_tf32": False,
        "precision": "float32/highest",
        "threads": 1,
        "interop_threads": 1,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "difference_from_training": "Windows RTX4060 vs Linux T4; same library versions, not identical execution environment",
    }
    cpu = load_data()
    data = cpu.clone().to("cuda:0")
    assert data.edge_index.size(1) == 468710 and cpu.num_nodes == 203769
    historical = json.loads(
        (ROOT / "docs/audit/task28_pg_elliptic_full30_raw_final200_case_study.json").read_text()
    )
    references = historical["per_target"]["GCN"]
    assert [r["node_id"] for r in references] == nodes
    candidates = {}
    for node, ref in zip(nodes, references, strict=True):
        ids = torch.nonzero(computation_subgraph_edge_mask(cpu, node, 2)).flatten()
        assert ids.tolist() == ref["candidate_global_edge_indices"]
        candidates[node] = ids
    evidence["all200_candidate_id_order_historical_parity"] = True
    evidence["graph"] = {
        "nodes": cpu.num_nodes,
        "directed_edge_entries": data.edge_index.size(1),
        "flow": "source_to_target",
        "hops": 2,
        "directed_argument": False,
        "edge_identity": "global directed edge_index column; reverse/parallel entries distinct",
    }
    smoked = set(nodes[:3] + [159576])
    models = {}
    for b in ("gcn", "graphsage"):
        model = (
            build_models(cpu)["gcn_compatible" if b == "gcn" else "graphsage"]
            .to("cuda:0")
            .eval()
            .requires_grad_(False)
        )
        state_before = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        emb = get_embeddings(model, data.x, data.edge_index)[-1].detach()
        assert emb.shape[1] == (90 if b == "gcn" else 103)
        nets = {}
        for seed in range(5):
            net = torch.nn.Sequential(
                Linear(3 * emb.shape[1], 64), torch.nn.ReLU(), Linear(64, 1)
            ).to("cuda:0")
            net.load_state_dict(
                {
                    k.removeprefix("mlp."): v
                    for k, v in checkpoints[b, seed]["algorithm_state_dict"].items()
                }
            )
            nets[seed] = net.eval().requires_grad_(False)
        models[b] = (model, emb, nets, state_before)
    evidence["smoke"] = {
        "passed": False,
        "targets": sorted(smoked),
        "repeated_target_seed_vectors": 0,
    }
    started = time.perf_counter()

    def target(b, node, repeat):
        _, emb, nets, _ = models[b]
        ids = candidates[node]
        checkpoint_ids = [r for r in evidence["checkpoints"] if r["backbone"] == b]
        identity = frozen.ident(
            {
                "policy": POLICY_ID,
                "scientific_protocol": frozen.PROTOCOL_ID,
                "runtime": evidence["runtime"],
                "evaluation_source_sha256": frozen.sha(__file__),
                "checkpoint_ids": checkpoint_ids,
                "node_id": node,
                "candidate_ids": ids.tolist(),
            }
        )
        path = OUT / "records" / b / f"{node}.json"
        if path.exists():
            record = json.loads(path.read_text())
            receipt = json.loads(path.with_suffix(".receipt.json").read_text())
            assert record["identity"] == identity and receipt["sha256"] == frozen.sha(path)
            assert (
                record["evaluation_policy_identity"]
                == receipt["evaluation_policy_identity"]
                == POLICY_ID
            )
            assert sorted(r["seed"] for r in record["runs"]) == list(range(5))
            for row in record["runs"]:
                values = np.asarray(row["raw_logits"], dtype=np.float32)
                assert rank(values, ids.tolist())[0] == row["selected_global_edge_indices"]
            return record
        inputs = torch.cat(
            [
                emb[data.edge_index[0]],
                emb[data.edge_index[1]],
                emb[node].view(1, -1).repeat(data.edge_index.size(1), 1),
            ],
            dim=-1,
        )
        rows = []
        for seed, net in nets.items():
            raw = net(inputs).view(-1)[ids.to("cuda:0")].cpu()
            assert (
                raw.dtype == torch.float32
                and len(raw) == len(ids)
                and bool(torch.isfinite(raw).all())
            )
            chosen, local = rank(raw.numpy(), ids.numpy())
            alternative, _ = rank(raw.numpy(), ids.numpy(), True)
            diag = frozen.summarize_logits(raw, local, torch)
            if not diag["boundary_tie_crosses_k"]:
                assert chosen == sorted(ids[torch.topk(raw, k=len(chosen)).indices].tolist())
            if repeat:
                raw_repeat = net(inputs).view(-1)[ids.to("cuda:0")].cpu()
                assert torch.equal(raw, raw_repeat)
                assert rank(raw_repeat.numpy(), ids.numpy())[0] == chosen
                evidence["smoke"]["repeated_target_seed_vectors"] += 1
            if b == "gcn" and seed == 3 and node == 159576:
                assert (
                    diag["boundary_tie_crosses_k"] and diag["boundary_value"] == -223.89364624023438
                )
                assert 417302 in chosen and 417344 not in chosen
                assert 417344 in alternative and 417302 not in alternative
                evidence["known_blocker_tie_resolution_verified"] = True
            rows.append(
                {
                    "seed": seed,
                    "raw_dtype": "float32",
                    "raw_logits": raw.tolist(),
                    "selected_global_edge_indices": chosen,
                    "alternative_selected_global_edge_indices": alternative,
                    "diagnostics": diag,
                    "repeated_inference_exact": repeat,
                    "tie_resolution": "global ID ascending for exactly equal RAW float32",
                }
            )
        record = {
            "evaluation_policy_identity": POLICY_ID,
            "original_scientific_protocol_identity": frozen.PROTOCOL_ID,
            "identity": identity,
            "backbone": b,
            "node_id": node,
            "candidate_global_edge_indices": ids.tolist(),
            "runs": rows,
        }
        write(path.relative_to(OUT), record)
        write(
            path.with_suffix(".receipt.json").relative_to(OUT),
            {"identity": identity, "sha256": frozen.sha(path)},
        )
        return record

    for b in models:
        for node in nodes:
            if node in smoked:
                target(b, node, True)
    assert evidence.get("known_blocker_tie_resolution_verified")
    evidence["smoke"].update(
        passed=True,
        wall_seconds=time.perf_counter() - started,
        approximate_full_inference_seconds=(time.perf_counter() - started) / len(smoked) * 200 / 2,
    )
    write("TASK29_FINAL_EVALUATION_INTEGRITY.json", evidence)
    print(
        f"SMOKE_ALL10_PASS blocker resolved; smoke_s={evidence['smoke']['wall_seconds']:.3f}; automatically full200",
        flush=True,
    )
    for b in models:
        for i, node in enumerate(nodes):
            target(b, node, False)
            if i % 20 == 0:
                print(
                    f"FULL {b} {i + 1}/200; elapsed_s={time.perf_counter() - started:.1f}; peakGiB={torch.cuda.max_memory_allocated() / 2**30:.3f}",
                    flush=True,
                )
        model, _, nets, before = models[b]
        assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items())
        for seed, net in nets.items():
            assert all(
                torch.equal(v.cpu(), checkpoints[b, seed]["algorithm_state_dict"]["mlp." + k])
                for k, v in net.state_dict().items()
            )
    evidence["inference_seconds_including_smoke"] = time.perf_counter() - started
    evidence["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    evidence["loaded_weights_unchanged"] = True
    evidence["old_raw_records_reused"] = False


def aggregate_saved(evidence):
    pairs, targets, diags, summary, sensitivity = [], [], [], [], {}
    all_records = {}
    for b in ("gcn", "graphsage"):
        vector_changed, target_selection_changed, target_pairs_changed = 0, 0, 0
        alt_target_means = []
        for node in evidence["target_ids"]:
            path = OUT / "records" / b / f"{node}.json"
            saved = json.loads(path.read_text())
            receipt = json.loads(path.with_suffix(".receipt.json").read_text())
            assert frozen.sha(path) == receipt["sha256"]
            assert (
                saved["evaluation_policy_identity"]
                == receipt["evaluation_policy_identity"]
                == POLICY_ID
            )
            assert saved["backbone"] == b and saved["node_id"] == node
            ids = saved["candidate_global_edge_indices"]
            assert ids == sorted(set(ids))
            rows = {r["seed"]: r for r in saved["runs"]}
            assert sorted(rows) == list(range(5)) and len(saved["runs"]) == 5
            chosen, alternatives = {}, {}
            changes = 0
            for seed, row in rows.items():
                raw = np.asarray(row["raw_logits"], dtype=np.float32)
                assert raw.tolist() == row["raw_logits"] and len(raw) == len(ids)
                assert rank(raw, ids)[0] == row["selected_global_edge_indices"]
                assert rank(raw, ids, True)[0] == row["alternative_selected_global_edge_indices"]
                chosen[seed] = set(row["selected_global_edge_indices"])
                alternatives[seed] = set(row["alternative_selected_global_edge_indices"])
                assert len(chosen[seed]) == max(1, math.ceil(0.2 * len(ids)))
                changes += chosen[seed] != alternatives[seed]
                diags.append(
                    {
                        "backbone": b,
                        "seed": seed,
                        "node_id": node,
                        **row["diagnostics"],
                        "repeated_inference_exact": row["repeated_inference_exact"],
                        "selection_changed_alternative": chosen[seed] != alternatives[seed],
                    }
                )
            vector_changed += changes
            target_selection_changed += changes > 0
            values, alts = [], []
            for a, c in itertools.combinations(range(5), 2):
                intersection = len(chosen[a] & chosen[c])
                union = len(chosen[a] | chosen[c])
                value = intersection / union
                alt = len(alternatives[a] & alternatives[c]) / len(
                    alternatives[a] | alternatives[c]
                )
                values.append(value)
                alts.append(alt)
                pairs.append(
                    {
                        "evaluation_policy_identity": POLICY_ID,
                        "backbone": b,
                        "node_id": node,
                        "seed_a": a,
                        "seed_b": c,
                        "intersection": intersection,
                        "union": union,
                        "jaccard": value,
                        "alternative_jaccard": alt,
                    }
                )
            target_pairs_changed += values != alts
            alt_target_means.append(statistics.mean(alts))
            targets.append(
                {
                    "evaluation_policy_identity": POLICY_ID,
                    "backbone": b,
                    "node_id": node,
                    "num_pairs": 10,
                    "mean_pairwise_jaccard": statistics.mean(values),
                    "pair_sample_sd_supplementary": statistics.stdev(values),
                    "alternative_mean_jaccard": statistics.mean(alts),
                    "changed_seed_selections": changes,
                }
            )
            all_records[b, node] = saved
        numbers = [r["mean_pairwise_jaccard"] for r in targets if r["backbone"] == b]
        diag = [r for r in diags if r["backbone"] == b]
        headline = statistics.mean(numbers)
        alternate = statistics.mean(alt_target_means)
        summary.append(
            {
                "evaluation_policy_identity": POLICY_ID,
                "backbone": b,
                "seeds": "0-4",
                "targets": 200,
                "edge_budget": "top20% RAW float32; exact ties ID ascending",
                "mean_jaccard": headline,
                "sample_sd": statistics.stdev(numbers),
                "ddof": 1,
                "median": statistics.median(numbers),
                "q1": float(np.quantile(numbers, 0.25)),
                "q3": float(np.quantile(numbers, 0.75)),
                "min": min(numbers),
                "max": max(numbers),
                "status": "VERIFIED_WITH_INTERPRETIVE_LIMITATIONS",
            }
        )
        sensitivity[b] = {
            "target_seed_vectors": 1000,
            "targets": 200,
            "exact_duplicate_vectors": sum(r["duplicate_value_excess"] > 0 for r in diag),
            "boundary_crossing_vectors": sum(r["boundary_tie_crosses_k"] for r in diag),
            "changed_seed_target_selections": vector_changed,
            "targets_with_changed_selections": target_selection_changed,
            "targets_with_affected_pairwise_jaccard": target_pairs_changed,
            "official_mean": headline,
            "alternative_mean": alternate,
            "absolute_difference": abs(headline - alternate),
            "official_larger_than_other_backbone_not_tested_here": None,
        }
        for key in (
            "exact_duplicate_vectors",
            "boundary_crossing_vectors",
            "changed_seed_target_selections",
        ):
            sensitivity[b][key + "_percent"] = 100 * sensitivity[b][key] / 1000
        sensitivity[b]["sigmoid_constant_vectors"] = sum(
            r["sigmoid_unique_float32"] == 1 for r in diag
        )
        sensitivity[b]["all_identical_raw_vectors"] = sum(r["all_identical"] for r in diag)
    assert len(pairs) == 4000 and len(targets) == 400 and len(diags) == 2000
    independent_verify(all_records, pairs, targets, summary, evidence)
    frozen.csv_write(OUT / "TASK29_FINAL_STABILITY_PAIRWISE.csv", pairs)
    frozen.csv_write(OUT / "TASK29_FINAL_STABILITY_PER_TARGET.csv", targets)
    frozen.csv_write(OUT / "TASK29_FINAL_STABILITY_SUMMARY.csv", summary)
    write(
        "TASK29_FINAL_STABILITY_RESULTS.json",
        {
            "verdict": VERDICT,
            "original_scientific_protocol_identity": frozen.PROTOCOL_ID,
            "results": summary,
            "aggregation": "10 unordered seed-pair Jaccards per target; equal200-target mean; sample SD ddof1 across200 target means",
            "sample_sd_statistical_unit": "target-level mean pairwise Jaccard, not seed pairs or backbone models",
            "quartiles": "numpy.quantile linear interpolation",
            "post_observation_addendum": True,
        },
    )
    write(
        "TASK29_RAW_LOGIT_RANKING_AUDIT.json",
        {
            "status": "FULL2000_RECORDS_VERIFIED",
            "records": diags,
            "near_tolerances_absolute_logit_units": list(frozen.NEAR_TOLERANCES),
            "tie_fraction": "(candidate_count-unique_float32_count)/candidate_count, not fraction of all tied-group members",
        },
    )
    write(
        "TASK29_TIE_SENSITIVITY_ANALYSIS.json",
        {
            "primary": "ascending IDs on exact equal scores",
            "alternative": "descending IDs on exact equal scores, diagnostic only",
            "backbones": sensitivity,
            "policy_selected_for_best_result": False,
        },
    )
    evidence["coverage"] = {
        "targets_each": 200,
        "checkpoints": 10,
        "seed_target_vectors": 2000,
        "target_means": 400,
        "pairwise_rows": 4000,
    }
    return summary, sensitivity, diags


def independent_verify(records, pairs, targets, summary, evidence):
    # Independent bit-vector overlap and explicit two-pass ddof1 formula, not primary set/mean helpers.
    pair_lookup = {(r["backbone"], r["node_id"], r["seed_a"], r["seed_b"]): r for r in pairs}
    target_lookup = {(r["backbone"], r["node_id"]): r for r in targets}
    checked = 0
    max_difference = 0.0
    recomputed = {}
    for b in ("gcn", "graphsage"):
        means = []
        for node in evidence["target_ids"]:
            record = records[b, node]
            ids = np.asarray(record["candidate_global_edge_indices"], dtype=np.int64)
            # Read the file again: do not trust the primary aggregator's in-memory sets.
            path = OUT / "records" / b / f"{node}.json"
            disk = json.loads(path.read_text())
            assert disk == record
            masks = {
                r["seed"]: np.isin(
                    ids, np.asarray(r["selected_global_edge_indices"], dtype=np.int64)
                )
                for r in disk["runs"]
            }
            values = []
            for a in range(5):
                for c in range(a + 1, 5):
                    numerator = int(np.count_nonzero(np.logical_and(masks[a], masks[c])))
                    denominator = int(np.count_nonzero(np.logical_or(masks[a], masks[c])))
                    value = numerator / denominator
                    primary = pair_lookup[b, node, a, c]
                    assert numerator == primary["intersection"] and denominator == primary["union"]
                    delta = abs(value - primary["jaccard"])
                    assert delta <= 1e-14
                    max_difference = max(max_difference, delta)
                    checked += 1
                    values.append(value)
            mean = math.fsum(values) / 10
            assert abs(mean - target_lookup[b, node]["mean_pairwise_jaccard"]) <= 1e-14
            means.append(mean)
        overall = math.fsum(means) / 200
        sd = math.sqrt(math.fsum((v - overall) ** 2 for v in means) / 199)
        primary = next(r for r in summary if r["backbone"] == b)
        assert (
            abs(overall - primary["mean_jaccard"]) <= 1e-14
            and abs(sd - primary["sample_sd"]) <= 1e-14
        )
        recomputed[b] = {"mean_jaccard": overall, "sample_sd": sd}
    assert checked == 4000
    evidence["independent_recomputation"] = {
        "status": "PASS",
        "pair_rows_checked": checked,
        "target_means_checked": 400,
        "tolerance_absolute": 1e-14,
        "max_pairwise_difference": max_difference,
        "results": recomputed,
        "method": "Persisted selected global IDs -> independent numpy boolean membership overlaps -> explicit fsum/two-pass sample variance",
    }


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--prepare", action="store_true")
    args = args.parse_args()
    prepare()
    if args.prepare:
        print(POLICY_ID)
        return 0
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["ELLIPTIC_BASELINE_REPO"] = str(frozen.BASE)
    frozen.OUT = OUT
    evidence = {
        "evaluation_policy_identity": POLICY_ID,
        "original_scientific_protocol_identity": frozen.PROTOCOL_ID,
        "verdict": "TASK29_FINAL_STABILITY_BLOCKED",
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_runs": 0,
        "evaluation_source_sha256": frozen.sha(__file__),
        "start_time_unix": time.time(),
        "old_evaluation_snapshot_before": snapshot(),
    }
    write("TASK29_EXECUTION_START.json", evidence)
    started = time.perf_counter()
    try:
        import torch

        with contextlib.ExitStack() as stack:
            for obj, name in [
                (torch.optim.Optimizer, "__init__"),
                (torch.optim.Adam, "step"),
                (torch.Tensor, "backward"),
                (torch.autograd, "backward"),
            ]:
                stack.enter_context(
                    patch.object(obj, name, side_effect=AssertionError("Evaluation-only guard"))
                )
            context = frozen.verify_inputs()
            evidence.update(context[3])
            write("TASK29_FINAL_EVALUATION_INTEGRITY.json", evidence)
            with torch.inference_mode():
                run_inference(context, evidence)
            aggregate_saved(evidence)
            evidence["verdict"] = VERDICT
    except Exception:
        evidence["error"] = traceback.format_exc()
        print(evidence["error"], flush=True)
    finally:
        expected = {
            **evidence["old_evaluation_snapshot_before"],
            **evidence.get("protected_files_before", {}),
        }
        changed = [p for p, h in expected.items() if frozen.sha(p) != h]
        assert not changed, changed
        evidence["all_input_old_evidence_files_after_verified"] = len(expected)
        evidence["unchanged_files"] = True
        evidence["input_sha256_after"] = {p: frozen.sha(p) for p in expected}
        evidence["total_evaluation_wall_seconds"] = time.perf_counter() - started
        evidence["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        write("TASK29_FINAL_EVALUATION_INTEGRITY.json", evidence)
    print(evidence["verdict"], flush=True)
    return 0 if evidence["verdict"] == VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())
