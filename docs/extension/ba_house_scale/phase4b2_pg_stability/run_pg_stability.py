from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import random
import statistics as st
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch_geometric

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parent
BASE = DOC.parent
PRIMARY = BASE / "phase4b1_pg_primary"
POLICY = BASE / "phase4b1_pg_policy"
ART = ROOT / "artifacts/extensions/ba_house_scale/xai/pgexplainer_stability"
TARGET = ROOT / "artifacts/extensions/ba_house_scale/targets"
CELLS = [(s, b) for s in ("ba1000_ext", "ba1500_ext") for b in ("gcn", "graphsage")]
SEEDS = list(range(5))
spec = importlib.util.spec_from_file_location("frozen_primary", PRIMARY / "run_pg.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
EVENTS = []


def now():
    return datetime.now(UTC).isoformat()


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def rec(p):
    with p.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest}


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def markdown(name, text):
    with (DOC / name).open("x", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        torch.save(value, f)


def emit(event):
    event = {"timestamp_utc": now(), **event}
    EVENTS.append(event)
    print(json.dumps(event), flush=True)


def table(stem, rows):
    write(DOC / (stem + ".json"), {"rows": rows})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (DOC / (stem + ".csv")).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
            )


def snapshot(paths):
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(rec, sorted(paths)))


def distribution(values):
    values = list(values)
    return {
        "mean": st.mean(values),
        "std": st.stdev(values) if len(values) > 1 else 0.0,
        "median": st.median(values),
        "min": min(values),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "max": max(values),
        "N": len(values),
    }


def pair_record(s, b, node, a, z, rows):
    left, right = (rows[t] for t in (a, z))
    if left["candidate_global_indices"] != right["candidate_global_indices"]:
        raise RuntimeError("STABILITY_FAILED_PROTOCOL: candidate order mismatch")
    la, lb = set(left["selected_global_indices"]), set(right["selected_global_indices"])
    if len(la) != len(lb):
        raise RuntimeError("STABILITY_FAILED_PROTOCOL: K mismatch")
    return {
        "scale": s,
        "backbone": b,
        "target_id": node,
        "seed_a": a,
        "seed_b": z,
        "Jaccard": len(la & lb) / len(la | lb),
        "K_a": len(la),
        "K_b": len(lb),
        "intersection_size": len(la & lb),
        "union_size": len(la | lb),
    }


def freeze():
    if (DOC / "01_protocol_freeze.json").exists() or any(ART.rglob("*.pt")):
        raise RuntimeError("New stability namespace already contains scientific outputs")
    ready = read(POLICY / "06_stability_readiness.json")
    decision = read(POLICY / "05_primary_phase_decision.json")
    verifier = read(POLICY / "07_policy_verifier_receipt.json")
    if ready["status"] != "PG_STABILITY_READY" or not decision["all_four_cells_accepted"]:
        raise RuntimeError("Stability prerequisite not satisfied")
    if verifier["failed_check_count"] or not verifier["status"].endswith("_PASS"):
        raise RuntimeError("Policy verifier prerequisite failed")
    expected = read(POLICY / "hash_manifest_after.json")["files"]
    paths = {ROOT / r["path"] for r in expected} | {p for p in POLICY.rglob("*") if p.is_file()}
    before = snapshot(paths)
    lookup = {r["path"]: r for r in before}
    if any(lookup.get(r["path"]) != r for r in expected):
        raise RuntimeError("Historical input mutation before stability")
    manifest = read(POLICY / "run_manifest.json")
    for r in (
        manifest["output_hashes"]
        + manifest["source_hashes"]
        + [verifier["manifest"], verifier["verifier"]]
    ):
        if rec(ROOT / r["path"]) != r:
            raise RuntimeError("Policy provenance mismatch")
    cfg, receipt = P.protocol()
    historical_runtime = read(PRIMARY / "command_receipt.json")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no device fallback authorized")
    if (
        str(torch.__version__) != historical_runtime["torch"]
        or torch_geometric.__version__ != historical_runtime["pyg"]
    ):
        raise RuntimeError("Framework/version mismatch with primary")
    inputs = {}
    for s in ("ba1000_ext", "ba1500_ext"):
        m = read(TARGET / s / "target_manifest.json")
        pools = {
            "train": read(TARGET / s / "pg_train_pool.json")["node_ids"],
            "validation": read(TARGET / s / "pg_validation_pool.json")["node_ids"],
            "test": read(TARGET / s / "final_test_targets.json")["selected_ids"],
        }
        if (
            pools["train"] != m["pg_train_pool_ids"]
            or pools["validation"] != m["pg_validation_pool_ids"]
            or pools["test"] != m["selected_ids"]
        ):
            raise RuntimeError("Frozen target/pool mismatch")
        if [len(pools[k]) for k in ("train", "validation", "test")] != (
            [94, 32, 24] if s == "ba1000_ext" else [141, 48, 36]
        ):
            raise RuntimeError("Frozen pool count mismatch")
        if any(v != sorted(set(v)) for v in pools.values()) or any(
            set(pools[a]) & set(pools[b]) for a, b in itertools.combinations(pools, 2)
        ):
            raise RuntimeError("Pool order/disjointness mismatch")
        paths = [ROOT / m["dataset_path"]] + [
            ROOT / m["checkpoints"][b]["path"] for b in ("gcn", "graphsage")
        ]
        if [rec(p)["sha256"] for p in paths] != P.HASHES[s]:
            raise RuntimeError("Dataset/backbone identity mismatch")
        inputs[s] = {"manifest": m, **pools, "diagnostic": pools["train"][:8]}
    sources = [
        rec(DOC / n)
        for n in ("run_pg_stability.py", "verify_pg_stability.py", "test_pg_stability.py")
    ]
    protocol = {
        "phase": "BA-4B2",
        "frozen_at_utc": now(),
        "historical_protocol": rec(PRIMARY / "01_pg_protocol_receipt.json"),
        "primary_runner": rec(PRIMARY / "run_pg.py"),
        "config": cfg,
        "protocol_parity": receipt["verdict"],
        "primary_protocol_description": {
            k: receipt[k]
            for k in (
                "architecture",
                "target",
                "training",
                "entropy",
                "temperature",
                "inference",
                "perturbation",
                "selection",
            )
        },
        "cells": CELLS,
        "seeds": SEEDS,
        "primary_seed42_context_only": True,
        "pools": {
            s: {k: inputs[s][k] for k in ("train", "validation", "test", "diagnostic")}
            for s in inputs
        },
        "sources": sources,
        "source_evidence": receipt["evidence"],
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "pyg": torch_geometric.__version__,
            "device": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "torch_threads": 1,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "seed_handling": (
            "Unchanged repo seed_everything(seed): Python, NumPy, torch CPU, "
            "torch CUDA manual_seed_all. Fresh algorithm, optimizer and materialization "
            "each run. No bitwise determinism override."
        ),
        "evaluation": (
            "Reuse historical evaluate, top20 budget, sklearn recovery and "
            "compute_fidelity helpers. Validation/first8 diagnostics each epoch remain "
            "no-grad and RNG/state-neutral, exactly as primary. No checkpoint selection."
        ),
        "training_storage": (
            "Compact per-update numeric trace and 30-epoch history, not full historical "
            "q/yhat snapshots; logging only, no change to training calls or RNG stream."
        ),
        "statistics": (
            "sample std ddof=1, as existing PG stability pipeline; score std population "
            "per target; metric mean/std across five seed aggregates, "
            "not pooled target-seed observations"
        ),
        "stability": (
            "All 10 unordered pairs per target, then mean per target; cell distribution "
            "across target means. No Jaccard acceptance threshold."
        ),
        "saturation_interpretation": (
            "Any observed ZERO_SATURATION triggers a reported limitation, not numerical "
            "invalidation. Persistent means at least one identical target saturated in "
            "all five seeds; otherwise seed-sensitive when saturation is observed. "
            "Counts/fractions are primary evidence."
        ),
        "numerical_policy_unchanged": rec(POLICY / "01_policy_definition.json"),
        "policy_P3_role": (
            "Raw closeness remains diagnostic only, not a hard gate in authoritative policy."
        ),
        "BA500_context": "Omitted: no BA500 experiment or parity investigation in this task.",
        "retry_policy": (
            "No automatic retry or overwrites. A technical retry requires a logged "
            "separate attempt under identical seed/protocol; no selection by performance."
        ),
    }
    write(DOC / "hash_manifest_before.json", {"files": before, "protected_count": len(before)})
    write(DOC / "01_protocol_freeze.json", protocol)
    markdown(
        "01_protocol_freeze.md",
        "# Frozen BA-4B2 PG stability protocol\n\n```json\n"
        + json.dumps(protocol, indent=2)
        + "\n```",
    )
    emit({"event": "protocol_frozen", "protected_count": len(before), "seeds": SEEDS})


def evaluate(alg, model, cpu, data, s, b, seed, node, epoch, phase):
    row, payload = P.evaluate(alg, model, cpu, data, s, b, node, epoch, phase)
    row["seed"] = payload["seed"] = seed
    row["exact_zero_count"] = int(
        (payload["global_edge_scores"][payload["candidate_global_indices"]] == 0).sum()
    )
    row["candidate_global_indices"] = payload["candidate_global_indices"].tolist()
    return row, payload


def train(s, b, seed, inp, frozen):
    folder = ART / s / b / f"seed_{seed}"
    if any(folder.rglob("*")):
        raise RuntimeError("Scientific output exists; refusing overwrite")
    P.seed_everything(seed)
    cfg = copy.deepcopy(frozen["config"])
    cfg["explainer"]["seed"] = seed
    m = inp["manifest"]
    cpu = torch.load(ROOT / m["dataset_path"], map_location="cpu", weights_only=False)
    data = cpu.clone().to("cuda")
    cp = torch.load(ROOT / m["checkpoints"][b]["path"], map_location="cpu", weights_only=False)
    model = P.build_model_from_checkpoint(cp, 10, 4).to("cuda").eval()
    for param in model.parameters():
        param.requires_grad_(False)
    backbone_hash = P.model_parameter_hash(model)
    with torch.no_grad():
        teacher = model(data.x, data.edge_index).argmax(-1).detach()
    for nodes, split in (
        (inp["train"], "train_mask"),
        (inp["validation"], "val_mask"),
        (inp["test"], "test_mask"),
    ):
        if (
            not bool(getattr(cpu, split)[nodes].all())
            or not bool((cpu.y[nodes] == 1).all())
            or not bool((teacher[nodes] == 1).all())
        ):
            raise RuntimeError("STABILITY_FAILED_PROTOCOL: split/common-correct")
    alg = P.build_pgexplainer_from_config(cfg).to("cuda")
    P.wrap_pgexplainer(model, alg)
    P.materialize_pgexplainer(alg, model, data.x, data.edge_index, inp["train"][0])
    if any(
        id(param) in {id(p) for p in model.parameters()}
        for group in alg.optimizer.param_groups
        for param in group["params"]
    ):
        raise RuntimeError("STABILITY_FAILED_PROTOCOL: backbone in optimizer")
    initial = {k: v.detach().cpu().clone() for k, v in alg.state_dict().items()}
    save(
        folder / "initialization.pt",
        {
            "seed": seed,
            "algorithm_state_dict": initial,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all(),
        },
    )
    trace, context = P.instrument(alg)
    history, diagnostics, validation, tests = [], [], [], []
    train_seconds = 0.0
    emit({"event": "run_started", "scale": s, "backbone": b, "seed": seed})
    for epoch in range(30):
        start = time.perf_counter()
        for node in inp["train"]:
            context.update(epoch=epoch, node_id=node, update=len(trace) + 1)
            alg.train(epoch, model, data.x, data.edge_index, target=teacher, index=node)
        train_seconds += time.perf_counter() - start
        updates = trace[-len(inp["train"]) :]
        state = {k: v.detach().cpu().clone() for k, v in alg.state_dict().items()}
        row = {
            "scale": s,
            "backbone": b,
            "seed": seed,
            "epoch": epoch,
            "train_targets_processed": len(updates),
            "optimizer_updates": len(updates),
            "cumulative_updates": len(trace),
            "temperature": alg._get_temperature(epoch),
            "parameter_norm": P.norm(alg.parameters()),
            "gradient_norm_mean": st.mean(r["gradient_norm"] for r in updates),
            "nonfinite_count": 0,
        }
        row.update(
            {
                "mean_" + k: st.mean(r[k] for r in updates)
                for k in ("prediction_loss", "size_loss", "entropy_loss", "total_loss")
            }
        )
        history.append(row)
        rng_cpu, rng_gpu = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        for node in inp["diagnostic"]:
            r, _ = evaluate(alg, model, cpu, data, s, b, seed, node, epoch, "train_diagnostic")
            diagnostics.append(r)
        for node in inp["validation"]:
            r, _ = evaluate(alg, model, cpu, data, s, b, seed, node, epoch, "validation")
            validation.append(r)
        if (
            not torch.equal(rng_cpu, torch.get_rng_state())
            or not torch.equal(rng_gpu, torch.cuda.get_rng_state())
            or any(not torch.equal(v, alg.state_dict()[k].cpu()) for k, v in state.items())
        ):
            raise RuntimeError("STABILITY_FAILED_PROTOCOL: diagnostic state/RNG side effect")
        if epoch % 5 == 4:
            emit(
                {
                    "event": "epoch_completed",
                    "scale": s,
                    "backbone": b,
                    "seed": seed,
                    "epoch": epoch,
                    "updates": len(trace),
                    "loss": row["mean_total_loss"],
                }
            )
    if len(trace) != len(inp["train"]) * 30 or alg._curr_epoch != 29:
        raise RuntimeError("STABILITY_FAILED_PROTOCOL: update count/epoch")
    save(
        folder / "final_pg.pt",
        {
            "algorithm_state_dict": state,
            "algorithm_current_epoch": 29,
            "explainer_config": cfg,
            "seed": seed,
            "scale": s,
            "backbone": b,
            "train_node_ids": inp["train"],
            "validation_node_ids": inp["validation"],
            "test_node_ids": inp["test"],
            "optimizer_updates": len(trace),
            "optimizer_state_dict": alg.optimizer.state_dict(),
        },
    )
    compact = [
        {k: v for k, v in r.items() if k not in ("q", "y_hat", "target")}
        | {"target": int(r["target"])}
        for r in trace
    ]
    write(folder / "training_trace.json", {"rows": compact, "logging_only": True})
    write(folder / "training_history.json", {"rows": history})
    write(folder / "validation_diagnostics.json", {"rows": validation})
    write(folder / "train_diagnostics.json", {"rows": diagnostics})
    for node in inp["test"]:
        r, payload = evaluate(alg, model, cpu, data, s, b, seed, node, 29, "test")
        r["score_artifact"] = (folder / f"node_{node}.pt").relative_to(ROOT).as_posix()
        tests.append(r)
        save(folder / f"node_{node}.pt", payload)
    if P.model_parameter_hash(model) != backbone_hash:
        raise RuntimeError("STABILITY_FAILED_PROTOCOL: backbone changed")
    summary = {
        "scale": s,
        "backbone": b,
        "seed": seed,
        "epochs": 30,
        "final_epoch": 29,
        "updates": len(trace),
        "train_pool": len(inp["train"]),
        "validation_pool": len(inp["validation"]),
        "test_targets": len(tests),
        "training_seconds": train_seconds,
        "final_loss": history[-1]["mean_total_loss"],
        "failure_count": 0,
        "retry_count": 0,
        "nonfinite_count": 0,
        "initial_state_hash": P.tensor_hash(torch.cat([v.flatten() for v in initial.values()])),
        "final_state_hash": P.tensor_hash(torch.cat([v.flatten() for v in state.values()])),
    }
    write(folder / "training_summary.json", summary)
    manifest = {
        **summary,
        "run_identity": f"{s}/{b}/seed_{seed}/attempt_001",
        "accepted_attempt": 1,
        "overwritten_artifacts": False,
        "seed_config": {
            "python": seed,
            "numpy": seed,
            "torch_cpu": seed,
            "torch_cuda_all": seed,
            "utility": "gnnexplainer_benchmark.utils.reproducibility.seed_everything",
            "bitwise_determinism_forced": False,
        },
        "train_node_ids": inp["train"],
        "validation_node_ids": inp["validation"],
        "test_node_ids": inp["test"],
        "diagnostics_rng_unchanged": True,
        "diagnostics_state_unchanged": True,
        "validation_checkpoint_selection": False,
        "fixed_final_epoch": 29,
        "backbone_parameter_hash_before": backbone_hash,
        "backbone_parameter_hash_after": P.model_parameter_hash(model),
        "dataset": rec(ROOT / m["dataset_path"]),
        "backbone_checkpoint": rec(ROOT / m["checkpoints"][b]["path"]),
        "final_checkpoint": rec(folder / "final_pg.pt"),
        "protocol": rec(DOC / "01_protocol_freeze.json"),
        "artifacts": [rec(p) for p in sorted(folder.iterdir()) if p.is_file()],
    }
    write(folder / "manifest.json", manifest)
    emit(
        {
            "event": "run_completed",
            "scale": s,
            "backbone": b,
            "seed": seed,
            "updates": len(trace),
            "targets": len(tests),
            "zero_saturation": sum(r["zero_saturation"] for r in tests),
            "ranking_collapse": sum(r["ranking_collapse"] for r in tests),
        }
    )
    return tests, summary, manifest


def summarize(targets, summaries):
    pairs, per_target, seed_aggs, cells, saturation, contexts = [], [], [], [], [], []
    historical = read(PRIMARY / "05_pg_primary_test_results.json")["rows"]
    for s, b in CELLS:
        rows = [r for r in targets if (r["scale"], r["backbone"]) == (s, b)]
        ids = read(TARGET / s / "final_test_targets.json")["selected_ids"]
        for node in ids:
            node_rows = {r["seed"]: r for r in rows if r["node_id"] == node}
            if set(node_rows) != set(SEEDS):
                raise RuntimeError("STABILITY_INSUFFICIENT_EVIDENCE")
            values = [
                pair_record(s, b, node, a, z, node_rows)
                for a, z in itertools.combinations(SEEDS, 2)
            ]
            pairs.extend(values)
            per_target.append(
                {
                    "scale": s,
                    "backbone": b,
                    "target_id": node,
                    "target_stability": st.mean(r["Jaccard"] for r in values),
                    "pair_count": len(values),
                }
            )
        for seed in SEEDS:
            rr = [r for r in rows if r["seed"] == seed]
            ag = {"scale": s, "backbone": b, "seed": seed, "N": len(rr)}
            ag.update({metric: st.mean(r[metric] for r in rr) for metric in P.METRICS})
            seed_aggs.append(ag)
            saturation.append(
                {
                    "scale": s,
                    "backbone": b,
                    "seed": seed,
                    "N": len(rr),
                    "zero_saturated_target_count": sum(r["zero_saturation"] for r in rr),
                    "near_zero_target_count": sum(r["near_zero_saturation"] for r in rr),
                    "ranking_collapse_count": sum(r["ranking_collapse"] for r in rr),
                    "exact_zero_entry_count": sum(r["exact_zero_count"] for r in rr),
                    "score_mean": st.mean(r["score_mean"] for r in rr),
                    "score_std": st.mean(r["score_std"] for r in rr),
                    "saturated_fraction": st.mean(r["zero_saturation"] for r in rr),
                }
            )
        sat_counts = [
            sum(r["zero_saturation"] for r in rows if r["node_id"] == node) for node in ids
        ]
        fractions = [
            r["saturated_fraction"] for r in saturation if (r["scale"], r["backbone"]) == (s, b)
        ]
        persistent = sum(n == 5 for n in sat_counts)
        metrics = {
            metric: distribution(
                r[metric] for r in seed_aggs if (r["scale"], r["backbone"]) == (s, b)
            )
            for metric in P.METRICS
        }
        zero = sum(r["zero_saturation"] for r in rows)
        verdict = "STABILITY_COMPLETE_WITH_SATURATION_LIMITATION" if zero else "STABILITY_COMPLETE"
        cells.append(
            {
                "scale": s,
                "backbone": b,
                "seeds": SEEDS,
                "N_targets": len(ids),
                "target_seed_rows": len(rows),
                "pair_rows": len(ids) * 10,
                "stability": distribution(
                    r["target_stability"]
                    for r in per_target
                    if (r["scale"], r["backbone"]) == (s, b)
                ),
                "cross_seed_metrics": metrics,
                "saturated_fraction_mean": st.mean(fractions),
                "saturated_fraction_min": min(fractions),
                "saturated_fraction_max": max(fractions),
                "targets_saturated_5_of_5": persistent,
                "targets_saturated_at_least_4_of_5": sum(n >= 4 for n in sat_counts),
                "targets_never_saturated": sum(n == 0 for n in sat_counts),
                "saturation_interpretation": "persistent_across_seeds"
                if persistent
                else "seed_sensitive"
                if zero
                else "not_observed",
                "ranking_collapse_total": sum(r["ranking_collapse"] for r in rows),
                "cell_verdict": verdict,
            }
        )
        saved = [r for r in historical if (r["scale"], r["backbone"]) == (s, b)]
        for metric in P.METRICS:
            primary_value = st.mean(r[metric] for r in saved)
            dist = metrics[metric]
            contexts.append(
                {
                    "scale": s,
                    "backbone": b,
                    "metric": metric,
                    "seed42_value": primary_value,
                    "seeds0_4_mean": dist["mean"],
                    "seeds0_4_std": dist["std"],
                    "z_like_descriptive_distance": abs(primary_value - dist["mean"]) / dist["std"]
                    if dist["std"] > 0
                    else None,
                    "hypothesis_test": False,
                    "included_in_official_stability": False,
                }
            )
    for stem, rr in (
        ("03_training_summary", summaries),
        ("04_target_level_metrics", targets),
        ("05_pairwise_stability", pairs),
        ("06_seed_level_aggregates", seed_aggs),
        ("07_cell_stability_summary", cells),
        ("08_saturation_across_seeds", saturation),
        ("09_seed42_context", contexts),
        ("13_target_stability", per_target),
    ):
        table(stem, rr)
    write(
        DOC / "10_cross_scale_pg_stability.json",
        {
            "cells": cells,
            "interpretation": (
                "Across the independently generated BA-House configurations, "
                "the recorded values describe stochastic PG initialization/training "
                "variability. They do not identify a causal effect of graph size."
            ),
            "std_ddof": 1,
            "seed42_excluded": True,
            "BA500_omitted": True,
            "no_jaccard_threshold": True,
        },
    )
    lines = [
        "# PG stability across independent BA-House configurations",
        "",
        "Std uses ddof=1. Metric summaries use five seed aggregates. "
        "Stability uses target means of ten pairs.",
        "",
        "| Cell | Jaccard mean +/- std | Min | AUROC mean +/- std | AP mean +/- std | "
        "Fid+ | Fid- | Sparsity | Zero fraction | Collapse | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(d):
        return f"{d['mean']:.6f} +/- {d['std']:.6f}"

    for c in cells:
        m = c["cross_seed_metrics"]
        lines.append(
            f"| {c['scale']} {c['backbone']} | {fmt(c['stability'])} | "
            f"{c['stability']['min']:.6f} | {fmt(m['edge_auroc'])} | "
            f"{fmt(m['edge_ap'])} | {fmt(m['fid_plus'])} | {fmt(m['fid_minus'])} | "
            f"{fmt(m['sparsity'])} | {c['saturated_fraction_mean']:.6f} | "
            f"{c['ranking_collapse_total']} | {c['cell_verdict']} |"
        )
    lines.extend(
        [
            "",
            "Across the independently generated BA-House configurations, these "
            "measurements describe variability, not a causal effect of graph size. "
            "No arbitrary good/bad Jaccard threshold is applied. GCN saturation, "
            "when observed, remains a model limitation; it is not silently corrected. "
            "Seed42 is contextual only. BA500 is omitted.",
        ]
    )
    markdown("10_cross_scale_pg_stability.md", "\n".join(lines))
    limited = any(
        c["backbone"] == "gcn" and c["saturation_interpretation"] == "persistent_across_seeds"
        for c in cells
    )
    decision = {
        "status": "BA_HOUSE_SCALE_PG_STABILITY_COMPLETE_WITH_GCN_SATURATION_LIMITATION"
        if limited
        else "BA_HOUSE_SCALE_PG_STABILITY_COMPLETE",
        "cells": [{k: c[k] for k in ("scale", "backbone", "cell_verdict")} for c in cells],
        "all_20_runs_completed": True,
        "no_retuning": True,
        "primary_seed42_unchanged": True,
        "numerical_policy_unchanged": True,
        "no_jaccard_quality_verdict": True,
        "failure_count": 0,
        "retry_count": 0,
        "seed42_context_only": True,
    }
    write(DOC / "11_final_phase_decision.json", decision)
    markdown(
        "11_final_phase_decision.md",
        "# Final PG stability phase decision\n\n```json\n"
        + json.dumps(decision, indent=2)
        + "\n```",
    )
    markdown(
        "00_README.md",
        "# BA-4B2 PG stability seeds 0..4\n\n"
        + decision["status"]
        + "\n\nTwenty independent training runs; frozen pools/targets/backbones and "
        "historical PyG protocol. All files are exclusive-create; no historical outputs "
        "overwritten. Same helper metrics and sigmoid scores; no rescue/centering/retuning. "
        "Seed42 is contextual, never a sixth official seed. Sample std ddof=1 follows the "
        "existing stability pipeline. Score mean/std are target-weighted. "
        "BA500 omitted without a parity claim.\n\nRunner stages: --freeze, --run. "
        "Independent verification: verify_pg_stability.py. "
        "No scientific acceptance threshold on Jaccard is invented.",
    )
    return decision


def run():
    torch.set_num_threads(1)
    frozen = read(DOC / "01_protocol_freeze.json")
    for r in (
        frozen["sources"]
        + [
            frozen["primary_runner"],
            frozen["historical_protocol"],
            frozen["numerical_policy_unchanged"],
        ]
        + frozen["source_evidence"]
    ):
        if rec(ROOT / r["path"]) != r:
            raise RuntimeError("Frozen source mismatch")
    if (
        str(torch.__version__) != frozen["runtime"]["torch"]
        or torch_geometric.__version__ != frozen["runtime"]["pyg"]
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Frozen runtime mismatch")
    targets, summaries, manifests = [], [], []
    write(
        DOC / "command_receipt.json",
        {
            "argv": sys.argv,
            "executable": sys.executable,
            "started_at_utc": now(),
            "cwd": str(ROOT),
            "training_order": [[s, b, seed] for s, b in CELLS for seed in SEEDS],
            "no_parallel_training": True,
        },
    )
    try:
        for s, b in CELLS:
            inp = {"manifest": read(TARGET / s / "target_manifest.json"), **frozen["pools"][s]}
            for seed in SEEDS:
                rows, summary, manifest = train(s, b, seed, inp, frozen)
                targets.extend(rows)
                summaries.append(summary)
                manifests.append(manifest)
        decision = summarize(targets, summaries)
        before = read(DOC / "hash_manifest_before.json")["files"]
        after = snapshot({ROOT / r["path"] for r in before})
        write(
            DOC / "hash_manifest_after.json",
            {"files": after, "protected_count": len(after), "exact_equality": before == after},
        )
        if before != after:
            raise RuntimeError("Historical input preservation failure")
        write(
            DOC / "execution_log.json",
            {"events": EVENTS, "failure_count": 0, "retry_count": 0, "completed_runs": 20},
        )
        outputs = [
            p
            for root in (DOC, ART)
            for p in root.rglob("*")
            if p.is_file()
            and p.name not in ("02_run_manifest.json", "12_stability_verifier_receipt.json")
        ]
        write(
            DOC / "02_run_manifest.json",
            {
                "cells": manifests,
                "cells_expected": CELLS,
                "seeds": SEEDS,
                "total_runs": 20,
                "total_updates": sum(m["updates"] for m in manifests),
                "targets_expected": 600,
                "pair_rows_expected": 1200,
                "source_hashes": frozen["sources"],
                "output_hashes": [rec(p) for p in sorted(outputs)],
                "protected_count": len(before),
                "input_preservation": True,
                "no_historical_mutation": True,
                "completed_at_utc": now(),
                "phase_status": decision["status"],
            },
        )
        emit({"event": "phase_completed", "status": decision["status"], "total_runs": 20})
    except Exception as error:
        write(
            DOC / "failure_receipt.json",
            {
                "timestamp_utc": now(),
                "exception": repr(error),
                "traceback": traceback.format_exc(),
                "completed_runs": len(manifests),
                "events": EVENTS,
                "overwritten_artifacts": False,
                "automatic_retry": False,
                "no_retuning": True,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze() if args.freeze else run()
