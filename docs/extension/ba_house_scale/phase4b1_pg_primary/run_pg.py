from __future__ import annotations

import csv
import hashlib
import inspect
import json
import statistics as st
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import torch
import torch_geometric
import yaml
from torch_geometric.explain.algorithm import PGExplainer

from gnnexplainer_benchmark.evaluation.budget import select_top_ratio_edges
from gnnexplainer_benchmark.evaluation.edge_ground_truth import compute_edge_ground_truth_metrics
from gnnexplainer_benchmark.evaluation.fidelity import compute_fidelity
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.explainers.pgexplainer_smoke import model_parameter_hash
from gnnexplainer_benchmark.explainers.pgexplainer_training import (
    build_pgexplainer_from_config,
    materialize_pgexplainer,
    wrap_pgexplainer,
)
from gnnexplainer_benchmark.training.checkpointing import build_model_from_checkpoint
from gnnexplainer_benchmark.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parent
OLD = ROOT / "docs/extension/ba_house_scale/phase4a_gnn_random"
TARGET = ROOT / "artifacts/extensions/ba_house_scale/targets"
ART = ROOT / "artifacts/extensions/ba_house_scale/xai/pgexplainer"
RES = ROOT / "results/extensions/ba_house_scale/xai/pgexplainer"
CELLS = [(s, b) for s in ("ba1000_ext", "ba1500_ext") for b in ("gcn", "graphsage")]
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
METRICS = [
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
]
LABELS = [
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
FLAGS = [
    "one_saturation",
    "zero_saturation",
    "near_one_saturation",
    "near_zero_saturation",
    "ranking_collapse",
    "nonfinite",
]
NAMES = [
    "01_pg_protocol_receipt.md",
    "01_pg_protocol_receipt.json",
    "02_pg_training_history.csv",
    "02_pg_training_history.json",
    "03_pg_collapse_diagnostics.csv",
    "03_pg_collapse_diagnostics.json",
    "04_pg_validation_diagnostics.csv",
    "04_pg_validation_diagnostics.json",
    "05_pg_primary_test_results.csv",
    "05_pg_primary_test_results.json",
    "06_pg_primary_aggregate.md",
    "06_pg_primary_aggregate.json",
    "07_pg_vs_ba4a_reference.md",
    "07_pg_vs_ba4a_reference.json",
    "08_verifier_receipt.json",
    "protected_inputs_before.json",
    "protected_inputs_after.json",
    "command_receipt.json",
    "execution_log.json",
    "run_manifest.json",
    "failure_receipt.json",
]
EVENTS = []


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def rec(p):
    with p.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=digest)


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        torch.save(value, f)


def emit(value):
    EVENTS.append(value)
    print(json.dumps(value), flush=True)


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def norm(tensors):
    return float(torch.sqrt(sum(t.detach().double().square().sum() for t in tensors)).item())


def finite(*tensors):
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        raise RuntimeError("BA_HOUSE_SCALE_PG_NONFINITE_FAILURE")


def cell_dir(s, b):
    return ART / s / b / "seed_42"


def preflight():
    for p in [DOC / n for n in NAMES] + list(ART.rglob("*")) + list(RES.rglob("*")):
        if p.is_file():
            raise RuntimeError(
                "BA_HOUSE_SCALE_PG_PRIMARY_OUTPUT_ALREADY_EXISTS: " + json.dumps(rec(p))
            )
    prior = read(OLD / "07_verifier_receipt.json")
    if prior["status"] != "BA_HOUSE_SCALE_BA4A_INDEPENDENT_VERIFIER_PASS":
        raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: BA4A prerequisite")
    if (
        read(ROOT / "docs/extension/ba_house_scale/phase3_targets/04_verifier_receipt.json")[
            "status"
        ]
        != "BA_HOUSE_SCALE_TARGET_INDEPENDENT_VERIFIER_PASS"
    ):
        raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: BA3 prerequisite")
    inputs = {}
    for s in HASHES:
        m = read(TARGET / s / "target_manifest.json")
        paths = [ROOT / m["dataset_path"]] + [
            ROOT / m["checkpoints"][b]["path"] for b in ("gcn", "graphsage")
        ]
        if [rec(p)["sha256"] for p in paths] != HASHES[s]:
            raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: " + s)
        tr = read(TARGET / s / "pg_train_pool.json")["node_ids"]
        va = read(TARGET / s / "pg_validation_pool.json")["node_ids"]
        te = read(TARGET / s / "final_test_targets.json")["selected_ids"]
        if (
            tr != m["pg_train_pool_ids"]
            or va != m["pg_validation_pool_ids"]
            or te != m["selected_ids"]
        ):
            raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: pool identities")
        if [len(tr), len(va), len(te)] != ([94, 32, 24] if s == "ba1000_ext" else [141, 48, 36]):
            raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: counts")
        if (
            any(x != sorted(set(x)) for x in (tr, va, te))
            or set(tr) & set(va)
            or set(tr) & set(te)
            or set(va) & set(te)
        ):
            raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: order/disjointness")
        inputs[s] = dict(manifest=m, train=tr, validation=va, test=te, diagnostic=tr[:8])
        cpu = torch.load(paths[0], map_location="cpu", weights_only=False)
        all_nodes = tr + va + te
        predictions = []
        for checkpoint_path in paths[1:]:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model_from_checkpoint(checkpoint, 10, 4).eval()
            with torch.no_grad():
                predictions.append(model(cpu.x, cpu.edge_index).argmax(-1))
        for nodes, split in [(tr, "train_mask"), (va, "val_mask"), (te, "test_mask")]:
            if not bool(getattr(cpu, split)[nodes].all()) or not bool((cpu.y[nodes] == 1).all()):
                raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: split/class")
        for node in all_nodes:
            mask = computation_subgraph_edge_mask(cpu, node, 3)
            positive = mask & cpu.edge_mask.bool()
            own = (cpu.motif_instance_id[cpu.edge_index[0]] == cpu.motif_instance_id[node]) & (
                cpu.motif_instance_id[cpu.edge_index[1]] == cpu.motif_instance_id[node]
            )
            if (
                int(positive.sum()) != 12
                or int((positive & own).sum()) != 12
                or any(int(pred[node]) != 1 for pred in predictions)
            ):
                raise RuntimeError(
                    "BA_HOUSE_SCALE_PG_INPUT_MISMATCH: common-correct/structural cleanliness"
                )
    return inputs


def protocol():
    cfg = yaml.safe_load((ROOT / "configs/explainer/pgexplainer.yaml").read_text())
    e = cfg["explainer"]
    expected = dict(
        explanation_type="phenomenon",
        task_level="node",
        return_type="raw",
        edge_mask_type="object",
        node_mask_type=None,
        epochs=30,
        learning_rate=0.003,
        seed=42,
    )
    discrepancies = {k: e.get(k) for k, v in expected.items() if e.get(k) != v}
    hyper = dict(
        epochs=30,
        learning_rate=0.003,
        edge_size=0.05,
        edge_ent=1.0,
        temperature_start=5.0,
        temperature_end=2.0,
        bias=0.01,
    )
    paths = [
        "artifacts/explainers/pgexplainer_gcn_ba_house_seed42",
        "artifacts/explainers/pgexplainer_graphsage_ba_house_common_seed42",
    ]
    for p in paths:
        m = read(ROOT / (p + "_manifest.json"))
        cp = torch.load(ROOT / (p + ".pt"), map_location="cpu", weights_only=False)
        if (
            m["pgexplainer_hyperparameters"] != hyper
            or cp["explainer_config"] != cfg
            or cp.get("algorithm_current_epoch", m["epochs_completed"] - 1) != 29
        ):
            discrepancies[p] = "historical checkpoint/config mismatch"
        shapes = {k: list(v.shape) for k, v in cp["algorithm_state_dict"].items()}
        if shapes != {
            "mlp.0.weight": [64, 12],
            "mlp.0.bias": [64],
            "mlp.2.weight": [1, 64],
            "mlp.2.bias": [1],
        }:
            discrepancies[p + "/shapes"] = shapes
    if (
        e["regularization"] != dict(edge_size=0.05, edge_ent=1.0)
        or e["temperature"] != dict(start=5.0, end=2.0)
        or e["implementation_coefficients"]["bias"]["value"] != 0.01
        or e["checkpoint_selection"] != dict(strategy="fixed_final_epoch", final_epoch=30)
    ):
        discrepancies["coefficients_or_selection"] = e
    checks = {
        "SUM": "mask.sum()" in inspect.getsource(PGExplainer._add_mask_regularization),
        "smoothed_MEAN_entropy": "0.99 * mask + 0.005"
        in inspect.getsource(PGExplainer._add_mask_regularization)
        and "mask_ent.mean()" in inspect.getsource(PGExplainer._add_mask_regularization),
        "last_embedding": "[-1]" in inspect.getsource(PGExplainer._get_embeddings),
        "Concrete": "torch.rand_like(logits)" in inspect.getsource(PGExplainer._concrete_sample),
    }
    if not all(checks.values()):
        discrepancies["installed_source"] = checks
    for p in [
        "configs/evaluation/pgexplainer_gcn.yaml",
        "configs/evaluation/pgexplainer_graphsage_common_pool.yaml",
    ]:
        v = yaml.safe_load((ROOT / p).read_text())["evaluation"]
        if v["selected_edge_ratio"] != 0.2 or v["candidate_edge_scope"] != "computation_subgraph":
            discrepancies[p] = v
    if discrepancies:
        raise RuntimeError("BA_HOUSE_SCALE_PG_PROTOCOL_MISMATCH: " + json.dumps(discrepancies))
    sources = [
        "configs/explainer/pgexplainer.yaml",
        "configs/evaluation/pgexplainer_gcn.yaml",
        "configs/evaluation/pgexplainer_graphsage_common_pool.yaml",
        "scripts/train_pgexplainer_gcn.py",
        "scripts/run_pgexplainer_common_pool_sensitivity.py",
        "src/gnnexplainer_benchmark/explainers/pgexplainer_training.py",
        "src/gnnexplainer_benchmark/explainers/pgexplainer_evaluation.py",
        "docs/pgexplainer_protocol.md",
        "docs/audit/ba_house_protocol_table.md",
        "docs/audit/pgexplainer_installed_mechanism_deep_dive.md",
        ".venv/Lib/site-packages/torch_geometric/explain/algorithm/pg_explainer.py",
    ]
    sources += [p + suffix for p in paths for suffix in (".pt", "_manifest.json")]
    return cfg, dict(
        verdict="BA_HOUSE_SCALE_PG_PROTOCOL_PARITY_PASS",
        resolved_config=cfg,
        hyperparameters=hyper,
        evidence=[rec(ROOT / p) for p in sources],
        architecture="last MessagePassing output dim4; [src,dst,target]; 12->64->1; ReLU; biases",
        target="detached frozen-backbone hard argmax; raw multiclass CE",
        training=(
            "installed PGExplainer.train unchanged; ascending IDs; target-wise Adam; "
            "full-E Concrete draw; candidate SUM regularization"
        ),
        entropy="MEAN H(0.99*sigmoid(sampled_logit)+0.005)",
        temperature="5*(2/5)^(epoch/30), epoch0..29",
        inference="float32 sigmoid(raw MLP logits), outside candidate zero; CPU native torch.topk",
        perturbation="physical global REMOVE/RETAIN columns, unchanged x",
        selection="fixed final epoch29; no validation or seed-based selection",
        historical_metadata_note=(
            "Original GCN checkpoint omits algorithm_current_epoch; "
            "its manifest records30 epochs and frozen reload helper uses epochs-1 fallback. "
            "Common-pool GraphSAGE explicitly stores29."
        ),
        limitation=(
            "All train/validation/test pools are class1-only because GCN limits "
            "common-correct eligibility; no pool changes."
        ),
        diagnostic_rule=(
            "first8 sorted frozen train IDs; identical within scale; snapshot after every epoch"
        ),
        definitions=dict(
            one_saturation="fraction(score>.99)>=.95",
            zero_saturation="fraction(score<.01)>=.95",
            near_one_saturation="fraction(score>.95)>=.95",
            near_zero_saturation="fraction(score<.05)>=.95",
            ranking_collapse="C>1 and unique float32 sigmoid scores<=1",
            nonfinite="any NaN/Inf loss/logit/score/parameter/gradient",
        ),
        invalidation=(
            "Any observed ranking collapse invalidates scientific cell; saturation "
            "alone does not; all finite cells train30 epochs."
        ),
        diagnostics_do_not_modify_training=True,
        expected_total_updates=14100,
    )


def protect():
    before = read(OLD / "protected_inputs_after.json")["files"]
    m = read(OLD / "run_manifest.json")
    v = read(OLD / "07_verifier_receipt.json")
    rows = (
        before
        + m["output_hashes"]
        + m["source_hashes"]
        + [v["run_manifest"], v["protected_after"], rec(OLD / "07_verifier_receipt.json")]
    )
    expected = {r["path"]: r for r in rows}
    with ThreadPoolExecutor(8) as pool:
        actual = list(pool.map(lambda p: rec(ROOT / p), sorted(expected)))
    if any(r != expected[r["path"]] for r in actual):
        raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: protected inputs")
    write(DOC / "protected_inputs_before.json", dict(file_count=len(actual), files=actual))
    emit(dict(protected_files=len(actual), preservation="PASS"))
    return actual


def after_check(before):
    with ThreadPoolExecutor(8) as pool:
        after = list(pool.map(lambda r: rec(ROOT / r["path"]), before))
    equal = after == before
    write(
        DOC / "protected_inputs_after.json",
        dict(file_count=len(after), files=after, exact_equality=equal),
    )
    if not equal:
        raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_PRESERVATION_FAILURE")


def stats(raw, scores, selected):
    result = {}
    for name, value in [("raw_logit", raw), ("score", scores)]:
        for key, number in [
            ("min", value.min()),
            ("max", value.max()),
            ("mean", value.mean()),
            ("std", value.std(unbiased=False)),
        ]:
            result[name + "_" + key] = float(number)
    for key, mask in [
        ("lt01", scores < 0.01),
        ("gt99", scores > 0.99),
        ("lt05", scores < 0.05),
        ("gt95", scores > 0.95),
    ]:
        result["fraction_" + key] = float(mask.double().mean())
    result.update(
        unique_float32_scores=int(scores.unique().numel()),
        selected_set_hash=tensor_hash(selected),
        one_saturation=result["fraction_gt99"] >= 0.95,
        zero_saturation=result["fraction_lt01"] >= 0.95,
        near_one_saturation=result["fraction_gt95"] >= 0.95,
        near_zero_saturation=result["fraction_lt05"] >= 0.95,
        ranking_collapse=scores.numel() > 1 and scores.unique().numel() <= 1,
        nonfinite=False,
    )
    return result


@torch.no_grad()
def evaluate(alg, model, cpu, data, s, b, node, epoch, phase):
    mask = computation_subgraph_edge_mask(cpu, node, 3)
    indices = mask.nonzero().view(-1)
    gt = cpu.edge_mask.bool()[mask]
    if gt.sum().item() != 12:
        raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: candidate GT")
    emb = alg._get_embeddings(model, data.x, data.edge_index)
    raw = alg.mlp(alg._get_inputs(emb, data.edge_index, node)).view(-1)
    scores = raw.sigmoid().cpu()
    finite(raw, scores)
    raw = raw.cpu()
    scores[~mask] = 0
    chosen, budget = select_top_ratio_edges(scores, mask, 0.2)
    selected = chosen.nonzero().view(-1)
    row = dict(
        scale=s,
        backbone=b,
        seed=42,
        epoch=epoch,
        phase=phase,
        node_id=node,
        candidate_edge_count=len(indices),
        gt_count=int(gt.sum()),
        selected_edge_count=len(selected),
        selected_global_indices=selected.tolist(),
        sparsity=budget.sparsity,
    )
    row.update(stats(raw[mask], scores[mask], selected))
    if phase != "train_diagnostic":
        row.update(
            {k: v.value for k, v in compute_edge_ground_truth_metrics(scores[mask], gt).items()}
        )
        row.update(asdict(compute_fidelity(model, data, node, chosen.to(data.x.device))))
    payload = dict(
        scale=s,
        backbone=b,
        seed=42,
        epoch=epoch,
        phase=phase,
        node_id=node,
        global_raw_logits=raw,
        global_edge_scores=scores,
        candidate_global_indices=indices,
        restricted_global_gt=gt,
        selected_global_indices=selected,
        row=row,
    )
    return row, payload


def instrument(alg):
    trace = []
    context = {}
    original_loss, original_sample = alg._loss, alg._concrete_sample

    def sample(self, logits, temperature=1.0):
        finite(logits)
        value = original_sample(logits, temperature)
        finite(value, value.sigmoid())
        return value

    def loss(self, y_hat, y, edge_mask):
        value = original_loss(y_hat, y, edge_mask)
        finite(value, y_hat, edge_mask)
        with torch.no_grad():
            q = edge_mask.sigmoid()
            smooth = 0.99 * q + 0.005
            pred = torch.nn.functional.cross_entropy(y_hat, y)
            size = q.sum() * 0.05
            ent = (-smooth * smooth.log() - (1 - smooth) * (1 - smooth).log()).mean()
            context.update(
                prediction_loss=float(pred),
                size_loss=float(size),
                entropy_loss=float(ent),
                total_loss=float(value),
                q=q.cpu(),
                y_hat=y_hat.detach().cpu(),
                target=y.detach().cpu(),
            )
        return value

    def pre_step(optimizer, args, kwargs):
        gradients = [p.grad for p in alg.mlp.parameters()]
        finite(*gradients, *list(alg.mlp.parameters()))
        context["gradient_norm"] = norm(gradients)

    def post_step(optimizer, args, kwargs):
        finite(*list(alg.mlp.parameters()))
        trace.append(dict(context))

    alg._loss = MethodType(loss, alg)
    alg._concrete_sample = MethodType(sample, alg)
    alg.optimizer.register_step_pre_hook(pre_step)
    alg.optimizer.register_step_post_hook(post_step)
    return trace, context


def train_cell(s, b, inputs, cfg):
    seed_everything(42)
    m = inputs["manifest"]
    cpu = torch.load(ROOT / m["dataset_path"], weights_only=False, map_location="cpu")
    data = cpu.clone().to("cuda")
    cp = torch.load(ROOT / m["checkpoints"][b]["path"], weights_only=False, map_location="cpu")
    model = build_model_from_checkpoint(cp, 10, 4).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frozen_hash = model_parameter_hash(model)
    with torch.no_grad():
        teacher = model(data.x, data.edge_index).argmax(-1).detach()
    for nodes, split in [
        (inputs["train"], "train_mask"),
        (inputs["validation"], "val_mask"),
        (inputs["test"], "test_mask"),
    ]:
        if (
            not bool(getattr(cpu, split)[nodes].all())
            or not bool((cpu.y[nodes] == 1).all())
            or not bool((teacher[nodes] == 1).all())
        ):
            raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_MISMATCH: split/prediction")
    alg = build_pgexplainer_from_config(cfg).to("cuda")
    wrap_pgexplainer(model, alg)
    materialize_pgexplainer(alg, model, data.x, data.edge_index, inputs["train"][0])
    model_ids = {id(p) for p in model.parameters()}
    if any(id(p) in model_ids for g in alg.optimizer.param_groups for p in g["params"]):
        raise RuntimeError("BA_HOUSE_SCALE_PG_PROTOCOL_MISMATCH: optimizer contains backbone")
    trace, ctx = instrument(alg)
    history, diagnostics, validation, tests = [], [], [], []
    train_seconds = 0.0
    for epoch in range(30):
        start = time.perf_counter()
        for node in inputs["train"]:
            ctx.update(epoch=epoch, node_id=node, update=len(trace) + 1)
            alg.train(epoch, model, data.x, data.edge_index, target=teacher, index=node)
        train_seconds += time.perf_counter() - start
        updates = trace[-len(inputs["train"]) :]
        gradients = {k: p.grad.detach().cpu().clone() for k, p in alg.named_parameters()}
        state = {k: v.detach().cpu().clone() for k, v in alg.state_dict().items()}
        row = dict(
            scale=s,
            backbone=b,
            seed=42,
            epoch=epoch,
            train_targets_processed=len(updates),
            optimizer_updates=len(updates),
            cumulative_updates=len(trace),
            temperature=alg._get_temperature(epoch),
            parameter_norm=norm(alg.parameters()),
            gradient_norm_mean=st.mean(r["gradient_norm"] for r in updates),
            gradient_norm_min=min(r["gradient_norm"] for r in updates),
            gradient_norm_max=max(r["gradient_norm"] for r in updates),
            gradient_norm_final=updates[-1]["gradient_norm"],
            nonfinite_count=0,
        )
        row.update(
            {
                "mean_" + k: st.mean(r[k] for r in updates)
                for k in ("prediction_loss", "size_loss", "entropy_loss", "total_loss")
            }
        )
        history.append(row)
        # Snapshot and RNG checks ensure diagnostics cannot affect the scientific stream.
        rng_cpu, rng_gpu = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        diag_payloads, val_payloads = [], []
        for node in inputs["diagnostic"]:
            r, p = evaluate(alg, model, cpu, data, s, b, node, epoch, "train_diagnostic")
            r.update(
                parameter_norm=row["parameter_norm"],
                gradient_norm=row["gradient_norm_final"],
                parameters_finite=True,
                gradients_finite=True,
            )
            diagnostics.append(r)
            diag_payloads.append(p)
        for node in inputs["validation"]:
            r, p = evaluate(alg, model, cpu, data, s, b, node, epoch, "validation")
            validation.append(r)
            val_payloads.append(p)
        if (
            not torch.equal(rng_cpu, torch.get_rng_state())
            or not torch.equal(rng_gpu, torch.cuda.get_rng_state())
            or any(not torch.equal(v, alg.state_dict()[k].cpu()) for k, v in state.items())
        ):
            raise RuntimeError("BA_HOUSE_SCALE_PG_PROTOCOL_MISMATCH: diagnostic side effect")
        snapshot = dict(
            scale=s,
            backbone=b,
            seed=42,
            epoch=epoch,
            state_dict=state,
            last_gradients=gradients,
            history=row,
            diagnostic=diag_payloads,
            validation=val_payloads,
            optimizer_steps=[int(v["step"]) for v in alg.optimizer.state.values()],
            diagnostic_no_optimizer_update=True,
            diagnostic_rng_unchanged=True,
        )
        save(cell_dir(s, b) / f"epoch_{epoch:02d}.pt", snapshot)
        emit(
            dict(
                scale=s,
                backbone=b,
                epoch=epoch,
                updates=len(trace),
                loss=row["mean_total_loss"],
                diagnostic_zero=sum(r["row"]["zero_saturation"] for r in diag_payloads),
                diagnostic_ranking_collapse=sum(
                    r["row"]["ranking_collapse"] for r in diag_payloads
                ),
            )
        )
    if len(trace) != len(inputs["train"]) * 30 or alg._curr_epoch != 29:
        raise RuntimeError("BA_HOUSE_SCALE_PG_PROTOCOL_MISMATCH: updates")
    save(
        cell_dir(s, b) / "final_pg.pt",
        dict(
            algorithm_state_dict=state,
            algorithm_current_epoch=29,
            explainer_config=cfg,
            seed=42,
            train_node_ids=inputs["train"],
            validation_node_ids=inputs["validation"],
            test_node_ids=inputs["test"],
            optimizer_updates=len(trace),
            optimizer_state_dict=alg.optimizer.state_dict(),
        ),
    )
    save(cell_dir(s, b) / "training_trace.pt", trace)
    for node in inputs["test"]:
        r, p = evaluate(alg, model, cpu, data, s, b, node, 29, "test")
        tests.append(r)
        save(cell_dir(s, b) / f"node_{node}.pt", p)
    if model_parameter_hash(model) != frozen_hash:
        raise RuntimeError("BA_HOUSE_SCALE_PG_INPUT_PRESERVATION_FAILURE: backbone weights")
    all_rows = diagnostics + validation + tests
    manifest = dict(
        scale=s,
        backbone=b,
        seed=42,
        epochs_completed=30,
        final_epoch=29,
        optimizer_updates=len(trace),
        train_node_ids=inputs["train"],
        validation_node_ids=inputs["validation"],
        test_node_ids=inputs["test"],
        diagnostic_node_ids=inputs["diagnostic"],
        backbone_parameter_hash_before=frozen_hash,
        backbone_parameter_hash_after=model_parameter_hash(model),
        final_checkpoint=rec(cell_dir(s, b) / "final_pg.pt"),
        dataset=rec(ROOT / m["dataset_path"]),
        backbone_checkpoint=rec(ROOT / m["checkpoints"][b]["path"]),
        training_runtime_seconds=train_seconds,
        fixed_final_checkpoint=True,
        validation_optimizer_updates=0,
        seed_selection=False,
        failure_count=0,
        nonfinite_count=0,
        retry_count=0,
        ranking_collapse_observations=sum(r["ranking_collapse"] for r in all_rows),
        scientific_valid=not any(r["ranking_collapse"] for r in all_rows),
        artifacts=[rec(p) for p in sorted(cell_dir(s, b).glob("*.pt"))],
    )
    write(cell_dir(s, b) / "manifest.json", manifest)
    return history, diagnostics, validation, tests, manifest


def rows_report(stem, rows):
    write(DOC / (stem + ".json"), dict(rows=rows))
    with (DOC / (stem + ".csv")).open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
            for r in rows
        )


def markdown(tables):
    output = [
        "# BA-House PG primary seed42",
        "",
        "Class1-only frozen common-correct pools. Fixed epoch29; sigmoid scores; no tuning.",
        "",
    ]
    for t in tables:
        output += [
            "## " + t["name"],
            "",
            "| " + " | ".join(t["columns"]) + " |",
            "| " + " | ".join(["---"] * len(t["columns"])) + " |",
        ]
        output += [
            "| " + " | ".join(f"{v:.9g}" if isinstance(v, float) else str(v) for v in r) + " |"
            for r in t["rows"]
        ]
        output += [""]
    return "\n".join(output)


def aggregate(history, diagnostics, tests, manifests):
    tables = []

    def table(name, columns, rows):
        return dict(name=name, columns=columns, rows=rows)

    tables.append(
        table(
            "PG-B1 Training",
            [
                "Scale",
                "Backbone",
                "TrainPool",
                "ValPool",
                "Epochs",
                "Updates",
                "FinalPredictionLoss",
                "FinalSizeLoss",
                "FinalEntropyLoss",
                "FinalTotalLoss",
            ],
            [
                [
                    s,
                    b,
                    len(m["train_node_ids"]),
                    len(m["validation_node_ids"]),
                    30,
                    m["optimizer_updates"],
                ]
                + [
                    history[i * 30 + 29]["mean_" + k]
                    for k in ("prediction_loss", "size_loss", "entropy_loss", "total_loss")
                ]
                for i, ((s, b), m) in enumerate(zip(CELLS, manifests, strict=True))
            ],
        )
    )
    collapse, primary, comparisons = [], [], []
    reference = read(OLD / "05_ba4a_aggregate_results.json")["tables"]
    for s, b in CELLS:
        d = [r for r in diagnostics if r["scale"] == s and r["backbone"] == b and r["epoch"] == 29]
        t = [r for r in tests if r["scale"] == s and r["backbone"] == b]
        collapse.append([s, b, len(d)] + [sum(r[k] for r in d) for k in FLAGS])
        primary.append([s, b, len(t)] + [st.mean(r[k] for r in t) for k in METRICS])
        ref_g = dict(
            zip(
                reference[0]["columns"],
                next(r for r in reference[0]["rows"] if r[:2] == [s, b]),
                strict=True,
            )
        )
        ref_r = dict(
            zip(
                reference[2]["columns"],
                next(r for r in reference[2]["rows"] if r[:2] == [s, b]),
                strict=True,
            )
        )
        for k, label in zip(METRICS, LABELS, strict=True):
            if k in (
                "edge_auroc",
                "edge_ap",
                "fid_plus",
                "one_minus_fid_minus",
                "remove_delta",
                "sparsity",
            ):
                comparisons.append(
                    [s, b, label, st.mean(r[k] for r in t), ref_g[label], ref_r[label]]
                )
    tables += [
        table(
            "PG-B2 Final train diagnostic subset",
            [
                "Scale",
                "Backbone",
                "DiagnosticTargets",
                "OneSaturation",
                "ZeroSaturation",
                "NearOne",
                "NearZero",
                "RankingCollapse",
                "Nonfinite",
            ],
            collapse,
        ),
        table("PG-B3 Primary test", ["Scale", "Backbone", "N"] + LABELS, primary),
        table(
            "PG-B4 PG vs frozen references",
            ["Scale", "Backbone", "Metric", "PG", "GNNExplainer", "Random"],
            comparisons,
        ),
    ]
    test_diag = [
        dict(
            scale=s,
            backbone=b,
            **{k: sum(r[k] for r in tests if (r["scale"], r["backbone"]) == (s, b)) for k in FLAGS},
            score_mean=st.mean(
                r["score_mean"] for r in tests if (r["scale"], r["backbone"]) == (s, b)
            ),
            score_std=st.mean(
                r["score_std"] for r in tests if (r["scale"], r["backbone"]) == (s, b)
            ),
        )
        for s, b in CELLS
    ]
    return dict(
        tables=tables,
        test_diagnostics=test_diag,
        diagnostic_table_scope="Final epoch29 first8; full trajectory remains in 03 JSON/CSV",
        update_budget_note=(
            "30 epochs means2820 updates per BA1000 backbone and4230 per "
            "BA1500 backbone; total14100."
        ),
        all_epoch_flags=[
            dict(
                scale=s,
                backbone=b,
                **{
                    k: sum(r[k] for r in diagnostics if (r["scale"], r["backbone"]) == (s, b))
                    for k in FLAGS
                },
            )
            for s, b in CELLS
        ],
    )


def main():
    torch.set_num_threads(1)
    inputs = preflight()
    cfg, receipt = protocol()
    if "--preflight" in sys.argv:
        emit(
            dict(
                protocol=receipt["verdict"],
                pools={
                    s: {k: len(v[k]) for k in ("train", "validation", "test")}
                    for s, v in inputs.items()
                },
            )
        )
        return
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable: no scientific retry/device change authorized automatically"
        )
    before = protect()
    receipt["pools"] = {
        s: {k: v[k] for k in ("train", "validation", "test", "diagnostic")}
        for s, v in inputs.items()
    }
    write(DOC / "01_pg_protocol_receipt.json", receipt)
    with (DOC / "01_pg_protocol_receipt.md").open("x", encoding="utf-8") as f:
        f.write(
            "# PG protocol parity receipt\n\n```json\n" + json.dumps(receipt, indent=2) + "\n```\n"
        )
    write(
        DOC / "command_receipt.json",
        dict(
            argv=sys.argv,
            executable=sys.executable,
            cwd=str(ROOT),
            python=sys.version,
            torch=torch.__version__,
            pyg=torch_geometric.__version__,
            device=torch.cuda.get_device_name(),
            training_order=CELLS,
            seed=42,
            no_stability=True,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        ),
    )
    history, diag, val, test, manifests = [], [], [], [], []
    try:
        for s, b in CELLS:
            h, d, v, t, m = train_cell(s, b, inputs[s], cfg)
            history.extend(h)
            diag.extend(d)
            val.extend(v)
            test.extend(t)
            manifests.append(m)
        for stem, rows in [
            ("02_pg_training_history", history),
            ("03_pg_collapse_diagnostics", diag),
            ("04_pg_validation_diagnostics", val),
            ("05_pg_primary_test_results", test),
        ]:
            rows_report(stem, rows)
        summary = aggregate(history, diag, test, manifests)
        write(DOC / "06_pg_primary_aggregate.json", summary)
        with (DOC / "06_pg_primary_aggregate.md").open("x", encoding="utf-8") as f:
            f.write(markdown(summary["tables"]))
        write(
            DOC / "07_pg_vs_ba4a_reference.json",
            dict(tables=[summary["tables"][3]], source=rec(OLD / "05_ba4a_aggregate_results.json")),
        )
        with (DOC / "07_pg_vs_ba4a_reference.md").open("x", encoding="utf-8") as f:
            f.write(markdown([summary["tables"][3]]))
        write(RES / "primary_seed42_summary.json", summary)
        write(
            DOC / "execution_log.json",
            dict(events=EVENTS, failed_runs=0, nonfinite=0, retries=0, skips=0),
        )
        outputs = [
            p
            for root in (DOC, ART, RES)
            for p in root.rglob("*")
            if p.is_file() and p.suffix != ".py"
        ]
        write(
            DOC / "run_manifest.json",
            dict(
                cells=manifests,
                output_hashes=[rec(p) for p in sorted(outputs)],
                source_hashes=[rec(DOC / n) for n in ("run_pg.py", "verify_pg.py")],
                seed=42,
                total_updates=sum(m["optimizer_updates"] for m in manifests),
            ),
        )
        after_check(before)
        result = subprocess.run(
            [sys.executable, "-B", str(DOC / "verify_pg.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise RuntimeError("Independent verifier failed")
    except Exception as exc:
        if not (DOC / "protected_inputs_after.json").exists():
            after_check(before)
        write(
            DOC / "failure_receipt.json",
            dict(error=str(exc), scientific_changes=False, stability_ready=False),
        )
        raise


if __name__ == "__main__":
    main()
