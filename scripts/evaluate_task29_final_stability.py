"""Frozen TASK29 evaluation only; fail closed at any tied top-k boundary."""

import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "docs/kaggle/task29"
RELEASE = TASK / "dist/release_002"
OUT = ROOT / "docs/audit/elliptic_pg_stability/task29_final_evaluation"
BASE = ROOT.parent / "elliptic-gnn-xai"
PROTOCOL_ID = "e126d23ccccc8fd078ae2a8faa5e2913b8a2ddf9619d35857b25c0855392f6de"
NEAR_TOLERANCES = (1e-7, 1e-6, 1e-5)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ident(value):
    return digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checked_archive(source, expected=None):
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    if expected is not None and digest(raw) != expected:
        raise ValueError("ZIP outer SHA mismatch")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    if len(names) != len(set(names)) or "ARCHIVE_MANIFEST.json" not in names:
        raise ValueError("Duplicate members or missing archive manifest")
    if sum(i.file_size for i in archive.infolist()) > 4 * 1024**3:
        raise ValueError("Archive too large")
    for info in archive.infolist():
        name = info.filename
        mode = info.external_attr >> 16
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or ":" in name
            or any(p in (".", "..", "") for p in name.split("/"))
            or info.is_dir()
            or info.flag_bits & 1
            or mode & 0o170000 not in (0, 0o100000)
        ):
            raise ValueError("Unsafe ZIP member: " + name)
    manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
    if set(names) != set(manifest["files"]) | {"ARCHIVE_MANIFEST.json"}:
        raise ValueError("Unexpected archive member set")
    if archive.testzip() is not None:
        raise ValueError("ZIP CRC failure")
    for name, row in manifest["files"].items():
        data = archive.read(name)
        if len(data) != row["size_bytes"] or digest(data) != row["sha256"]:
            raise ValueError("Member integrity failure: " + name)
    return archive, manifest, digest(raw)


def jaccard(left, right):
    a, b = set(left), set(right)
    if not a or not b:
        raise ValueError("Empty explanation")
    return len(a & b) / len(a | b)


def aggregate(rows):
    values = [r["mean_pairwise_jaccard"] for r in rows]
    return {
        "mean_jaccard": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "sample_sd_unit": "200 per-target means of 10 unordered seed-pair Jaccards",
        "ddof": 1,
    }


def summarize_logits(raw, selected_local, torch):
    import numpy as np

    a = raw.numpy()
    n = len(a)
    k = len(selected_local)
    descending = np.sort(a)[::-1]
    boundary = float(descending[k - 1])
    above = int((a > boundary).sum())
    equal = int((a == boundary).sum())
    unique = int(np.unique(a).size)
    cross = above < k < above + equal
    wide = a.astype(np.float64)
    return {
        "candidate_edge_count": n,
        "k": k,
        "finite_count": int(np.isfinite(a).sum()),
        "unique_raw_float32": unique,
        "duplicate_value_excess": n - unique,
        "tie_fraction": (n - unique) / n,
        "all_identical": unique == 1,
        "raw_min": float(wide.min()),
        "raw_max": float(wide.max()),
        "raw_mean": float(wide.mean()),
        "raw_std_population": float(wide.std(ddof=0)),
        "boundary_value": boundary,
        "boundary_tie_multiplicity": equal,
        "boundary_tie_crosses_k": cross,
        "boundary_gap": float(float(descending[k - 1]) - float(descending[k])) if k < n else None,
        "near_boundary_counts": {
            str(t): int((np.abs(wide - boundary) <= t).sum()) for t in NEAR_TOLERANCES
        },
        "near_tie_sensitivity": {
            str(t): {
                "boundary_gap_le_2tol": bool(
                    k < n and float(descending[k - 1]) - float(descending[k]) <= 2 * t
                ),
                "selected_within_tol": int((np.abs(wide[selected_local] - boundary) <= t).sum()),
            }
            for t in NEAR_TOLERANCES
        },
        "sigmoid_unique_float32": int(torch.unique(raw.sigmoid()).numel()),
        "sigmoid_std_population": float(raw.sigmoid().double().std(correction=0)),
    }


def selection(raw, torch):
    k = max(1, math.ceil(0.2 * raw.numel()))
    indices = torch.topk(raw, k=k).indices
    diagnostics = summarize_logits(raw, indices.numpy(), torch)
    return indices, diagnostics


def csv_write(path, rows):
    if not rows:
        raise ValueError("No CSV rows")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_cached_record(record, record_identity):
    if record["identity"] != record_identity:
        raise ValueError(
            "Cached evaluation identity changed; preserve old outputs, use new reviewed output directory"
        )
    if sorted(r["seed"] for r in record["runs"]) != list(range(5)):
        raise ValueError("Cached seed matrix incomplete or duplicated")
    if any(r["diagnostics"]["boundary_tie_crosses_k"] for r in record["runs"]):
        raise ValueError(
            "EXACT_RAW_BOUNDARY_TIE: cached diagnostic record is not admitted as official"
        )


def verify_inputs():
    import torch

    upload = json.loads((RELEASE / "UPLOAD_MANIFEST.json").read_text())
    # Anchor outer packages to the previously reviewed active release, not supplied ZIP metadata alone.
    active = json.loads((TASK / "ACTIVE_RELEASE.json").read_text())
    anchors = {r["name"]: r["sha256"] for r in active["artifacts"]}
    code, _, code_sha = checked_archive(
        RELEASE / "TASK29_KAGGLE_CODE_BUNDLE.zip", anchors["TASK29_KAGGLE_CODE_BUNDLE.zip"]
    )
    frozen, _, input_sha = checked_archive(
        RELEASE / "TASK29_ELLIPTIC_FROZEN_INPUTS.zip", anchors["TASK29_ELLIPTIC_FROZEN_INPUTS.zip"]
    )
    protocol = json.loads(code.read("execution/protocol.json"))
    assert ident(protocol) == PROTOCOL_ID
    preflight = json.loads(
        (ROOT / "docs/audit/elliptic_pg_stability/preflight_manifest.json").read_text()
    )
    assert preflight["evaluation"]["future_boundary_tie_policy"].startswith("STOP")
    protected = {p: h for p, h in preflight["protected_input_sha256"].items()}
    assert all(sha(p) == h for p, h in protected.items())
    for name, expected in protocol["frozen_files"]["baseline"].items():
        assert sha(BASE / name) == expected == digest(frozen.read("baseline/" + name))
        protected[str(BASE / name)] = expected
    # Import the exact source bytes shipped in Release002, never dirty local implementations.
    source_root = OUT / "frozen_source"
    for name in code.namelist():
        if name.startswith("benchmark/"):
            path = source_root / name.removeprefix("benchmark/")
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = code.read(name)
            if path.exists():
                assert path.read_bytes() == raw
            else:
                path.write_bytes(raw)
    sys.path[:0] = [str(source_root / "scripts"), str(source_root / "src"), str(TASK)]
    from state_verify import check_tensors
    from common import training_config

    target_path = source_root / "configs/targets/elliptic_xai_common_test_targets.csv"
    assert sha(target_path) == preflight["evaluation"]["final_target_sha256"]
    targets = [int(r["node_id"]) for r in csv.DictReader(io.StringIO(target_path.read_text()))]
    assert len(targets) == len(set(targets)) == 200 and targets == sorted(targets)
    expected_train = [
        int(r["node_id"])
        for r in csv.DictReader(
            io.StringIO(
                code.read("benchmark/configs/targets/elliptic_pg_common_train.csv").decode()
            )
        )
    ]
    expected_val = [
        int(r["node_id"])
        for r in csv.DictReader(
            io.StringIO(code.read("benchmark/configs/targets/elliptic_pg_common_val.csv").decode())
        )
    ]
    assert set(targets).isdisjoint(expected_train) and set(targets).isdisjoint(expected_val)
    checkpoints, archives, inventory = {}, [], []
    for session, seeds in [("A", [0, 1, 2]), ("B", [3, 4])]:
        path = ROOT / f"docs/kaggle/TASK29_ELLIPTIC_STABILITY_SESSION_{session}_ALL.zip"
        master, _, outer_sha = checked_archive(path)
        protected[str(path)] = outer_sha
        receipt = json.loads(master.read("session_receipt.json"))
        assert receipt["status"] == "COMPLETE" and receipt["session"] == session
        assert receipt["protocol_identity"] == PROTOCOL_ID and not receipt["failures"]
        assert json.loads(master.read("input_source_manifest.json")) == protocol
        expected = {(b, s) for b in ("gcn", "graphsage") for s in seeds}
        assert {(r["backbone"], r["seed"]) for r in receipt["results"]} == expected
        assert len(receipt["results"]) == len(expected)
        assert {(r["backbone"], r["seed"]) for r in receipt["matrix"]} == expected
        archives.append(
            {
                "path": str(path),
                "sha256": outer_sha,
                "status": "COMPLETE",
                "platform_selected_master_attested": False,
                "reason": "current_master.json/execution_summary.json not supplied; canonical masters verified by contents only",
            }
        )
        for row in receipt["results"]:
            b, seed = row["backbone"], row["seed"]
            key = (b, seed)
            assert key not in checkpoints and row["status"] == "COMPLETE"
            name = f"TASK29_{b.upper()}_SEED{seed}_COMPLETE.zip"
            archive, member_manifest, _ = checked_archive(master.read(name), row["zip_sha256"])
            blob = archive.read("checkpoint.pt")
            assert digest(blob) == row["checkpoint_sha256"]
            state_blob = archive.read("latest_epoch_state.pt")
            saved = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            epoch = torch.load(io.BytesIO(state_blob), map_location="cpu", weights_only=False)
            cfg = json.loads(archive.read("training.json"))
            _, expected_cfg = training_config(protocol, b, seed)
            # The reviewed builder uses native Path; reproduce Linux provenance on Windows.
            for field in (
                "dataset_path",
                "gcn_checkpoint",
                "graphsage_checkpoint",
                "common_train_file",
                "common_val_file",
                "pgexplainer_config",
                "artifacts_root",
                "metrics_root",
            ):
                expected_cfg["task28_training"][field] = expected_cfg["task28_training"][
                    field
                ].replace("\\", "/")
            assert cfg == expected_cfg and saved["training_config"] == cfg["task28_training"]
            assert saved["backbone"] == ("GCN" if b == "gcn" else "GraphSAGE")
            assert (
                saved["seed"] == seed
                and saved["epochs_completed"] == 30
                and saved["algorithm_current_epoch"] == 29
            )
            assert epoch["epoch_completed"] == 30 and len(epoch["epoch_rows"]) == 30
            assert all(
                r["num_train_instances"] == 2000 and r["epoch"] == i + 1
                for i, r in enumerate(epoch["epoch_rows"])
            )
            assert (
                epoch["hashes"]["task_config"]
                == saved["hashes"]["task_config"]
                == digest(archive.read("training.json"))
            )
            assert (
                saved["train_node_ids"] == expected_train
                and saved["validation_node_ids"] == expected_val
            )
            for k, v in protocol["runner_input_hashes"].items():
                assert saved["hashes"][k] == epoch["hashes"][k] == v
            check_tensors(saved, b, 60000, protocol["optimizer_group"], torch)
            check_tensors(epoch, b, 60000, protocol["optimizer_group"], torch)
            assert all(
                torch.equal(v, epoch["algorithm_state_dict"][k])
                for k, v in saved["algorithm_state_dict"].items()
            )
            assert saved["config_identity"] == epoch["config_identity"]
            runtime = json.loads(archive.read("runtime.json"))
            assert runtime["runtime_contract"] == receipt["runtime_contract"]
            assert (
                runtime["hard_compatibility"]
                == saved["hard_compatibility"]
                == epoch["hard_compatibility"]
            )
            assert runtime["runtime_contract"]["source_identity"] == protocol["execution_identity"]
            assert json.loads(archive.read("protocol.json")) == protocol
            verify = json.loads(archive.read("integrity_verification.json"))
            assert verify["epochs"] == 30 and verify["optimizer_updates"] == 60000
            assert verify["epoch_state"]["sha256"] == digest(state_blob)
            assert verify["checkpoint_sha256"] == digest(blob)
            checkpoints[key] = saved
            inventory.append(
                {
                    "backbone": b,
                    "seed": seed,
                    "session": session,
                    "checkpoint_sha256": digest(blob),
                    "epoch_state_sha256": digest(state_blob),
                    "seed_zip_sha256": row["zip_sha256"],
                    "config_identity": saved["config_identity"],
                    "epochs": 30,
                    "updates": 60000,
                }
            )
    assert set(checkpoints) == {(b, s) for b in ("gcn", "graphsage") for s in range(5)}
    assert len({r["checkpoint_sha256"] for r in inventory}) == 10
    protected[str(RELEASE / "TASK29_KAGGLE_CODE_BUNDLE.zip")] = code_sha
    protected[str(RELEASE / "TASK29_ELLIPTIC_FROZEN_INPUTS.zip")] = input_sha
    return (
        protocol,
        targets,
        checkpoints,
        {
            "archives": archives,
            "checkpoints": inventory,
            "protected_files_before": protected,
            "evaluation_protocol": preflight["evaluation"],
            "target_ids": targets,
            "target_sha256": sha(target_path),
            "upload_manifest": upload,
        },
    )


def evaluate(context, full=False):
    import torch
    import torch_geometric
    from torch_geometric.nn import Linear
    from torch_geometric.utils import get_embeddings
    from prepare_elliptic_task26 import load_data, build_models
    from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask

    protocol, nodes, checkpoints, evidence = context
    assert platform.python_version() == "3.11.9" and torch.__version__ == "2.11.0+cu128"
    assert torch_geometric.__version__ == "2.8.0.post1" and torch.cuda.is_available()
    torch.set_num_threads(1)
    if not full:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cpu = load_data()
    assert cpu.edge_index.size(1) == 468710
    data = cpu.clone().to("cuda:0")
    evidence["runtime"] = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "pyg": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "deterministic": True,
        "threads": 1,
        "interop_threads": 1,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False,
        "precision": torch.get_float32_matmul_precision(),
        "inference_environment_identical_to_training": False,
        "difference": "Windows RTX4060 versus Linux T4; inference deterministic contract, not training flag replay",
    }
    historical = json.loads(
        (ROOT / "docs/audit/task28_pg_elliptic_full30_raw_final200_case_study.json").read_text()
    )
    candidate_reference = historical["per_target"]["GCN"]
    assert [r["node_id"] for r in candidate_reference] == nodes
    candidates = {}
    for node, reference in zip(nodes, candidate_reference, strict=True):
        mask = computation_subgraph_edge_mask(cpu, node, 2)
        ids = torch.nonzero(mask).flatten()
        assert ids.tolist() == reference["candidate_global_edge_indices"]
        candidates[node] = ids
    evidence["canonical_candidate_check"] = (
        "Exact IDs/order equal archived historical final200 for all200 targets"
    )
    evidence["graph"] = {
        "edges": data.edge_index.size(1),
        "nodes": cpu.num_nodes,
        "edge_id": "global directed edge_index column; parallel entries distinct; reverse direction distinct",
        "flow": "source_to_target",
        "hops": 2,
        "directed_argument": False,
        "duplicate_directed_endpoint_entries": int(
            cpu.edge_index.size(1) - torch.unique(cpu.edge_index, dim=1).size(1)
        ),
    }
    write(OUT / "TASK29_FINAL_EVALUATION_INTEGRITY.json", evidence)
    records = {}
    started = time.perf_counter()
    for b in ("gcn", "graphsage"):
        model = build_models(cpu)["gcn_compatible" if b == "gcn" else "graphsage"]
        model = model.to("cuda:0").eval().requires_grad_(False)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        embeddings = get_embeddings(model, data.x, data.edge_index)[-1].detach()
        assert embeddings.shape[1] == (90 if b == "gcn" else 103)
        modules = {}
        for seed in range(5):
            saved = checkpoints[b, seed]
            net = torch.nn.Sequential(
                Linear(3 * embeddings.shape[1], 64), torch.nn.ReLU(), Linear(64, 1)
            ).to("cuda:0")
            net.load_state_dict(
                {k.removeprefix("mlp."): v for k, v in saved["algorithm_state_dict"].items()}
            )
            modules[seed] = net.eval().requires_grad_(False)
        # Full-edge concat and GEMM order exactly as historical helper; reuse input across seeds only.
        for index, node in enumerate(nodes if full else nodes[:3]):
            target_file = OUT / "records" / b / f"{node}.json"
            record_identity = ident(
                {
                    "protocol": PROTOCOL_ID,
                    "runtime": evidence["runtime"],
                    "checkpoints": [r for r in evidence["checkpoints"] if r["backbone"] == b],
                    "source": sha(__file__),
                    "node": node,
                    "candidates": candidates[node].tolist(),
                }
            )
            if target_file.exists():
                cached = json.loads(target_file.read_text())
                validate_cached_record(cached, record_identity)
                assert (
                    sha(target_file)
                    == json.loads(target_file.with_suffix(".receipt.json").read_text())["sha256"]
                )
                records[b, node] = cached
                continue
            ids = candidates[node]
            inputs = torch.cat(
                [
                    embeddings[data.edge_index[0]],
                    embeddings[data.edge_index[1]],
                    embeddings[node].view(1, -1).repeat(data.edge_index.size(1), 1),
                ],
                dim=-1,
            )
            run_rows = []
            for seed, net in modules.items():
                raw = net(inputs).view(-1)[ids.to("cuda:0")].cpu()
                assert (
                    raw.dtype == torch.float32
                    and bool(torch.isfinite(raw).all())
                    and len(raw) == len(ids)
                )
                local, diagnostics = selection(raw, torch)
                selected = ids[local].sort().values.tolist()
                assert len(set(selected)) == len(selected) == max(1, math.ceil(0.2 * len(ids)))
                # Repeat all10 checkpoints for first three deterministic targets before wider coverage.
                if index < 3:
                    repeat = net(inputs).view(-1)[ids.to("cuda:0")].cpu()
                    assert torch.equal(raw, repeat)
                    assert torch.equal(local, torch.topk(repeat, k=len(local)).indices)
                row = {
                    "seed": seed,
                    "selected_global_edge_indices": selected,
                    "raw_logits": raw.tolist(),
                    "diagnostics": diagnostics,
                    "repeated_inference_exact": index < 3,
                }
                run_rows.append(row)
            record = {
                "identity": record_identity,
                "backbone": b,
                "node_id": node,
                "candidate_global_edge_indices": ids.tolist(),
                "runs": run_rows,
            }
            write(target_file, record)
            write(
                target_file.with_suffix(".receipt.json"),
                {"sha256": sha(target_file), "identity": record_identity},
            )
            records[b, node] = record
            if any(r["diagnostics"]["boundary_tie_crosses_k"] for r in run_rows):
                evidence["blocking_record"] = str(target_file)
                evidence["blocker"] = (
                    "EXACT_RAW_BOUNDARY_TIE: frozen protocol requires STOP, no approved deterministic tie resolution"
                )
                raise ValueError(evidence["blocker"])
            del inputs
            if index == 2:
                print(
                    f"SMOKE_PART {b}:15 checkpoint-target forwards+repeats PASS; elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if index % 20 == 0:
                print(
                    f"EVALUATED {b} {index + 1}/200; GPU peak={torch.cuda.max_memory_allocated() / 2**30:.3f}GiB",
                    flush=True,
                )
        assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items())
        for seed, net in modules.items():
            assert all(
                torch.equal(v.cpu(), checkpoints[b, seed]["algorithm_state_dict"]["mlp." + k])
                for k, v in net.state_dict().items()
            )
        del model, embeddings, modules
        torch.cuda.empty_cache()
    evidence["inference_wall_seconds"] = time.perf_counter() - started
    evidence["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if not full:
        print("ALL10_CHECKPOINTS_SMOKE_PASS; automatically proceeding to full200", flush=True)
        return evaluate(context, full=True)
    return records


def finish(records, evidence):
    pair_rows, target_rows, summaries, diagnostics = [], [], [], []
    for b in ("gcn", "graphsage"):
        for node in evidence["target_ids"]:
            record = records[b, node]
            runs = {r["seed"]: r for r in record["runs"]}
            values = []
            for a, c in itertools.combinations(range(5), 2):
                left, right = (
                    set(runs[a]["selected_global_edge_indices"]),
                    set(runs[c]["selected_global_edge_indices"]),
                )
                value = jaccard(left, right)
                pair_rows.append(
                    {
                        "backbone": b,
                        "node_id": node,
                        "seed_a": a,
                        "seed_b": c,
                        "intersection": len(left & right),
                        "union": len(left | right),
                        "jaccard": value,
                    }
                )
                values.append(value)
            target_rows.append(
                {
                    "backbone": b,
                    "node_id": node,
                    "num_pairs": 10,
                    "mean_pairwise_jaccard": statistics.mean(values),
                    "pair_sample_sd_descriptive": statistics.stdev(values),
                }
            )
            diagnostics.extend(
                {"backbone": b, "node_id": node, "seed": r["seed"], **r["diagnostics"]}
                for r in runs.values()
            )
        rows = [r for r in target_rows if r["backbone"] == b]
        summaries.append(
            {
                "backbone": b,
                "seeds": "0-4",
                "targets": 200,
                "edge_budget": "top20% RAW float32",
                **aggregate(rows),
                "status": "verified",
            }
        )
    # Independent path reads persisted selections and recomputes with a boolean membership vector.
    independent = {}
    for b in ("gcn", "graphsage"):
        target_values = []
        for node in evidence["target_ids"]:
            saved = json.loads((OUT / "records" / b / f"{node}.json").read_text())
            memberships = {
                r["seed"]: [
                    e in r["selected_global_edge_indices"]
                    for e in saved["candidate_global_edge_indices"]
                ]
                for r in saved["runs"]
            }
            values = [
                sum(x and y for x, y in zip(memberships[a], memberships[c], strict=True))
                / sum(x or y for x, y in zip(memberships[a], memberships[c], strict=True))
                for a, c in itertools.combinations(range(5), 2)
            ]
            target_values.append(sum(values) / 10)
        independent[b] = {
            "mean": sum(target_values) / 200,
            "sample_sd": math.sqrt(
                sum((v - sum(target_values) / 200) ** 2 for v in target_values) / 199
            ),
        }
        primary = next(s for s in summaries if s["backbone"] == b)
        assert abs(independent[b]["mean"] - primary["mean_jaccard"]) < 1e-14
        assert abs(independent[b]["sample_sd"] - primary["sample_sd"]) < 1e-14
    assert len(pair_rows) == 4000 and len(target_rows) == 400 and len(diagnostics) == 2000
    csv_write(OUT / "TASK29_FINAL_STABILITY_SUMMARY.csv", summaries)
    csv_write(OUT / "TASK29_FINAL_STABILITY_PER_TARGET.csv", target_rows)
    csv_write(OUT / "TASK29_FINAL_STABILITY_PAIRWISE.csv", pair_rows)
    write(
        OUT / "TASK29_RAW_LOGIT_RANKING_AUDIT.json",
        {
            "records": diagnostics,
            "near_tolerances": NEAR_TOLERANCES,
            "official_selections_changed_by_sensitivity": False,
            "tie_fraction_definition": "(n-unique_float32)/n",
        },
    )
    evidence["independent_recomputation"] = {"status": "PASS", "results": independent}
    evidence["coverage"] = {
        "targets_per_backbone": 200,
        "seed_target_records": 2000,
        "pair_rows": 4000,
    }
    write(
        OUT / "TASK29_FINAL_STABILITY_RESULTS.json",
        {
            "verdict": "TASK29_FINAL_STABILITY_COMPLETE_WITH_INTERPRETIVE_LIMITATIONS",
            "protocol_identity": PROTOCOL_ID,
            "results": summaries,
            "aggregation": "mean10 pairs per target; equal-target mean; sample SD(ddof1) across200 target means",
            "caveats": [
                "GCN diagnostic",
                "Stability is not Fidelity",
                "Single frozen classifier per backbone",
                "Final200 previously examined; replication not untouched holdout",
            ],
        },
    )


def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["ELLIPTIC_BASELINE_REPO"] = str(BASE)
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {
        "verdict": "TASK29_FINAL_STABILITY_BLOCKED",
        "evaluation_source_sha256": sha(__file__),
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_runs": 0,
        "official_stability_produced": False,
    }
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
            context = verify_inputs()
            evidence.update(context[3])
            context = (*context[:3], evidence)
            write(OUT / "TASK29_FINAL_EVALUATION_INTEGRITY.json", evidence)
            with torch.inference_mode():
                records = evaluate(context)
            finish(records, evidence)
            evidence["verdict"] = "TASK29_FINAL_STABILITY_COMPLETE_WITH_INTERPRETIVE_LIMITATIONS"
            evidence["official_stability_produced"] = True
    except Exception:
        evidence["traceback"] = traceback.format_exc()
        write(
            OUT / "TASK29_FINAL_STABILITY_RESULTS.json",
            {
                "verdict": "TASK29_FINAL_STABILITY_BLOCKED",
                "official_metrics": None,
                "reason": evidence.get("blocker", evidence["traceback"]),
                "training_required": False,
            },
        )
        print(evidence["traceback"], flush=True)
    finally:
        protected = evidence.get("protected_files_before", {})
        changed = [p for p, h in protected.items() if sha(p) != h]
        assert not changed, changed
        evidence["protected_files_after_verified"] = len(protected)
        evidence["unexpected_input_changes"] = changed
        evidence["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        write(OUT / "TASK29_FINAL_EVALUATION_INTEGRITY.json", evidence)
        print(evidence["verdict"], flush=True)
    return 0 if evidence["official_stability_produced"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
