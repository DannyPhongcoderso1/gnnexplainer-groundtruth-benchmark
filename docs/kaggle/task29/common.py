"""Task29 standard-library control plane; importing this module performs no work."""

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working/task29")
BENCHMARK = Path("/kaggle/working/task29-benchmark")
BASELINE = Path("/kaggle/working/elliptic-gnn-xai")
PYTHON = Path("/kaggle/working/task29-venv/bin/python")
GPU = {"graphsage": 0, "gcn": 1}
SEEDS = {"A": [0, 1, 2], "B": [3, 4]}
HISTORICAL_SECONDS = {"graphsage": 9211.966367751998, "gcn": 8998.261014752998}
GIB = 1024**3


class GateError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_finite(value)
    return value


def validate_finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite JSON value")
    elif isinstance(value, dict):
        for child in value.values():
            validate_finite(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            validate_finite(child)


def atomic_json(path, value, overwrite=True):
    path = Path(path)
    require(not path.is_symlink(), "Symlink output rejected")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        require(not path.exists(), f"Refusing overwrite: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    require(read_json(path) == json.loads(data), f"JSON read-back failed: {path}")


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_key(backbone, seed):
    require(backbone in GPU and type(seed) is int and seed in range(5), "Invalid fresh5 run")
    return f"{backbone}/seed_{seed}"


def matrix(session):
    require(session in SEEDS, "Session must be exactly A or B")
    return [
        {
            "session": session,
            "backbone": b,
            "physical_gpu": GPU[b],
            "seed": seed,
            "run_key": run_key(b, seed),
        }
        for b in GPU
        for seed in SEEDS[session]
    ]


def validate_session_config(config):
    require(config["schema"] == 1, "Unknown session schema")
    require(
        config["matrix"] == matrix(config["session"]), "Immutable session/GPU/seed matrix changed"
    )
    require(config["session_limit_seconds"] == 43200, "Reviewed session limit must remain12h")
    require(900 <= config["finalization_margin_seconds"] <= 7200, "Unsafe finalization margin")
    require(1 <= config["slowdown_factor"] <= 3, "Invalid runtime guard multiplier")
    require(30 <= config["resource_interval_seconds"] <= 60, "Invalid resource interval")
    require(50 <= config["target_heartbeat_interval"] <= 100, "Invalid target interval")
    require(30 <= config["heartbeat_seconds"] <= 60, "Invalid heartbeat interval")
    require(
        config["external_persistence"]["enabled"] is False,
        "Enable external persistence only in separately reviewed launch settings, not session template",
    )
    return config


def time_allows_seed(remaining, expected, margin, factor):
    return remaining >= expected * factor + margin


def validate_mode(
    mode,
    smoke_authorized=False,
    production_approved=False,
    approved_smoke_sha256="",
    smoke_receipt=None,
):
    require(mode in ("PREFLIGHT_ONLY", "SMOKE_TEST", "FULL_TRAIN"), "Unknown mode")
    if mode == "SMOKE_TEST":
        require(smoke_authorized is True, "SMOKE_TEST requires explicit smoke authorization")
    if mode == "FULL_TRAIN":
        require(
            production_approved is True, "FULL_TRAIN is disabled until explicit production approval"
        )
        require(
            len(approved_smoke_sha256) == 64 and smoke_receipt is not None,
            "Approved smoke receipt and its reviewed SHA256 are required",
        )
        require(sha(smoke_receipt) == approved_smoke_sha256, "Approved smoke receipt hash mismatch")
        require(
            read_json(smoke_receipt)["status"] == "SMOKE_PASS_REVIEW_REQUIRED",
            "Smoke receipt does not pass",
        )


def child_environment(backbone, baseline=BASELINE, benchmark=BENCHMARK):
    require(backbone in GPU, "Unknown GPU binding")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(GPU[backbone]),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        ELLIPTIC_BASELINE_REPO=str(baseline),
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join([str(benchmark), str(benchmark / "src"), str(HERE)]),
    )
    return env


def training_config(
    protocol, backbone, seed, work=WORK, benchmark=BENCHMARK, baseline=BASELINE, smoke_scope=None
):
    directory = Path(work) / ("smoke" if smoke_scope else "runs") / run_key(backbone, seed)
    if smoke_scope:
        directory = directory / smoke_scope
    cfg = dict(protocol["historical_training_config"])
    cfg.update(
        dataset_path=str(Path(baseline) / "datasets/elliptic/elliptic_data.pt"),
        gcn_checkpoint=str(Path(baseline) / "artifacts/checkpoints/gcn_baseline.pt"),
        graphsage_checkpoint=str(Path(baseline) / "artifacts/checkpoints/graphsage_baseline.pt"),
        common_train_file=str(Path(benchmark) / "configs/targets/elliptic_pg_common_train.csv"),
        common_val_file=str(Path(benchmark) / "configs/targets/elliptic_pg_common_val.csv"),
        pgexplainer_config=str(Path(benchmark) / "configs/explainer/pgexplainer.yaml"),
        artifacts_root=str(directory / "artifacts"),
        metrics_root=str(directory / "metrics"),
        task29_execution_identity=protocol["execution_identity"],
    )
    return directory, {"task28_training": cfg}


def state_paths(directory, backbone, seed):
    name = f"elliptic_pg_{backbone}_seed{seed}"
    return {
        "final": Path(directory) / "artifacts" / f"{name}.pt",
        "manifest": Path(directory) / "artifacts" / f"{name}_manifest.json",
        "run": Path(directory) / "artifacts" / f"{name}_run",
    }


def validate_session_a(receipt, protocol_identity, contract):
    require(
        receipt["session"] == "A" and receipt["status"] == "COMPLETE",
        "Session B requires a complete verified Session A receipt",
    )
    require(receipt["matrix"] == matrix("A"), "Session A matrix mismatch")
    require(receipt["protocol_identity"] == protocol_identity, "Cross-session protocol mismatch")
    require(receipt["runtime_contract"] == contract, "Cross-session runtime fingerprint mismatch")
    require(
        sorted(receipt["completed_keys"]) == sorted(r["run_key"] for r in matrix("A")),
        "Session A six seeds not complete",
    )
    require(
        sorted(run_key(r["backbone"], r["seed"]) for r in receipt["results"])
        == sorted(r["run_key"] for r in matrix("A")),
        "Session A result rows incomplete/duplicated/misassigned",
    )
    require(
        all(r["status"] == "COMPLETE" and r.get("zip_sha256") for r in receipt["results"]),
        "Missing verified per-seed ZIP receipts",
    )
    require(
        all(
            len(r["zip_sha256"]) == 64
            and Path(r["zip_path"]).name
            == f"TASK29_{r['backbone'].upper()}_SEED{r['seed']}_COMPLETE.zip"
            for r in receipt["results"]
        ),
        "Session A ZIP identity mismatch",
    )


def validate_restore_receipt(receipt, session, protocol_identity, contract):
    require(
        receipt["session"] == session and receipt["matrix"] == matrix(session),
        "Restore receipt belongs to another session/matrix",
    )
    require(
        receipt["protocol_identity"] == protocol_identity
        and receipt["runtime_contract"] == contract,
        "Incompatible restored runtime/protocol",
    )
