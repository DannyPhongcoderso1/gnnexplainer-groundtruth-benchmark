"""Generate immutable upload artifacts from frozen Git bytes and original baseline inputs."""

import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from common import HERE, identity, matrix, read_json, require, sha, validate_session_config, utc
from release_inventory import approved_sources, inventory
from zip_utils import create_zip, verify_zip

COMMIT = "6034c13bfab601bef823c07194c70574cd49ec41"
ROOT = HERE.parents[2]
BASE = ROOT.parent / "elliptic-gnn-xai"
RUNTIME = [
    "aggregate_stability.py",
    "bootstrap.py",
    "common.py",
    "export_utils.py",
    "extract_raw.py",
    "orchestrator.py",
    "persistence.py",
    "preflight.py",
    "progress.py",
    "recovery.py",
    "resources.py",
    "restore.py",
    "smoke.py",
    "state_verify.py",
    "training_adapter.py",
    "zip_utils.py",
    "setup.sh",
]


def generated(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def notebook(session, code_sha, input_sha, code_manifest_sha="", input_manifest_sha=""):
    parameters = f"""import time
STARTED_UNIX = time.time()  # Includes bootstrap/setup in the12h budget.
SESSION = {session!r}  # This notebook is locked to this session.
MODE = "PREFLIGHT_ONLY"
AUTHORIZE_SMOKE = False
PRODUCTION_APPROVED = False
CODE_ZIP = "/kaggle/input/task29-code/TASK29_KAGGLE_CODE_BUNDLE.zip"
INPUTS_ZIP = "/kaggle/input/task29-frozen-inputs/TASK29_ELLIPTIC_FROZEN_INPUTS.zip"
APPROVED_SMOKE = ""  # Actual mounted smoke_receipt.json path after review.
APPROVED_SMOKE_SHA256 = ""
SESSION_A_RECEIPT = ""  # Required for SESSION_B FULL_TRAIN.
SESSION_A_RECEIPT_SHA256 = ""
RESTORE_ZIP = ""  # Optional verified same-session ALL.zip after interruption.
RESTORE_SHA256 = ""
EXTERNAL_SETTINGS = ""  # Disabled; optional separately reviewed upload settings JSON, never credentials.
"""
    launch = f"""import hashlib, json, zipfile
from pathlib import Path
assert SESSION == {session!r}, "Do not switch sessions in this notebook"
CODE_SHA256 = {code_sha!r}
INPUTS_SHA256 = {input_sha!r}
CODE_MANIFEST_SHA256 = {code_manifest_sha!r}
INPUTS_MANIFEST_SHA256 = {input_manifest_sha!r}
code_location = Path(CODE_ZIP)
if code_location.is_file():
    with code_location.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == CODE_SHA256, "Code ZIP SHA256 mismatch"
    with zipfile.ZipFile(code_location) as archive:
        bootstrap_source = archive.read("execution/bootstrap.py").decode("utf-8")
else:
    if not code_location.exists() and (code_location.parent / "ARCHIVE_MANIFEST.json").is_file():
        code_location = code_location.parent
    manifest_file = code_location / "ARCHIVE_MANIFEST.json"
    with manifest_file.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == CODE_MANIFEST_SHA256, "Direct code manifest mismatch"
    source_file = code_location / "execution/bootstrap.py"
    manifest = json.loads(manifest_file.read_text())
    with source_file.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["files"]["execution/bootstrap.py"]["sha256"], "Bootstrap mismatch"
    bootstrap_source = source_file.read_text(encoding="utf-8")
scope = {{"__name__": "task29_reviewed_bootstrap"}}
exec(compile(bootstrap_source, "<sha256-verified-task29-bootstrap>", "exec"), scope)
scope["start"]({{"session": SESSION, "mode": MODE, "authorize_smoke": AUTHORIZE_SMOKE,
    "production_approved": PRODUCTION_APPROVED, "code_zip": str(code_location), "code_sha256": CODE_SHA256,
    "code_manifest_sha256": CODE_MANIFEST_SHA256, "inputs_manifest_sha256": INPUTS_MANIFEST_SHA256,
    "inputs_zip": INPUTS_ZIP, "inputs_sha256": INPUTS_SHA256, "started_unix": STARTED_UNIX,
    "approved_smoke": APPROVED_SMOKE, "approved_smoke_sha256": APPROVED_SMOKE_SHA256,
    "session_a_receipt": SESSION_A_RECEIPT, "session_a_receipt_sha256": SESSION_A_RECEIPT_SHA256,
    "restore_zip": RESTORE_ZIP, "restore_sha256": RESTORE_SHA256,
    "external_settings": EXTERNAL_SETTINGS}})
"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# TASK29 SESSION {session}: Elliptic PGExplainer fresh5\n",
                "Defaults to PREFLIGHT_ONLY. Authorize smoke separately; review its receipt before FULL_TRAIN.\n",
                "GPU0=GraphSAGE; GPU1=GCN. Session B requires complete Session A evidence.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": parameters.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": launch.splitlines(keepends=True),
        },
    ]
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    for index, cell in enumerate(cells):
        cell["id"] = f"task29-{session.lower()}-{index}"
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python3", "language": "python", "name": "python3"}
        },
        "cells": cells,
    }


def build(output):
    import yaml

    output = Path(output).resolve()
    require(HERE / "dist" in output.parents, "Build output must be inside isolated Task29 dist/")
    require(not output.exists(), "Use a new build directory; never overwrite released archives")
    preflight = read_json(ROOT / "docs/audit/elliptic_pg_stability/preflight_manifest.json")
    require(
        all(sha(p) == h for p, h in preflight["protected_input_sha256"].items()),
        "Protected Step01 scientific artifact changed before build",
    )
    approved_sources(ignore_builder=True)
    pins = {}
    historical_lock = ROOT / "results/task28_primary_seed42_full_export/environment-lock.txt"
    for line in historical_lock.read_text().splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name] = version
    plan = read_json(ROOT / "docs/elliptic_pgexplainer_step04_kaggle_t4x2_plan.json")
    # The two build tools were audited/pinned by the successful setup plan, not listed in pip freeze.
    for name in ("setuptools", "wheel"):
        pins[name] = plan["runtime_dependencies"]["pins"][name]
    require(pins == read_json(HERE / "protocol.json")["dependency_pins"], "Pinned dependency drift")
    scientific = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", COMMIT, "--", "src"], cwd=ROOT, text=True
    ).splitlines()
    scientific = [
        n for n in scientific if n.endswith(".py") or n.endswith("LICENSE_UPSTREAM_GRAPHXAI")
    ]
    scientific += [
        "scripts/run_elliptic_pgexplainer_training.py",
        "scripts/prepare_elliptic_task26.py",
        "pyproject.toml",
        "README.md",
    ]
    live_frozen = [
        "configs/explainer/pgexplainer.yaml",
        "configs/evaluation/elliptic_pgexplainer_training.yaml",
        "configs/targets/elliptic_pg_common_train.csv",
        "configs/targets/elliptic_pg_common_val.csv",
        "configs/targets/elliptic_xai_common_test_targets.csv",
    ]
    baseline = [
        "datasets/elliptic/elliptic_data.pt",
        "artifacts/checkpoints/gcn_baseline.pt",
        "artifacts/checkpoints/graphsage_baseline.pt",
        "config/training_baseline_paper_gcn.yaml",
        "config/training_baseline_paper_sage.yaml",
    ]
    runtime_files = [HERE / n for n in RUNTIME]
    runtime_identity = identity({p.name: sha(p) for p in runtime_files})
    with tempfile.TemporaryDirectory(prefix="task29-build-") as temporary:
        temporary = Path(temporary)
        execution = temporary / "execution"
        execution.mkdir()
        # Stage generated trust assets; approved working-tree manifests stay historical.
        for name in RUNTIME + [
            "requirements-linux.txt",
            "historical-linux-lock.txt",
            "SESSION_A.json",
            "SESSION_B.json",
        ]:
            (execution / name).write_bytes((HERE / name).read_bytes())
        for session in ("A", "B"):
            validate_session_config(read_json(execution / f"SESSION_{session}.json"))
        (execution / "README.md").write_text(
            f"# TASK29 {output.name}\n\nThis immutable release contains the reviewed Step02.1 recovery fixes. "
            "Old dist/kaggle_ready and release_001 are stale, not this release.\n"
            "PREFLIGHT_ONLY is the default. Step03 smoke and production require separate explicit authorization.\n"
            "GPU0=GraphSAGE; GPU1=GCN; A=0,1,2; B=3,4. No automatic Session B.\n"
            "Tests and build tools are audit-only, excluded from the runtime import closure.\n"
            "Read IMPORT_RESULTS.md and the versioned release runbook before any execution.\n",
            encoding="utf-8",
        )
        (execution / "IMPORT_RESULTS.md").write_text(
            "# Download and Recovery Contract\n\nRead current_master.json and execution_summary.json.master for the selected verified archive/path/SHA. "
            "The first canonical ALL.zip is immutable and may be an older PARTIAL attempt. New attempts use "
            "master_attempts/<content_identity>/ with the same basename; preserve attempt folders when downloading.\n"
            "Download selected master, its external .zip.receipt.json, current_master.json, session_receipt.json, execution_summary.json, "
            "individual seed ZIPs/receipts, and the top-level task29-environment-lock.txt separately.\n"
            "Verify outer SHA/CRC/exact member hashes using trusted local zip_utils before extraction; COMPLETE requires A6/B4 seeds and full30/60000 state checks. "
            "A PARTIAL receipt cannot authorize B. Forensics are not admitted checkpoints.\n"
            "Restore only a reviewed same-session master/SHA with matching runtime/config; missing valid durable epoch blocks implicit restart.\n"
            "Exit20/21 denotes PARTIAL; preflight BLOCKED=23. Finalization occurs before nonzero exit where possible. "
            "Failed Kaggle versions may not preserve outputs: local verified ZIP is not external persistence or a hard-kill guarantee. "
            "External backup is disabled and not verified. No model-only Adam/RNG reset or seed substitution.\n",
            encoding="utf-8",
        )
        code_files = {}
        frozen = {"benchmark": {}, "baseline": {name: sha(BASE / name) for name in baseline}}
        for name in scientific:
            payload = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=ROOT)
            staged = temporary / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(payload)
            code_files["benchmark/" + name] = staged
            frozen["benchmark"][name] = sha(staged)
        for name in live_frozen:
            code_files["benchmark/" + name] = ROOT / name
            frozen["benchmark"][name] = sha(ROOT / name)
        for row in preflight["source_hash_checks"]:
            if row["name"] in (
                "pyg_pgexplainer",
                "gcn_model_construction_config",
                "sage_model_construction_config",
            ):
                continue
            name = Path(row["path"]).relative_to(ROOT).as_posix()
            payload = (
                Path(code_files["benchmark/" + name])
                .read_bytes()
                .replace(b"\r\n", b"\n")
                .replace(b"\r", b"\n")
            )
            require(
                hashlib.sha256(payload).hexdigest() == row["expected_historical_normalized_sha256"],
                "Frozen Git scientific source hash mismatch",
            )
        training = yaml.safe_load((ROOT / live_frozen[1]).read_text())["task28_training"]
        pg = yaml.safe_load((ROOT / live_frozen[0]).read_text())["explainer"]
        require(
            pg["regularization"] == {"edge_size": 0.05, "edge_ent": 1.0}
            and pg["epochs"] == 30
            and pg["learning_rate"] == 0.003,
            "Historical PG config mismatch",
        )
        protocol = {
            "schema": 1,
            "scientific_origin_commit": COMMIT,
            "execution_identity": runtime_identity,
            "method": "OTHER_ORIGINAL_PYG_HARD_CE_SUM",
            "fresh5": [0, 1, 2, 3, 4],
            "historical_training_config": training,
            "optimizer_group": preflight["optimizer"]["param_groups"][0],
            "dependency_pins": pins,
            "required_imports": [
                m
                for m in plan["runtime_dependencies"]["modules"]
                if not m.startswith("task28_kaggle")
            ],
            "pyg_pgexplainer_sha256": next(
                r["actual_normalized_sha256"]
                for r in preflight["source_hash_checks"]
                if r["name"] == "pyg_pgexplainer"
            ),
            "frozen_files": frozen,
            "runner_input_hashes": {
                "dataset": frozen["baseline"][baseline[0]],
                "gcn_checkpoint": frozen["baseline"][baseline[1]],
                "graphsage_checkpoint": frozen["baseline"][baseline[2]],
                "common_train": frozen["benchmark"][live_frozen[2]],
                "common_val": frozen["benchmark"][live_frozen[3]],
                "pgexplainer_config": frozen["benchmark"][live_frozen[0]],
            },
        }
        generated(execution / "protocol.json", protocol)
        generated(
            execution / "scientific_source_manifest.json",
            {
                "origin_commit": COMMIT,
                "files": frozen,
                "source_hash_policy": preflight["source_hash_policy"],
                "historical_normalized_sources": [
                    {"name": r["name"], "sha256": r["expected_historical_normalized_sha256"]}
                    for r in preflight["source_hash_checks"]
                ],
            },
        )
        for path in execution.iterdir():
            code_files["execution/" + path.name] = path
        closure = inventory(code_files, pins, protocol["required_imports"])
        generated(execution / "import_inventory.json", closure)
        execution_files = list(execution.iterdir())
        generated(
            execution / "execution_manifest.json",
            {"schema": 1, "files": {p.name: sha(p) for p in execution_files}},
        )
        execution_files.append(execution / "execution_manifest.json")
        for p in execution_files:
            code_files["execution/" + p.name] = p
        output.mkdir(parents=True)
        code_path = output / "TASK29_KAGGLE_CODE_BUNDLE.zip"
        input_path = output / "TASK29_ELLIPTIC_FROZEN_INPUTS.zip"
        code_sha = create_zip(
            code_path,
            code_files,
            {"kind": "CODE", "origin_commit": COMMIT, "protocol_identity": identity(protocol)},
        )
        input_sha = create_zip(
            input_path,
            {"baseline/" + name: BASE / name for name in baseline},
            {"kind": "FROZEN_INPUTS", "protocol_identity": identity(protocol)},
        )
        verify_zip(
            code_path,
            code_sha,
            required=[
                "execution/protocol.json",
                "execution/orchestrator.py",
                "execution/recovery.py",
                "execution/import_inventory.json",
                "benchmark/scripts/run_elliptic_pgexplainer_training.py",
            ],
        )
        verify_zip(input_path, input_sha, required=["baseline/" + name for name in baseline])
        import zipfile

        with zipfile.ZipFile(code_path) as archive:
            code_manifest_sha = hashlib.sha256(archive.read("ARCHIVE_MANIFEST.json")).hexdigest()
        with zipfile.ZipFile(input_path) as archive:
            input_manifest_sha = hashlib.sha256(archive.read("ARCHIVE_MANIFEST.json")).hexdigest()
        for session in ("A", "B"):
            generated(
                output / f"TASK29_SESSION_{session}.ipynb",
                notebook(session, code_sha, input_sha, code_manifest_sha, input_manifest_sha),
            )
        generated(
            output / "UPLOAD_MANIFEST.json",
            {
                "status": "UPLOAD_ZIPS_VERIFIED_OFFLINE",
                "protocol_identity": identity(protocol),
                "files": {
                    p.name: {"sha256": sha(p), "size_bytes": p.stat().st_size}
                    for p in [
                        code_path,
                        input_path,
                        output / "TASK29_SESSION_A.ipynb",
                        output / "TASK29_SESSION_B.ipynb",
                    ]
                },
                "actual_kaggle_gpu_smoke_executed": False,
                "full_train_default": False,
            },
        )
        # Detached inventory binds artifacts, not itself. External active pointer binds this inventory.
        old = read_json(HERE / "protocol.json")
        science = lambda p: identity({k: v for k, v in p.items() if k != "execution_identity"})
        require(science(old) == science(protocol), "Scientific/runtime dependency protocol drift")
        generated(
            output / ("TASK29_" + output.name.upper() + "_MANIFEST.json"),
            {
                "schema": 1,
                "release_id": output.name,
                "build_timestamp_utc": utc(),
                "client_audit_date": "2026-10-14",
                "status": "BUILT_PENDING_INDEPENDENT_VERIFICATION",
                "origin_commit": COMMIT,
                "packaging_generator_sha256": sha(Path(__file__)),
                "execution_identity_before": old["execution_identity"],
                "execution_identity": runtime_identity,
                "protocol_identity_before": identity(old),
                "protocol_identity": identity(protocol),
                "scientific_protocol_identity": science(protocol),
                "scientific_protocol_identity_before": science(old),
                "identity_change_reason": "Reviewed R1-R5 infrastructure code; scientific fields and pins unchanged. Runtime identity now excludes audit tests/build tools.",
                "execution_manifest_sha256": sha(execution / "execution_manifest.json"),
                "scientific_source_manifest_sha256": sha(
                    execution / "scientific_source_manifest.json"
                ),
                "dependency_inventory_identity": identity(pins),
                "import_inventory_identity": identity(closure),
                "scientific_files": frozen,
                "execution_files": read_json(execution / "execution_manifest.json")["files"],
                "source_file_count": sum(n.endswith(".py") for n in code_files),
                "files": read_json(output / "UPLOAD_MANIFEST.json")["files"],
                "archive_manifest_sha256": {
                    "code": code_manifest_sha,
                    "inputs": input_manifest_sha,
                },
                "session_matrix": {s: matrix(s) for s in ("A", "B")},
                "gpu_assignment": {"graphsage": 0, "gcn": 1},
                "r1_r5_source_sha256": {
                    n: sha(HERE / n)
                    for n in [
                        "bootstrap.py",
                        "orchestrator.py",
                        "training_adapter.py",
                        "recovery.py",
                        "restore.py",
                        "export_utils.py",
                        "zip_utils.py",
                    ]
                },
                "tests_shipped": False,
                "test_policy": "Trusted audit harness exercises the extracted runtime; no test modules are imported by training.",
                "authorization": {
                    "default_mode": "PREFLIGHT_ONLY",
                    "smoke_authorized": False,
                    "production_approved": False,
                    "external_backup": False,
                },
                "verification_receipt": "docs/audit/elliptic_pg_stability/TASK29_STEP02_2_VERIFICATION_RECEIPT.json",
                "hash_cycle_policy": "runtime file map -> execution_identity -> protocol/scientific/import manifests -> execution manifest (excludes self) -> archive manifests -> outer ZIPs -> notebooks -> detached release manifest -> verified external active pointer",
            },
        )
    print(json.dumps(read_json(output / "UPLOAD_MANIFEST.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
