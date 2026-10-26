"""Verified per-seed/session export. Local verification is not external persistence."""

import json
import zipfile
from pathlib import Path

from common import (
    GateError,
    atomic_json,
    identity,
    matrix,
    read_json,
    require,
    run_key,
    sha,
    state_paths,
    utc,
)
from zip_utils import create_zip, reusable_zip, verify_zip


def consistent_json(path, value):
    path = Path(path)
    require(not path.is_symlink(), "Symlink metadata rejected")
    if path.exists():
        require(read_json(path) == value, f"Conflicting immutable metadata: {path}")
    else:
        atomic_json(path, value, overwrite=False)


def files_under(directory, prefix, exclude=()):
    result = {}
    directory = Path(directory)
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.suffix not in (".tmp", ".partial"):
            relative = path.relative_to(directory).as_posix()
            if relative not in exclude:
                result[prefix + relative] = path
    return result


def seed_package(directory, backbone, seed, verification, session_dir, protocol_path):
    directory = Path(directory)
    transaction = directory / "publication_transaction.json"
    if not transaction.exists():
        atomic_json(transaction, {"status": "NOT_STARTED", "utc": utc()})
    try:
        atomic_json(transaction, {"status": "STAGING", "utc": utc()})
        result = _seed_package(directory, backbone, seed, verification, session_dir, protocol_path)
        atomic_json(transaction, {"status": "COMPLETE", "result": result, "utc": utc()})
        return result
    except Exception as exc:
        integrity_error = isinstance(exc, (GateError, zipfile.BadZipFile, json.JSONDecodeError))
        category = (
            "FAILED_INTEGRITY"
            if integrity_error and "disk" not in str(exc).lower()
            else "FAILED_RECOVERABLE"
        )
        evidence = {"status": category, "verification": verification, "error": str(exc)}
        try:
            consistent_json(
                directory / "publication_forensics" / (identity(evidence) + ".json"), evidence
            )
            atomic_json(transaction, {**evidence, "utc": utc()})
        except Exception:
            pass  # Preserve the original publication error even if the disk cannot record diagnostics.
        raise


def _seed_package(directory, backbone, seed, verification, session_dir, protocol_path):
    directory, session_dir = Path(directory), Path(session_dir)
    require(
        verification["status"] == "FINAL_VERIFIED",
        "Only verified full30 seeds can be packaged complete",
    )
    paths = state_paths(directory, backbone, seed)
    require(
        sha(paths["final"]) == verification["checkpoint_sha256"],
        "Final checkpoint hash changed before publication",
    )
    if "epoch_state" in verification:
        require(
            sha(paths["run"] / "checkpoints/epoch_030.pt") == verification["epoch_state"]["sha256"],
            "Verified epoch state changed before publication",
        )
    completion = {
        "status": "FULL30_VERIFIED_AWAITING_ARCHIVE_COMMIT",
        "backbone": backbone,
        "seed": seed,
        "verification": verification,
        "persistence": "LOCAL_ONLY_UNTIL_NOTEBOOK_FINALIZATION",
        "external_backup_confirmed": False,
    }
    marker = directory / "TASK29_COMPLETE.json"
    if marker.exists():
        result = verify_completed_package(directory)
        require(
            result["seed"] == seed
            and result["backbone"] == backbone
            and result["checkpoint_sha256"] == verification["checkpoint_sha256"],
            "Complete publication identity changed",
        )
        require(
            verify_zip(result["zip_path"], result["zip_sha256"])["metadata"] == completion,
            "Complete archive metadata conflict",
        )
        return result
    consistent_json(directory / "integrity_verification.json", verification)
    consistent_json(directory / "completion_manifest.json", completion)
    files = {
        "checkpoint.pt": paths["final"],
        "latest_epoch_state.pt": paths["run"] / "checkpoints/epoch_030.pt",
        "training.json": directory / "training.json",
        "runtime.json": directory / "runtime.json",
        "protocol.json": protocol_path,
        "completion_manifest.json": directory / "completion_manifest.json",
        "integrity_verification.json": directory / "integrity_verification.json",
        "durable_progress.json": directory / "durable_progress.json",
        "original_manifest.json": paths["manifest"],
        "original_COMPLETE.json": paths["run"] / "COMPLETE.json",
    }
    files.update(files_under(directory / "metrics", "metrics/"))
    for name in (
        "stdout.log",
        "stderr.log",
        "training_events.jsonl",
        "recovery_admission.json",
        "recovery_admissions.jsonl",
    ):
        if (directory / name).is_file():
            files["logs/" + name] = directory / name
    archive = session_dir / f"TASK29_{backbone.upper()}_SEED{seed}_COMPLETE.zip"
    atomic_json(directory / "publication_transaction.json", {"status": "VERIFYING", "utc": utc()})
    digest = reusable_zip(archive, files, completion, creator=create_zip)
    packaged = verify_zip(archive, digest)
    require(
        packaged["metadata"]["seed"] == seed
        and packaged["files"]["checkpoint.pt"]["sha256"] == verification["checkpoint_sha256"],
        "Seed ZIP identity/checkpoint mismatch",
    )
    if "epoch_state" in verification:
        require(
            packaged["files"]["latest_epoch_state.pt"]["sha256"]
            == verification["epoch_state"]["sha256"],
            "Seed ZIP verified epoch mismatch",
        )
    result = {
        "status": "COMPLETE",
        "backbone": backbone,
        "seed": seed,
        "checkpoint_path": str(paths["final"]),
        "checkpoint_sha256": verification["checkpoint_sha256"],
        "zip_path": str(archive),
        "zip_sha256": digest,
        "zip_size_bytes": archive.stat().st_size,
        "training_runtime_seconds": verification["training_runtime_seconds"],
        "external_backup_confirmed": False,
    }
    atomic_json(marker, result, overwrite=False)
    return result


def verify_completed_package(directory):
    marker = Path(directory) / "TASK29_COMPLETE.json"
    if not marker.exists():
        return None
    result = read_json(marker)
    require(
        sha(result["checkpoint_path"]) == result["checkpoint_sha256"],
        "Completed checkpoint changed",
    )
    verify_zip(result["zip_path"], result["zip_sha256"])
    require(result["status"] == "COMPLETE", "Non-complete completion marker")
    return result


def session_package(session_dir, receipt, recoverable, imports_guide, forensic_files=None):
    session_dir = Path(session_dir)
    require(receipt["session"] in ("A", "B"), "Invalid master session")
    if receipt["status"] == "COMPLETE":
        expected = sorted(r["run_key"] for r in matrix(receipt["session"]))
        require(
            sorted(receipt["completed_keys"]) == expected
            and sorted(run_key(r["backbone"], r["seed"]) for r in receipt["results"]) == expected
            and all(r["status"] == "COMPLETE" for r in receipt["results"]),
            "Incomplete master cannot claim COMPLETE",
        )
    index_path = session_dir / "current_master.json"
    previous = read_json(index_path) if index_path.exists() else None
    if previous:
        require(
            previous["session"] == receipt["session"], "Master directory belongs to another session"
        )
        verify_zip(previous["path"], previous["sha256"])
    atomic_json(session_dir / "session_receipt.json", receipt)
    files = {
        n: p
        for n, p in files_under(session_dir, "").items()
        if not (
            n.startswith(("master_attempts/", "publication_forensics/"))
            or n in ("current_master.json", "execution_summary.json")
            or n.startswith("TASK29_ELLIPTIC_STABILITY_SESSION_")
        )
    }
    for path in session_dir.glob("TASK29_ELLIPTIC_STABILITY_SESSION_*_ALL.zip"):
        require(
            path.name == f"TASK29_ELLIPTIC_STABILITY_SESSION_{receipt['session']}_ALL.zip",
            "Wrong-session canonical master in directory",
        )
    for name, path in files.items():
        if name.endswith(".zip"):
            nested = verify_zip(path)
            if "seed" in nested["metadata"]:
                require(0 <= nested["metadata"]["seed"] <= 4, "Invalid nested seed")
    for result in receipt.get("results", []):
        require(
            run_key(result["backbone"], result["seed"])
            in {r["run_key"] for r in matrix(receipt["session"])},
            "Nested seed belongs to another session",
        )
        nested = session_dir / Path(result["zip_path"]).name
        manifest = verify_zip(nested, result["zip_sha256"])
        require(
            manifest["metadata"]["seed"] == result["seed"]
            and manifest["metadata"]["backbone"] == result["backbone"],
            "Nested ZIP run mismatch",
        )
        if "checkpoint_sha256" in result:
            require(
                manifest["files"]["checkpoint.pt"]["sha256"] == result["checkpoint_sha256"],
                "Nested checkpoint SHA mismatch",
            )
    files["IMPORT_RESULTS.md"] = imports_guide
    files.update(forensic_files or {})
    for key, data in recoverable.items():
        for name, path in data.items():
            files[f"recoverable/{key}/{name}"] = path
    metadata = {
        "session": receipt["session"],
        "status": receipt["status"],
        "protocol_identity": receipt["protocol_identity"],
    }
    fingerprint = identity({"files": {n: sha(p) for n, p in files.items()}, "metadata": metadata})
    if previous and previous["content_identity"] == fingerprint:
        manifest = verify_zip(previous["path"], previous["sha256"])
        require(
            identity(
                {
                    "files": {n: r["sha256"] for n, r in manifest["files"].items()},
                    "metadata": manifest["metadata"],
                }
            )
            == fingerprint,
            "Master index fingerprint conflict",
        )
        return previous
    canonical = session_dir / f"TASK29_ELLIPTIC_STABILITY_SESSION_{receipt['session']}_ALL.zip"
    archive = (
        canonical
        if not canonical.exists() and not previous
        else session_dir / "master_attempts" / fingerprint / canonical.name
    )
    digest = reusable_zip(archive, files, metadata, creator=create_zip)
    verify_zip(archive, digest, required=["session_receipt.json"])
    result = {
        "path": str(archive),
        "sha256": digest,
        "size_bytes": archive.stat().st_size,
        "status": "MASTER_ZIP_VERIFIED_LOCAL",
        "external_backup_confirmed": False,
        "session": receipt["session"],
        "content_identity": fingerprint,
        "previous_master": None
        if not previous
        else {"path": previous["path"], "sha256": previous["sha256"]},
    }
    atomic_json(index_path, result)
    return result
