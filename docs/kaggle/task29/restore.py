"""Explicit verified restoration of a session ZIP; no scientific execution."""

import argparse
import shutil
from pathlib import Path

from common import (
    HERE,
    WORK,
    identity,
    matrix,
    read_json,
    require,
    sha,
    state_paths,
    training_config,
    validate_restore_receipt,
)
from zip_utils import extract_verified, verify_zip
from export_utils import consistent_json


def copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    require(source.is_file() and not source.is_symlink(), "Missing/symlink restore source")
    require(not destination.is_symlink(), "Symlink restore destination")
    if destination.exists():
        require(sha(source) == sha(destination), "Refusing incompatible existing restore output")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    require(sha(source) == sha(destination), "Restore SHA256 read-back mismatch")


def restore(archive, expected_sha, session, contract, work=WORK):
    protocol = read_json(HERE / "protocol.json")
    staging = Path(work) / "restore_staging" / expected_sha
    extract_verified(archive, staging, expected_sha)
    receipt = read_json(staging / "session_receipt.json")
    validate_restore_receipt(receipt, session, identity(protocol), contract)
    for row in matrix(session):
        backbone, seed, key = row["backbone"], row["seed"], row["run_key"]
        directory, config = training_config(protocol, backbone, seed, work=work)
        paths = state_paths(directory, backbone, seed)
        recovery = staging / "recoverable" / key
        completed = next(
            (r for r in receipt["results"] if r["backbone"] == backbone and r["seed"] == seed), None
        )
        if completed:
            seed_archive = staging / Path(completed["zip_path"]).name
            verify_zip(seed_archive, completed["zip_sha256"])
            inner = staging / ("seed_" + backbone + str(seed))
            extract_verified(seed_archive, inner, completed["zip_sha256"])
            require(
                read_json(inner / "training.json") == config, "Complete restore config mismatch"
            )
            for name, destination in [
                ("checkpoint.pt", paths["final"]),
                ("latest_epoch_state.pt", paths["run"] / "checkpoints/epoch_030.pt"),
                ("original_manifest.json", paths["manifest"]),
                ("original_COMPLETE.json", paths["run"] / "COMPLETE.json"),
                ("training.json", directory / "training.json"),
                ("runtime.json", directory / "runtime.json"),
                ("durable_progress.json", directory / "durable_progress.json"),
            ]:
                copy_exact(inner / name, destination)
            target_zip = Path(work) / "sessions" / session / seed_archive.name
            copy_exact(seed_archive, target_zip)
            completed.update(zip_path=str(target_zip), checkpoint_path=str(paths["final"]))
            consistent_json(directory / "TASK29_COMPLETE.json", completed)
        elif recovery.is_dir():
            require(
                read_json(recovery / "training.json") == config, "Partial restore config mismatch"
            )
            durable = read_json(recovery / "durable_progress.json")
            epoch = durable["epoch_completed"]
            target = paths["run"] / "checkpoints" / f"epoch_{epoch:03d}.pt"
            require(
                sha(recovery / "latest_epoch_state.pt") == durable["sha256"],
                "Recovery state hash mismatch",
            )
            require(durable["path"] == str(target), "Recovery epoch path identity mismatch")
            copy_exact(recovery / "latest_epoch_state.pt", target)
            for name in ("training.json", "runtime.json", "durable_progress.json"):
                copy_exact(recovery / name, directory / name)
            consistent_json(directory / f"verified_epoch_{epoch:03d}.json", durable)
    return staging / "session_receipt.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--session", choices=["A", "B"], required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    args = parser.parse_args()
    restore(args.archive, args.sha256, args.session, read_json(args.runtime_contract))
