"""Verified resume admission before scientific construction; rejected bytes are retained."""

import os
from pathlib import Path

from common import append_jsonl, atomic_json, identity, read_json, require, sha, state_paths, utc
from state_verify import verify_epoch


def preserve(path, root):
    path, root = Path(path), Path(root)
    require(path.is_file() and not path.is_symlink(), "Unsafe forensic source")
    root.mkdir(parents=True, exist_ok=True)
    digest = sha(path)
    destination = root / (path.name + "." + digest)
    if destination.exists():
        require(sha(destination) == digest, "Forensic content conflict")
        # Keep another immutable occurrence rather than unlinking a rejected file.
        number = 1
        while destination.with_name(destination.name + f".{number}").exists():
            number += 1
        destination = destination.with_name(destination.name + f".{number}")
    os.replace(path, destination)
    return {"original": str(path), "preserved": str(destination), "sha256": digest}


def resume_gate(
    directory,
    backbone,
    seed,
    config,
    protocol,
    runtime,
    identity_check,
    targets=2000,
    allow_restart=False,
):
    directory = Path(directory)
    require(
        runtime["runtime_contract"].get("protocol_identity") == identity(protocol),
        "Recovery protocol identity mismatch",
    )
    paths = state_paths(directory, backbone, seed)
    checkpoints = paths["run"] / "checkpoints"
    candidates = sorted(checkpoints.glob("epoch_*.pt"))
    receipts = list(directory.glob("verified_epoch_*.json"))
    if (directory / "durable_progress.json").exists():
        receipts.append(directory / "durable_progress.json")
    forensic = directory / "recovery_quarantine"
    rejected, verified = [], {}
    for path in receipts:
        try:
            row = read_json(path)
            epoch = row["epoch_completed"]
            require(type(epoch) is int and 1 <= epoch <= 30, "Receipt epoch invalid")
            checkpoint = checkpoints / f"epoch_{epoch:03d}.pt"
            require(row["path"] == str(checkpoint), "Receipt checkpoint path mismatch")
            require(
                row["status"] == "EPOCH_VERIFIED"
                and row["backbone"] == backbone
                and row["seed"] == seed,
                "Receipt run identity mismatch",
            )
            require(
                row["runtime_contract"] == runtime["runtime_contract"], "Receipt runtime mismatch"
            )
            require(
                checkpoint.is_file()
                and not checkpoint.is_symlink()
                and sha(checkpoint) == row["sha256"],
                "Receipt checkpoint SHA mismatch",
            )
            checked = verify_epoch(checkpoint, backbone, seed, config, protocol, runtime, targets)
            require(
                all(checked[k] == row[k] for k in checked), "Receipt state correspondence mismatch"
            )
            identity_check(checkpoint)
            verified[epoch] = row
        except Exception as exc:
            rejected.append({"reason": str(exc), **preserve(path, forensic / "receipts")})
    accepted = {Path(row["path"]) for row in verified.values()}
    for path in candidates:
        if path not in accepted:
            rejected.append(
                {
                    "reason": "Not bound to a valid durable receipt",
                    **preserve(path, forensic / "checkpoints"),
                }
            )
    selected = max(verified, default=0)
    prior = directory / "recovery_admission.json"
    previous = read_json(prior) if prior.exists() else {}
    history = previous.get("rejected", [])
    previous_started = previous.get("scientific_execution_started", False) or previous.get(
        "previous_execution_started", False
    )
    events = directory / "training_events.jsonl"
    stdout = directory / "stdout.log"
    had_state = bool(
        candidates
        or receipts
        or forensic.exists()
        or paths["run"].exists()
        or previous_started
        or (events.exists() and events.stat().st_size > 0)
        or (
            stdout.exists()
            and "TASK29_EVENT " in stdout.read_text(encoding="utf-8", errors="replace")
        )
    )
    status = "RESUME_VERIFIED" if selected else "FRESH_START"
    if not selected and had_state:
        status = "RESTART_EXPLICITLY_APPROVED" if allow_restart else "BLOCKED"
    admission = {
        "status": status,
        "backbone": backbone,
        "seed": seed,
        "epoch_completed": selected,
        "durable_updates": selected * targets,
        "verified_checkpoint_path": verified[selected]["path"] if selected else None,
        "protocol_identity": identity(protocol),
        "rejected": history + rejected,
        "allow_same_seed_restart": allow_restart,
        "previous_execution_started": bool(previous_started),
        "scientific_execution_started": False,
        "utc": utc(),
    }
    if selected:
        atomic_json(directory / "durable_progress.json", verified[selected])
    atomic_json(prior, admission)
    append_jsonl(directory / "recovery_admissions.jsonl", admission)
    require(
        status != "BLOCKED", "No valid durable state; same-seed restart requires explicit approval"
    )
    return admission


def begin_scientific_execution(directory, admission):
    require(admission["status"] != "BLOCKED", "Blocked admission cannot start scientific execution")
    started = {**admission, "scientific_execution_started": True, "utc": utc()}
    atomic_json(Path(directory) / "recovery_admission.json", started)
    append_jsonl(Path(directory) / "recovery_admissions.jsonl", started)
