"""Two fixed-role workers, bounded scheduling, observational monitoring and verified exports."""

import argparse
import json
import os
import queue
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

from common import (
    BENCHMARK,
    GIB,
    GPU,
    HERE,
    PYTHON,
    SEEDS,
    WORK,
    append_csv,
    append_jsonl,
    atomic_json,
    child_environment,
    identity,
    matrix,
    read_json,
    require,
    run_key,
    sha,
    time_allows_seed,
    training_config,
    utc,
    validate_mode,
    validate_restore_receipt,
    validate_session_a,
    validate_session_config,
)
from export_utils import seed_package, session_package, verify_completed_package
from persistence import export_kaggle_dataset
from preflight import file_gate, run as preflight_run
from resources import snapshot
from zip_utils import create_zip


def exit_code(status):
    return {
        "COMPLETE": 0,
        "PREFLIGHT_ONLY": 0,
        "SMOKE_ONLY": 0,
        "PARTIAL": 20,
        "PARTIAL_OUTPUT_PACKAGING": 21,
        "FAILED": 22,
        "BLOCKED": 23,
    }[status]


def blocked_status(session, reason):
    directory = WORK / "sessions" / session
    receipt = {
        "session": session,
        "status": "BLOCKED",
        "reason": reason,
        "completed_keys": [],
        "incomplete_keys": [r["run_key"] for r in matrix(session)],
        "utc": utc(),
    }
    atomic_json(directory / "blocked_receipt.json", receipt)
    atomic_json(
        directory / "progress_status.json",
        {
            **receipt,
            "receipt_path": str(directory / "blocked_receipt.json"),
            "runs": {
                r["run_key"]: {
                    **r,
                    "status": "BLOCKED",
                    "last_committed_epoch": None,
                    "last_observed_target": None,
                    "verified_checkpoint_path": None,
                    "terminal_reason": reason,
                }
                for r in matrix(session)
            },
        },
    )
    return receipt


def command_for(backbone, seed, session, config, authorization):
    return [
        str(PYTHON),
        "-u",
        str(HERE / "training_adapter.py"),
        "--session",
        session,
        "--backbone",
        backbone,
        "--seed",
        str(seed),
        "--config",
        str(config),
        "--authorization",
        str(authorization),
        "--operation",
        "train",
    ]


def simulate(session, durations, fail=None, budget=43200, margin=3600, factor=1.25):
    """Deterministic scheduling simulation; no processes, Torch or scientific work."""
    result = []
    stop = (
        min(
            (
                sum(durations[b][s] for s in SEEDS[session][: SEEDS[session].index(seed) + 1])
                for b, seed in [fail]
            ),
            default=float("inf"),
        )
        if fail
        else float("inf")
    )
    for backbone in GPU:
        elapsed = 0
        for seed in SEEDS[session]:
            expected = max(durations[backbone][: seed + 1])
            if elapsed >= stop or not time_allows_seed(budget - elapsed, expected, margin, factor):
                result.append({"backbone": backbone, "seed": seed, "status": "NOT_STARTED"})
                continue
            elapsed += durations[backbone][seed]
            result.append(
                {
                    "backbone": backbone,
                    "seed": seed,
                    "status": "FAILED" if fail == (backbone, seed) else "COMPLETE",
                    "finish_seconds": elapsed,
                    "gpu": GPU[backbone],
                }
            )
    return result


def classify_failure(text, exit_code):
    if exit_code == 86:
        return "RECOVERABLE_INTERRUPTION"
    if any(
        x in text
        for x in (
            "hash mismatch",
            "identity mismatch",
            "Nonfinite",
            "Non-finite",
            "Integrity failure",
            "settings changed",
            "checkpoint",
        )
    ):
        return "SCIENTIFIC_INTEGRITY"
    if "out of memory" in text.lower() or "CUDA" in text or "dependency" in text:
        return "RUNTIME_ENVIRONMENT"
    return "WORKER_RUNTIME_FAILURE"


class Controller:
    def __init__(self, config, receipt, authorization, started_unix, external=None):
        self.config, self.receipt, self.authorization = config, receipt, authorization
        self.session = config["session"]
        self.directory = WORK / "sessions" / self.session
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started_unix = started_unix
        self.protocol = read_json(HERE / "protocol.json")
        self.external = external or {"enabled": False}
        self.events = queue.Queue()
        self.stop_new = threading.Event()
        self.active, self.status, self.results, self.errors = {}, {}, [], []
        self.lock = threading.Lock()
        self.last_progress = {}
        self.boundary_stop_at = None
        self.monitor_failures = 0
        self.persistence_lock = threading.Lock()
        self.external_queue = queue.Queue()
        self.external_done = threading.Event()
        self.prior_seconds = {}
        self.terminal_reason = None
        self.status = {
            r["run_key"]: {
                **r,
                "status": "PENDING",
                "last_committed_epoch": None,
                "last_observed_target": None,
                "verified_checkpoint_path": None,
            }
            for r in matrix(self.session)
        }
        for name in (
            "orchestrator.log",
            "gpu0_graphsage.log",
            "gpu1_gcn.log",
            "integrity.log",
            "training_events.jsonl",
            "failures.jsonl",
        ):
            (self.directory / name).touch(exist_ok=True)

    def log(self, text, integrity=False):
        with self.lock:
            line = f"{utc()} [SESSION_{self.session}] {text}\n"
            with (self.directory / ("integrity.log" if integrity else "orchestrator.log")).open(
                "a"
            ) as stream:
                stream.write(line)
            print(line.rstrip(), flush=True)

    def failure(self, backbone, seed, category, text):
        self.stop_new.set()
        self.terminal_reason = category + ": " + text
        row = {
            "utc": utc(),
            "backbone": backbone,
            "seed": seed,
            "category": category,
            "traceback_or_error": text,
        }
        with self.lock:
            self.errors.append(row)
            append_jsonl(self.directory / "failures.jsonl", row)
            if backbone in GPU and seed in SEEDS[self.session]:
                current = self.status[run_key(backbone, seed)]
                if current["status"] != "COMPLETE":
                    current["status"] = (
                        "PARTIAL" if category == "RECOVERABLE_INTERRUPTION" else "FAILED"
                    )
                current["terminal_reason"] = text
        self.log(f"{category} {backbone} seed{seed}; see failures.jsonl and worker logs")
        if category == "SCIENTIFIC_INTEGRITY":
            self.request_boundary_stop("System-wide integrity failure")
        self.publish_status()

    def publish_status(self, terminal=None, receipt=None):
        with self.lock:
            completed = sorted(run_key(r["backbone"], r["seed"]) for r in self.results)
            if terminal:
                for key, row in self.status.items():
                    if row["status"] not in ("COMPLETE", "FAILED", "BLOCKED"):
                        row["status"] = "PARTIAL"
                    row.setdefault("terminal_reason", self.terminal_reason or terminal)
            snapshot = {
                "session": self.session,
                "status": terminal or "RUNNING",
                "utc": utc(),
                "runs": self.status,
                "completed_keys": completed,
                "incomplete_keys": sorted(set(self.status) - set(completed)),
                "failed_keys": sorted(k for k, r in self.status.items() if r["status"] == "FAILED"),
                "terminal_reason": (
                    self.terminal_reason
                    or (
                        "All expected seed packages verified"
                        if terminal == "COMPLETE"
                        else "Incomplete matrix; review session receipt"
                    )
                )
                if terminal
                else None,
                "receipt_path": str(self.directory / "session_receipt.json"),
                "elapsed_session_seconds": time.time() - self.started_unix,
            }
            if receipt:
                snapshot["status"] = receipt["status"]
            atomic_json(self.directory / "progress_status.json", snapshot)

    def request_boundary_stop(self, reason):
        self.stop_new.set()
        self.terminal_reason = reason
        if self.boundary_stop_at is None:
            self.boundary_stop_at = time.monotonic()
            atomic_json(WORK / "STOP_AT_EPOCH_BOUNDARY", {"reason": reason, "utc": utc()})
            self.log(
                "STOP_NEW_SCHEDULING; stop active workers after next verified epoch: " + reason
            )

    def pipe(self, process, stream, backbone, seed, destination):
        try:
            with destination.open("a", encoding="utf-8") as output:
                for line in stream:
                    output.write(line)
                    output.flush()
                    with self.lock:
                        with (self.directory / f"gpu{GPU[backbone]}_{backbone}.log").open(
                            "a"
                        ) as merged:
                            merged.write(
                                f"[SESSION_{self.session}][GPU{GPU[backbone]}][{backbone}][SEED={seed}] {line}"
                            )
                    if line.startswith("TASK29_EVENT "):
                        event = json.loads(line[len("TASK29_EVENT ") :])
                        append_jsonl(destination.parent / "training_events.jsonl", event)
                        self.events.put(event)
        except Exception:
            self.events.put(
                {
                    "event": "PIPE_ERROR",
                    "backbone": backbone,
                    "seed": seed,
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            stream.close()

    def worker(self, backbone):
        expected = max(
            self.config["historical_seconds"][backbone], self.prior_seconds.get(backbone, 0)
        )
        for seed in SEEDS[self.session]:
            directory, config = training_config(self.protocol, backbone, seed)
            try:
                directory.mkdir(parents=True, exist_ok=True)
                config_path = directory / "training.json"
                if config_path.exists():
                    require(read_json(config_path) == config, "Restored training config changed")
                else:
                    atomic_json(config_path, config, overwrite=False)
                completed = verify_completed_package(directory)
                if completed:
                    require(
                        completed["backbone"] == backbone and completed["seed"] == seed,
                        "Completion marker belongs to another run",
                    )
                    require(
                        read_json(directory / "runtime.json")["runtime_contract"]
                        == self.receipt["runtime_contract"],
                        "Completed run fingerprint changed",
                    )
                    with self.lock:
                        self.results.append(completed)
                        self.status[run_key(backbone, seed)].update(
                            status="COMPLETE",
                            last_committed_epoch=30,
                            verified_checkpoint_path=completed["checkpoint_path"],
                            terminal_reason="Verified existing complete seed",
                        )
                    self.publish_status()
                    self.log(f"ALREADY_COMPLETE {backbone} seed{seed}; no retraining")
                    continue
                # Publication retry performs only CPU verification/ZIP work, never retrains a published final state.
                from common import state_paths

                if (state_paths(directory, backbone, seed)["run"] / "COMPLETE.json").exists():
                    from state_verify import verify_final

                    runtime = read_json(directory / "runtime.json")
                    require(
                        runtime["runtime_contract"] == self.receipt["runtime_contract"],
                        "Publication retry runtime mismatch",
                    )
                    file_gate(self.protocol)
                    verification = verify_final(
                        directory, backbone, seed, config_path, self.protocol, runtime
                    )
                    completed = seed_package(
                        directory,
                        backbone,
                        seed,
                        verification,
                        self.directory,
                        HERE / "protocol.json",
                    )
                    with self.lock:
                        self.results.append(completed)
                        self.status[run_key(backbone, seed)].update(
                            status="COMPLETE",
                            last_committed_epoch=30,
                            verified_checkpoint_path=completed["checkpoint_path"],
                            terminal_reason="Publication recovered without training",
                        )
                    self.publish_status()
                    continue
                if self.stop_new.is_set():
                    break
                remaining = self.config["session_limit_seconds"] - (time.time() - self.started_unix)
                if not time_allows_seed(
                    remaining,
                    expected,
                    self.config["finalization_margin_seconds"],
                    self.config["slowdown_factor"],
                ):
                    self.stop_new.set()
                    self.terminal_reason = "New seed admission blocked by time budget"
                    self.log(
                        f"TIME_GUARD {backbone} seed{seed} not started; remaining_s={remaining:.0f}"
                    )
                    break
                key = run_key(backbone, seed)
                process = subprocess.Popen(
                    command_for(backbone, seed, self.session, config_path, self.authorization),
                    cwd=BENCHMARK,
                    env=child_environment(backbone),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                with self.lock:
                    self.active[backbone] = {
                        "process": process,
                        "seed": seed,
                        "directory": directory,
                    }
                    self.status[key].update(
                        status="STARTING", pid=process.pid, physical_gpu=GPU[backbone]
                    )
                    self.last_progress[key] = time.monotonic()
                relays = [
                    threading.Thread(
                        target=self.pipe, args=(process, stream, backbone, seed, directory / name)
                    )
                    for stream, name in [
                        (process.stdout, "stdout.log"),
                        (process.stderr, "stderr.log"),
                    ]
                ]
                for relay in relays:
                    relay.start()
                code = process.wait()
                for relay in relays:
                    relay.join()
                with self.lock:
                    self.active.pop(backbone, None)
                if code:
                    text = (directory / "stderr.log").read_text(encoding="utf-8")[-50000:]
                    self.failure(
                        backbone, seed, classify_failure(text, code), text or f"Worker exit{code}"
                    )
                    break
                from state_verify import verify_final

                runtime = read_json(directory / "runtime.json")
                file_gate(self.protocol)
                verification = verify_final(
                    directory, backbone, seed, config_path, self.protocol, runtime
                )
                self.log(
                    f"FINAL_VERIFIED {backbone} seed{seed} SHA256={verification['checkpoint_sha256']}",
                    integrity=True,
                )
                completed = seed_package(
                    directory, backbone, seed, verification, self.directory, HERE / "protocol.json"
                )
                with self.lock:
                    self.results.append(completed)
                    self.status[key].update(
                        {
                            "status": "COMPLETE",
                            "epoch": 30,
                            "optimizer_updates": 60000,
                            "durable_updates": 60000,
                            "zip_sha256": completed["zip_sha256"],
                            "last_committed_epoch": 30,
                            "verified_checkpoint_path": completed["checkpoint_path"],
                            "terminal_reason": "Verified full30 archive committed",
                        }
                    )
                self.publish_status()
                try:
                    with self.persistence_lock:
                        persisted = export_kaggle_dataset(
                            completed["zip_path"], self.external, self.directory
                        )
                except Exception:
                    persisted = {
                        "status": "EXTERNAL_FAILED_OR_UNCONFIRMED",
                        "external_backup_confirmed": False,
                    }
                    self.failure(backbone, seed, "EXTERNAL_PERSISTENCE", traceback.format_exc())
                completed["external_persistence"] = persisted
                completed["external_backup_confirmed"] = persisted["external_backup_confirmed"]
                atomic_json(directory / "persistence_receipt.json", persisted)
                expected = max(expected, verification["training_runtime_seconds"])
                self.log(
                    f"COMPLETE {backbone} seed{seed}; ZIP={completed['zip_path']} SHA256={completed['zip_sha256']}"
                )
            except Exception:
                category = (
                    "OUTPUT_PACKAGING"
                    if (directory / "completion_manifest.json").exists()
                    else "SCIENTIFIC_INTEGRITY"
                )
                self.failure(backbone, seed, category, traceback.format_exc())
                break

    def event(self, event):
        if event["event"] == "PIPE_ERROR":
            self.failure(event["backbone"], event["seed"], "OUTPUT_LOGGING", event["traceback"])
            return
        key = run_key(event["backbone"], event["seed"])
        with self.lock:
            append_jsonl(self.directory / "training_events.jsonl", event)
            current = self.status.setdefault(key, {})
            if current.get("status") != "COMPLETE":
                updates = dict(event)
                if current.get("status") in ("FAILED", "PARTIAL", "BLOCKED"):
                    updates.pop("status", None)
                current.update(updates)
                if event["event"] == "TARGET_PROGRESS":
                    current["last_observed_target"] = event["target_completed"]
                if event["event"] == "EPOCH_VERIFIED":
                    current.update(
                        last_committed_epoch=event["epoch"],
                        verified_checkpoint_path=event["checkpoint_path"],
                    )
            self.last_progress[key] = time.monotonic()
            if event["event"] == "EPOCH_VERIFIED":
                append_csv(
                    self.directory / "epoch_summary.csv",
                    {
                        k: event[k]
                        for k in (
                            "session",
                            "physical_gpu",
                            "backbone",
                            "seed",
                            "epoch",
                            "target_completed",
                            "durable_updates",
                            "mean_loss",
                            "epoch_duration_seconds",
                            "cumulative_training_seconds",
                            "learning_rate",
                            "temperature",
                            "finite",
                            "checkpoint_path",
                            "checkpoint_sha256",
                        )
                    },
                )
                if self.external.get("enabled"):
                    # External snapshot work is delegated after a completed epoch, never GPU-synchronized.
                    append_jsonl(self.directory / "external_snapshot_queue.jsonl", event)
                    self.external_queue.put(event)
        self.publish_status()
        if event["event"] == "TARGET_PROGRESS":
            self.log(
                f"[GPU{GPU[event['backbone']]}][{event['backbone']}][SEED={event['seed']}] "
                f"epoch={event['epoch']}/30 | target={event['target_completed']}/2000 | "
                f"progress={event['epoch_percent']:.1f}% | updates={event['optimizer_updates']}/60000 | "
                f"elapsed={event['elapsed_seconds']:.0f}s | status=TRAINING | "
                f"ETA={event['estimated_remaining_seconds']:.0f}s"
            )

    def resources(self):
        with self.lock:
            active = {
                b: {"pid": v["process"].pid, "seed": v["seed"]} for b, v in self.active.items()
            }
        try:
            data = snapshot(WORK, active)
            for gpu in data["gpus"]:
                append_csv(
                    self.directory / "resources.csv",
                    {
                        "utc": utc(),
                        "session": self.session,
                        "elapsed_seconds": time.time() - self.started_unix,
                        **gpu,
                        **data["memory"],
                        "disk_free_bytes": data["disk_free_bytes"],
                        "worker_pids": json.dumps(active),
                        "gpu_processes": json.dumps(data["gpu_processes"]),
                    },
                )
            if data["disk_free_bytes"] < 2 * GIB or data["memory"]["ram_available_bytes"] < GIB:
                self.request_boundary_stop("Disk/RAM safety reserve exhausted")
            self.monitor_failures = 0
        except Exception:
            self.monitor_failures += 1
            self.log("RESOURCE_MONITOR_UNAVAILABLE " + traceback.format_exc())
            if self.monitor_failures >= 3:
                self.request_boundary_stop("Three consecutive hardware monitor failures")
        now = time.monotonic()
        for backbone, row in active.items():
            key = run_key(backbone, row["seed"])
            age = now - self.last_progress.get(key, now)
            if age > self.config["stale_heartbeat_seconds"]:
                self.log(
                    f"STALE_HEARTBEAT {key} age_s={age:.0f}; diagnosing without utilization-based kill"
                )
        if (
            time.time() - self.started_unix
            >= self.config["session_limit_seconds"] - self.config["finalization_margin_seconds"]
        ):
            self.request_boundary_stop("Session finalization margin reached")
        if (
            self.boundary_stop_at
            and now - self.boundary_stop_at > self.config["boundary_stop_grace_seconds"]
        ):
            with self.lock:
                for row in self.active.values():
                    if row["process"].poll() is None:
                        os.killpg(
                            row["process"].pid,
                            signal.SIGKILL
                            if now - self.boundary_stop_at
                            > self.config["boundary_stop_grace_seconds"] + 60
                            else signal.SIGTERM,
                        )
            self.log(
                "TIME_GUARD_TERMINATION; latest verified completed epochs remain authoritative"
            )

    def external_snapshots(self):
        while not self.external_done.is_set() or not self.external_queue.empty():
            try:
                event = self.external_queue.get(timeout=1)
            except queue.Empty:
                continue
            if (
                time.time() - self.started_unix
                >= 43200 - self.config["finalization_margin_seconds"]
            ):
                append_jsonl(
                    self.directory / "external_persistence.jsonl",
                    {
                        "status": "EXTERNAL_SNAPSHOT_SKIPPED_TIME_GUARD",
                        "epoch": event["epoch"],
                        "external_backup_confirmed": False,
                    },
                )
                continue
            try:
                key = run_key(event["backbone"], event["seed"])
                directory = WORK / "runs" / key
                require(
                    sha(event["checkpoint_path"]) == event["checkpoint_sha256"],
                    "Snapshot epoch hash mismatch",
                )
                archive = (
                    self.directory
                    / "snapshots"
                    / (f"{event['backbone']}_seed{event['seed']}_epoch{event['epoch']:03d}.zip")
                )
                create_zip(
                    archive,
                    {
                        "latest_epoch_state.pt": event["checkpoint_path"],
                        "training.json": directory / "training.json",
                        "runtime.json": directory / "runtime.json",
                        "durable_progress.json": directory
                        / f"verified_epoch_{event['epoch']:03d}.json",
                    },
                    {"kind": "RECOVERY_SNAPSHOT", "run_key": key, "epoch": event["epoch"]},
                )
                with self.persistence_lock:
                    result = export_kaggle_dataset(archive, self.external, self.directory)
                append_jsonl(self.directory / "external_persistence.jsonl", result)
            except Exception:
                append_jsonl(
                    self.directory / "external_persistence.jsonl",
                    {
                        "status": "EXTERNAL_SNAPSHOT_FAILED_OR_UNCONFIRMED",
                        "external_backup_confirmed": False,
                        "error": traceback.format_exc(),
                    },
                )

    def recoverable(self):
        require(
            all(row["process"].poll() is not None for row in self.active.values()),
            "Cannot package recovery states while workers are active",
        )
        result = {}
        for row in matrix(self.session):
            directory = self.directory.parent.parent / "runs" / row["run_key"]
            if (directory / "TASK29_COMPLETE.json").exists():
                continue
            durable = directory / "durable_progress.json"
            if durable.is_file():
                receipt = read_json(durable)
                checkpoint = Path(receipt["path"])
                require(sha(checkpoint) == receipt["sha256"], "Recoverable epoch SHA256 changed")
                result[row["run_key"]] = {
                    "latest_epoch_state.pt": checkpoint,
                    "durable_progress.json": durable,
                    "training.json": directory / "training.json",
                    "runtime.json": directory / "runtime.json",
                }
                for name in ("stdout.log", "stderr.log"):
                    if (directory / name).is_file():
                        result[row["run_key"]][name] = directory / name
        return result

    def forensics(self):
        from export_utils import files_under

        result = {}
        for row in matrix(self.session):
            directory = self.directory.parent.parent / "runs" / row["run_key"]
            prefix = "forensics/" + row["run_key"] + "/"
            for name in ("recovery_quarantine", "publication_forensics"):
                result.update(files_under(directory / name, prefix + name + "/"))
            for name in (
                "recovery_admission.json",
                "recovery_admissions.jsonl",
                "publication_transaction.json",
            ):
                path = directory / name
                if path.is_file() and not path.is_symlink():
                    result[prefix + name] = path
        return result

    def finish(self):
        expected = {r["run_key"] for r in matrix(self.session)}
        completed = {run_key(r["backbone"], r["seed"]) for r in self.results}
        receipt = {
            "schema": 1,
            "session": self.session,
            "status": "COMPLETE" if completed == expected and not self.errors else "PARTIAL",
            "matrix": matrix(self.session),
            "protocol_identity": identity(self.protocol),
            "runtime_contract": self.receipt["runtime_contract"],
            "completed_keys": sorted(completed),
            "incomplete_keys": sorted(expected - completed),
            "results": self.results,
            "failures": self.errors,
            "elapsed_session_seconds": time.time() - self.started_unix,
            "total_training_gpu_seconds": sum(r["training_runtime_seconds"] for r in self.results),
            "persistence": "ZIP_VERIFIED_LOCAL; notebook output persistence requires platform finalization",
            "utc": utc(),
        }
        atomic_json(self.directory / "session_receipt.json", receipt)
        atomic_json(self.directory / "session_run_matrix.json", matrix(self.session))
        atomic_json(self.directory / "input_source_manifest.json", self.protocol)
        self.publish_status(receipt["status"], receipt)
        try:
            master = session_package(
                self.directory,
                receipt,
                self.recoverable(),
                HERE / "IMPORT_RESULTS.md",
                self.forensics(),
            )
        except Exception:
            self.failure("session", -1, "OUTPUT_PACKAGING", traceback.format_exc())
            master = {"status": "MASTER_ZIP_FAILED", "individual_seed_zips_retained": True}
            receipt["status"] = "PARTIAL_OUTPUT_PACKAGING"
            atomic_json(self.directory / "session_receipt.json", receipt)
        atomic_json(
            self.directory / "execution_summary.json", {"receipt": receipt, "master": master}
        )
        self.publish_status(receipt["status"], receipt)
        self.log(
            json.dumps(
                {
                    "session_status": receipt["status"],
                    "completed": receipt["completed_keys"],
                    "incomplete": receipt["incomplete_keys"],
                    "master": master,
                    "next_action": "Download verified outputs; review Session A before launching B"
                    if self.session == "A"
                    else "Download all fresh5 outputs; separately authorize later evaluation",
                }
            )
        )
        return receipt

    def run(self):
        workers = [threading.Thread(target=self.worker, args=(b,)) for b in GPU]
        self.workers = workers
        external = None
        if self.external.get("enabled"):
            external = threading.Thread(target=self.external_snapshots)
            external.start()
        for thread in workers:
            thread.start()
        last_monitor = 0
        while any(thread.is_alive() for thread in workers) or not self.events.empty():
            try:
                self.event(self.events.get(timeout=1))
            except queue.Empty:
                pass
            if time.monotonic() - last_monitor >= self.config["resource_interval_seconds"]:
                self.resources()
                last_monitor = time.monotonic()
        for thread in workers:
            thread.join()
        self.external_done.set()
        if external:
            external.join()
        return self.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=["A", "B"], required=True)
    parser.add_argument(
        "--mode", choices=["PREFLIGHT_ONLY", "SMOKE_TEST", "FULL_TRAIN"], default="PREFLIGHT_ONLY"
    )
    parser.add_argument("--authorize-smoke", action="store_true")
    parser.add_argument("--production-approved", action="store_true")
    parser.add_argument("--approved-smoke", type=Path)
    parser.add_argument("--approved-smoke-sha256", default="")
    parser.add_argument("--session-a-receipt", type=Path)
    parser.add_argument("--session-a-receipt-sha256", default="")
    parser.add_argument("--restore-receipt", type=Path)
    parser.add_argument("--restore-archive", type=Path)
    parser.add_argument("--restore-sha256", default="")
    parser.add_argument("--external-settings", type=Path)
    parser.add_argument("--started-unix", type=float, required=True)
    args = parser.parse_args()
    validate_mode(
        args.mode,
        args.authorize_smoke,
        args.production_approved,
        args.approved_smoke_sha256,
        args.approved_smoke,
    )
    config = validate_session_config(read_json(HERE / f"SESSION_{args.session}.json"))
    require(0 <= time.time() - args.started_unix < 43200, "Invalid notebook session-start clock")
    import fcntl

    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "session.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            receipt = preflight_run(args.session)
        except Exception:
            blocked_status(args.session, traceback.format_exc())
            return exit_code("BLOCKED")
        if args.mode == "PREFLIGHT_ONLY":
            print(json.dumps(receipt, indent=2))
            atomic_json(
                WORK / "sessions" / args.session / "mode_result.json",
                {"status": "PREFLIGHT_ONLY", "production_complete": False},
            )
            return exit_code("PREFLIGHT_ONLY")
        protocol = read_json(HERE / "protocol.json")
        if args.restore_archive:
            require(
                args.mode == "FULL_TRAIN" and len(args.restore_sha256) == 64,
                "Restore archive requires FULL_TRAIN review and SHA256",
            )
            from restore import restore

            args.restore_receipt = restore(
                args.restore_archive, args.restore_sha256, args.session, receipt["runtime_contract"]
            )
        if args.mode == "FULL_TRAIN":
            smoke = read_json(args.approved_smoke)
            require(
                smoke["runtime_contract"] == receipt["runtime_contract"]
                and smoke["protocol_identity"] == receipt["protocol_identity"],
                "Approved smoke does not match actual production runtime",
            )
        if args.session == "B" and args.mode == "FULL_TRAIN":
            require(
                args.session_a_receipt is not None and len(args.session_a_receipt_sha256) == 64,
                "Session B requires reviewed Session A receipt/hash",
            )
            require(
                sha(args.session_a_receipt) == args.session_a_receipt_sha256,
                "Session A receipt hash mismatch",
            )
            validate_session_a(
                read_json(args.session_a_receipt), identity(protocol), receipt["runtime_contract"]
            )
        if args.restore_receipt:
            validate_restore_receipt(
                read_json(args.restore_receipt),
                args.session,
                identity(protocol),
                receipt["runtime_contract"],
            )
        if (WORK / "STOP_AT_EPOCH_BOUNDARY").exists():
            require(
                args.restore_receipt is not None,
                "Stale stop marker: explicit verified restore required",
            )
            (WORK / "STOP_AT_EPOCH_BOUNDARY").unlink()
        approval = {
            "mode": args.mode,
            "allowed_seeds": [0] if args.mode == "SMOKE_TEST" else SEEDS[args.session],
            "runtime_contract": receipt["runtime_contract"],
            "protocol_identity": identity(protocol),
            "target_heartbeat_interval": config["target_heartbeat_interval"],
            "heartbeat_seconds": config["heartbeat_seconds"],
        }
        authorization = WORK / "sessions" / args.session / "authorization.json"
        atomic_json(authorization, approval)
        if args.mode == "SMOKE_TEST":
            from smoke import run

            smoke_output = WORK / "sessions" / args.session / "smoke_results"
            result = run(args.session, authorization, smoke_output)
            archive = WORK / "sessions" / args.session / f"TASK29_SESSION_{args.session}_SMOKE.zip"
            from export_utils import files_under

            create_zip(archive, files_under(smoke_output, ""), result)
            print(
                "SMOKE COMPLETED; REVIEW REQUIRED; FULL_TRAIN remains disabled. Receipt SHA256="
                + sha(smoke_output / "smoke_receipt.json")
            )
            atomic_json(
                WORK / "sessions" / args.session / "mode_result.json",
                {"status": "SMOKE_ONLY", "production_complete": False},
            )
            return exit_code("SMOKE_ONLY")
        external = (
            read_json(args.external_settings) if args.external_settings else {"enabled": False}
        )
        require(
            not external.get("enabled") or external.get("explicit_upload_authorized") is True,
            "External persistence requires separate explicit upload authorization",
        )
        controller = Controller(
            config, receipt, authorization, args.started_unix, external=external
        )
        if args.session == "B":
            previous = read_json(args.session_a_receipt)
            controller.prior_seconds = {
                b: max(
                    r["training_runtime_seconds"] for r in previous["results"] if r["backbone"] == b
                )
                for b in GPU
            }
        try:
            result = controller.run()
            return exit_code(result["status"])
        except Exception:
            error = traceback.format_exc()
            try:
                controller.request_boundary_stop("Orchestrator exception")
                controller.failure("session", -1, "CONTROL_PLANE", error)
                # Preserve only quiescent states; never archive a checkpoint being replaced.
                for row in list(controller.active.values()):
                    try:
                        row["process"].wait(timeout=config["boundary_stop_grace_seconds"])
                    except subprocess.TimeoutExpired:
                        os.killpg(row["process"].pid, signal.SIGTERM)
                        try:
                            row["process"].wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(row["process"].pid, signal.SIGKILL)
                            row["process"].wait(timeout=30)
                for thread in getattr(controller, "workers", []):
                    thread.join()
                while not controller.events.empty():
                    controller.event(controller.events.get_nowait())
            finally:
                controller.finish()
            raise


if __name__ == "__main__":
    raise SystemExit(main())
