"""Explicit subprocess entrypoint: frozen runner plus observational runtime hooks."""

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

from common import BASELINE, BENCHMARK, GPU, HERE, WORK, atomic_json, read_json, require, utc
from preflight import file_gate, runtime_gate
from progress import Observer
from resources import nvidia
from state_verify import exact_serialized_state, verify_epoch


class BoundaryStop(RuntimeError):
    pass


def emit(value):
    print(
        "TASK29_EVENT "
        + json.dumps({"utc": utc(), "pid": os.getpid(), **value}, sort_keys=True, allow_nan=False),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=["A", "B"], required=True)
    parser.add_argument("--backbone", choices=list(GPU), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--operation", choices=["train", "smoke"], required=True)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--uninstrumented", action="store_true")
    parser.add_argument("--smoke-barrier", type=Path)
    parser.add_argument("--allow-same-seed-restart", action="store_true")
    args = parser.parse_args()
    approval = read_json(args.authorization)
    require(
        approval["mode"] == ("FULL_TRAIN" if args.operation == "train" else "SMOKE_TEST"),
        "Worker lacks operation-specific authorization",
    )
    require(
        args.seed in approval["allowed_seeds"] and args.backbone in GPU, "Unauthorized worker seed"
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(GPU[args.backbone]), "Wrong physical GPU pin"
    )
    protocol = read_json(HERE / "protocol.json")
    file_gate(protocol)
    runtime = runtime_gate(protocol, worker=True)
    require(
        runtime["numerical"] == approval["runtime_contract"]["numerical"]
        and runtime["versions"] == approval["runtime_contract"]["versions"],
        "Worker numerical/runtime fingerprint mismatch",
    )
    import torch
    from torch_geometric.explain.algorithm import PGExplainer

    hardware = nvidia(["index", "uuid", "name", "driver_version"])
    physical = next(row for row in hardware if int(row["index"]) == GPU[args.backbone])
    logical_uuid = str(torch.cuda.get_device_properties(0).uuid)
    require(
        logical_uuid.removeprefix("GPU-").lower() == physical["uuid"].removeprefix("GPU-").lower(),
        "Logical cuda:0 does not map to required physical GPU UUID",
    )
    require(
        physical["driver_version"] in approval["runtime_contract"]["driver_versions"],
        "Worker driver differs from reviewed runtime",
    )
    os.chdir(BENCHMARK)
    sys.path[:0] = [str(BENCHMARK), str(BENCHMARK / "src"), str(BENCHMARK / "scripts")]
    import scripts.run_elliptic_pgexplainer_training as runner

    directory = args.config.parent
    runtime.update(
        hard_compatibility=runner.hard_compatibility(torch.device("cuda")),
        physical_gpu=GPU[args.backbone],
        physical_uuid=physical["uuid"],
        logical_uuid=logical_uuid,
        logical_device="cuda:0",
        baseline=str(BASELINE),
        runtime_contract=approval["runtime_contract"],
    )
    runtime_path = directory / "runtime.json"
    if runtime_path.exists():
        old = read_json(runtime_path)
        require(
            old["runtime_contract"] == runtime["runtime_contract"]
            and old["hard_compatibility"] == runtime["hard_compatibility"],
            "Restored worker fingerprint mismatch",
        )
    atomic_json(runtime_path, runtime)
    targets = 2 if args.operation == "smoke" else 2000
    from recovery import begin_scientific_execution, resume_gate

    def check_resume_identity(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = read_json(args.config)["task28_training"]
        pg = runner.load_yaml(Path(cfg["pgexplainer_config"]))
        pg["explainer"].update(seed=args.seed, epochs=30)
        expected = runner.config_identity(
            cfg,
            pg,
            runner.SPECS[args.backbone],
            args.seed,
            payload["hashes"],
            {
                "allow_debug_limits": args.operation == "smoke",
                "train_count": targets,
                "validation_count": 1 if args.operation == "smoke" else 500,
            },
            compatibility=runtime["hard_compatibility"],
        )
        require(payload["config_identity"] == expected, "Resume config identity mismatch")

    admission = resume_gate(
        directory,
        args.backbone,
        args.seed,
        args.config,
        protocol,
        runtime,
        check_resume_identity,
        targets=targets,
        allow_restart=args.allow_same_seed_restart,
    )
    observer = Observer(
        args.session,
        args.backbone,
        args.seed,
        emit,
        targets=targets,
        interval=approval["target_heartbeat_interval"],
        seconds=approval["heartbeat_seconds"],
    )
    if not args.uninstrumented:
        PGExplainer.train = observer.wrap(PGExplainer.train)
        original_save = runner.atomic_torch_save
        save_counts = {}

        def observed_save(path, payload):
            original_save(path, payload)
            if path.name.startswith("epoch_"):
                save_counts[str(path)] = save_counts.get(str(path), 0) + 1
                if save_counts[str(path)] == 2:
                    exact_serialized_state(path, payload)
                    cfg = read_json(args.config)["task28_training"]
                    pg = runner.load_yaml(Path(cfg["pgexplainer_config"]))
                    pg["explainer"].update(seed=args.seed, epochs=30)
                    scope = {
                        "allow_debug_limits": args.operation == "smoke",
                        "train_count": targets,
                        "validation_count": 1 if args.operation == "smoke" else 500,
                    }
                    require(
                        payload["config_identity"]
                        == runner.config_identity(
                            cfg,
                            pg,
                            runner.SPECS[args.backbone],
                            args.seed,
                            payload["hashes"],
                            scope,
                            compatibility=payload["hard_compatibility"],
                        ),
                        "Epoch config_identity recomputation mismatch",
                    )
                    receipt = verify_epoch(
                        path,
                        args.backbone,
                        args.seed,
                        args.config,
                        protocol,
                        runtime,
                        targets=targets,
                    )
                    receipt.update(
                        python_random_state=random.getstate(),
                        runtime_contract=approval["runtime_contract"],
                    )
                    atomic_json(
                        directory / f"verified_epoch_{receipt['epoch_completed']:03d}.json", receipt
                    )
                    atomic_json(directory / "durable_progress.json", receipt)
                    row = payload["epoch_rows"][-1]
                    emit(
                        {
                            "event": "EPOCH_VERIFIED",
                            "session": args.session,
                            "backbone": args.backbone,
                            "physical_gpu": GPU[args.backbone],
                            "seed": args.seed,
                            "epoch": receipt["epoch_completed"],
                            "target_completed": targets,
                            "durable_updates": receipt["durable_updates"],
                            "checkpoint_path": str(path),
                            "checkpoint_sha256": receipt["sha256"],
                            "mean_loss": row["mean_loss"],
                            "component_losses": None,
                            "component_loss_reason": "Not exposed by historical runner; no extra computations",
                            "epoch_duration_seconds": row["runtime_seconds"],
                            "cumulative_training_seconds": payload["runtime_ledger"][
                                "training_seconds"
                            ],
                            "learning_rate": 0.003,
                            "temperature": 5 * (2 / 5) ** ((receipt["epoch_completed"] - 1) / 30),
                            "finite": True,
                            "status": "EPOCH_DURABLE_VERIFIED",
                        }
                    )
                    if args.operation == "train" and (WORK / "STOP_AT_EPOCH_BOUNDARY").exists():
                        raise BoundaryStop("Session time/system guard: verified epoch preserved")

        runner.atomic_torch_save = observed_save
    if args.operation == "smoke":
        require(args.smoke_barrier is not None, "Smoke concurrency barrier is required")
        method = PGExplainer.train
        interval = {"calls": 0, "started": None, "ended": None}

        def timed_fixture(instance, epoch, *call_args, **call_kwargs):
            if interval["calls"] == 0:
                args.smoke_barrier.mkdir(parents=True, exist_ok=True)
                (args.smoke_barrier / f"{args.backbone}.ready").touch(exist_ok=False)
                other = "gcn" if args.backbone == "graphsage" else "graphsage"
                deadline = time.monotonic() + 180
                while not (args.smoke_barrier / f"{other}.ready").exists():
                    require(time.monotonic() < deadline, "Smoke concurrency barrier timeout")
                    time.sleep(0.05)
                interval["started"] = time.time()
            result = method(instance, epoch, *call_args, **call_kwargs)
            interval["calls"] += 1
            interval["ended"] = time.time()
            atomic_json(directory / f"smoke_interval_stop{args.stop_after_epoch}.json", interval)
            return result

        PGExplainer.train = timed_fixture
    argv = [
        "runner",
        "--config",
        str(args.config),
        "--backbone",
        args.backbone,
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
    ]
    if args.operation == "smoke":
        argv.extend(
            [
                "--allow-debug-limits",
                "--debug-train-limit",
                "2",
                "--debug-val-limit",
                "1",
                "--stop-after-epoch",
                str(args.stop_after_epoch or 2),
            ]
        )
    elif args.stop_after_epoch:
        raise RuntimeError("Production stop epoch must be controlled only by time guard")
    sys.argv = argv
    begin_scientific_execution(directory, admission)
    emit(
        {
            "event": "WORKER_STARTED",
            "session": args.session,
            "backbone": args.backbone,
            "seed": args.seed,
            "physical_gpu": GPU[args.backbone],
            "physical_uuid": physical["uuid"],
            "last_committed_epoch": admission["epoch_completed"],
            "durable_updates": admission["durable_updates"],
            "verified_checkpoint_path": admission["verified_checkpoint_path"],
        }
    )
    try:
        runner.main()
    except BoundaryStop:
        emit(
            {
                "event": "RECOVERABLE_BOUNDARY_STOP",
                "session": args.session,
                "backbone": args.backbone,
                "seed": args.seed,
            }
        )
        return 86
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
