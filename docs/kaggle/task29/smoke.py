"""Future explicitly authorized tiny continuation/instrumentation fixture, never import-run."""

import argparse
import subprocess
import threading
import time
from pathlib import Path

from common import (
    BENCHMARK,
    GPU,
    HERE,
    PYTHON,
    atomic_json,
    child_environment,
    read_json,
    require,
    state_paths,
    training_config,
)

ATOL = RTOL = 1e-7


def compare_value(a, b, torch, path="", discrete=False):
    discrete = discrete or "param_groups" in path or path.endswith("/step") or "/rng_state_" in path
    if isinstance(a, torch.Tensor):
        require(
            isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype,
            f"Smoke tensor schema mismatch:{path}",
        )
        require(
            bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "Nonfinite smoke state"
        )
        passed = (
            torch.equal(a, b)
            if discrete or not a.is_floating_point()
            else torch.allclose(a, b, atol=ATOL, rtol=RTOL)
        )
        require(bool(passed), f"Smoke numerical mismatch:{path}")
        return float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0
    import numpy as np

    if isinstance(a, np.ndarray):
        require(
            isinstance(b, np.ndarray) and np.array_equal(a, b), f"Smoke NumPy RNG mismatch:{path}"
        )
        return 0
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), f"Smoke mapping mismatch:{path}")
        return max(
            [compare_value(v, b[k], torch, path + "/" + str(k), discrete) for k, v in a.items()]
            or [0]
        )
    if isinstance(a, (list, tuple)):
        require(type(a) is type(b) and len(a) == len(b), f"Smoke sequence mismatch:{path}")
        return max(
            [
                compare_value(v, z, torch, path + "/" + str(i), discrete)
                for i, (v, z) in enumerate(zip(a, b, strict=True))
            ]
            or [0]
        )
    if isinstance(a, float) and not discrete:
        require(abs(a - b) <= ATOL + RTOL * abs(b), f"Smoke scalar mismatch:{path}")
        return abs(a - b)
    require(a == b, f"Smoke exact identity mismatch:{path}")
    return 0


def run(session, authorization, output):
    protocol = read_json(HERE / "protocol.json")
    output = Path(output)
    require(not output.exists(), "Smoke output already exists; review previous attempt")
    output.mkdir(parents=True)
    failures, diagnostics, timing = [], {}, {}

    def worker(backbone):
        try:
            states = {}
            started = time.time()
            for name, stop, uninstrumented in [
                ("reference", 2, True),
                ("instrumented", 2, False),
                ("split", 1, False),
                ("split", 2, False),
            ]:
                directory, config = training_config(
                    protocol, backbone, 0, smoke_scope=session + "_" + name
                )
                directory.mkdir(parents=True, exist_ok=True)
                config_path = directory / "training.json"
                if config_path.exists():
                    require(read_json(config_path) == config, "Smoke config changed on resume")
                else:
                    atomic_json(config_path, config, overwrite=False)
                command = [
                    str(PYTHON),
                    "-u",
                    str(HERE / "training_adapter.py"),
                    "--session",
                    session,
                    "--backbone",
                    backbone,
                    "--seed",
                    "0",
                    "--config",
                    str(config_path),
                    "--authorization",
                    str(authorization),
                    "--operation",
                    "smoke",
                    "--stop-after-epoch",
                    str(stop),
                    "--smoke-barrier",
                    str(output / "barriers" / f"{name}_{stop}"),
                ]
                if uninstrumented:
                    command.append("--uninstrumented")
                log = output / f"{backbone}_{name}_{stop}.log"
                with log.open("x", encoding="utf-8") as stream:
                    result = subprocess.run(
                        command,
                        cwd=BENCHMARK,
                        env=child_environment(backbone),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=900,
                    )
                require(
                    result.returncode == 0, f"Smoke subprocess failed:{backbone}/{name}; see{log}"
                )
                if stop == 2:
                    states[name] = (
                        state_paths(directory, backbone, 0)["run"] / "checkpoints/epoch_002.pt"
                    )
            import torch
            from state_verify import verify_epoch

            numerical = {}
            for name, path in states.items():
                config_path = path.parents[3] / "training.json"
                verify_epoch(path, backbone, 0, config_path, protocol, targets=2)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                numerical[name] = {
                    k: payload[k]
                    for k in (
                        "algorithm_state_dict",
                        "optimizer_state_dict",
                        "rng_state_cpu",
                        "rng_state_numpy",
                        "rng_state_cuda",
                    )
                }
                numerical[name]["losses"] = [r["mean_loss"] for r in payload["epoch_rows"]]
            diagnostics[backbone] = {
                "instrumentation_max_difference": compare_value(
                    numerical["reference"], numerical["instrumented"], torch
                ),
                "continuation_max_difference": compare_value(
                    numerical["instrumented"], numerical["split"], torch
                ),
                "actual_updates_per_branch": 4,
                "optimizer_step_verified": 4,
                "reference_uninstrumented": True,
                "two_completed_fixture_epochs": True,
                "rng_and_discrete_settings_exact": True,
            }
            instrumented_dir, _ = training_config(
                protocol, backbone, 0, smoke_scope=session + "_instrumented"
            )
            actual = read_json(instrumented_dir / "smoke_interval_stop2.json")
            require(actual["calls"] == 4, "Smoke actual update-return count incorrect")
            timing[backbone] = {
                "started": actual["started"],
                "ended": actual["ended"],
                "worker_started": started,
                "worker_ended": time.time(),
            }
        except Exception:
            import traceback

            failures.append({"backbone": backbone, "traceback": traceback.format_exc()})

    workers = [threading.Thread(target=worker, args=(backbone,)) for backbone in GPU]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    approval = read_json(authorization)
    passed = not failures and set(diagnostics) == set(GPU)
    overlap = (
        min(t["ended"] for t in timing.values()) - max(t["started"] for t in timing.values())
        if passed
        else 0
    )
    passed = passed and overlap > 0
    receipt = {
        "status": "SMOKE_PASS_REVIEW_REQUIRED" if passed else "SMOKE_FAIL",
        "session": session,
        "runtime_contract": approval["runtime_contract"],
        "protocol_identity": approval["protocol_identity"],
        "diagnostics": diagnostics,
        "failures": failures,
        "actual_training_interval_overlap_seconds": overlap,
        "training_intervals": timing,
        "reference_scope": "No Task29 progress/checkpoint hooks; shared diagnostic timing/barrier only",
        "atol": ATOL,
        "rtol": RTOL,
        "bitwise_cuda_equivalence_claimed": False,
        "final200_used": False,
        "production_approved": False,
    }
    atomic_json(output / "smoke_receipt.json", receipt, overwrite=False)
    require(passed, "Smoke failed; preserve logs and review, no automatic production")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=["A", "B"], required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.session, args.authorization, args.output)
