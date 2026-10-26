"""Fail-closed runtime/source/input gates. No model construction or forward."""

import importlib
import importlib.metadata
import inspect
import os
import platform
import shutil
import sys
from pathlib import Path

from common import (
    BASELINE,
    BENCHMARK,
    GIB,
    HERE,
    WORK,
    atomic_json,
    identity,
    read_json,
    require,
    sha,
)
from resources import hardware_gate, numerical_settings, snapshot


def file_gate(protocol, benchmark=BENCHMARK, baseline=BASELINE):
    for scope, root in [("benchmark", benchmark), ("baseline", baseline)]:
        for name, expected in protocol["frozen_files"][scope].items():
            path = Path(root) / name
            require(
                path.is_file() and not path.is_symlink() and sha(path) == expected,
                f"Frozen {scope} hash mismatch: {name}",
            )
    for name, expected in read_json(HERE / "execution_manifest.json")["files"].items():
        require(sha(HERE / name) == expected, f"Task29 execution source changed: {name}")


def runtime_gate(protocol, worker=False):
    require(
        shutil.which("git") is not None,
        "Frozen runner still requires the git executable for its fallback metadata probes",
    )
    require(
        not os.environ.get("GIT_DIR") and not os.environ.get("GIT_WORK_TREE"),
        "Unexpected Git metadata override in execution environment",
    )
    require(
        platform.system() == "Linux" and platform.python_version() == "3.11.9",
        "Training runtime requires Linux/Python3.11.9",
    )
    require(
        Path(sys.prefix).resolve() == Path("/kaggle/working/task29-venv").resolve(),
        "Refusing notebook/global interpreter",
    )
    versions = {}
    for name, expected in protocol["dependency_pins"].items():
        actual = importlib.metadata.version(name)
        require(
            actual == expected,
            f"Pinned dependency mismatch: {name} expected{expected}, got{actual}",
        )
        versions[name] = actual
    for extra in ("torch-scatter", "torch-sparse", "pyg-lib"):
        try:
            importlib.metadata.version(extra)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"Unreviewed optional acceleration backend: {extra}")
    sys.path[:0] = [str(BENCHMARK), str(BENCHMARK / "src"), str(BENCHMARK / "scripts")]
    os.environ["ELLIPTIC_BASELINE_REPO"] = str(BASELINE)
    for module in protocol["required_imports"]:
        importlib.import_module(module)
    import torch
    from torch_geometric.explain.algorithm import PGExplainer
    import hashlib

    content = (
        Path(inspect.getfile(PGExplainer))
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
    )
    require(
        hashlib.sha256(content).hexdigest() == protocol["pyg_pgexplainer_sha256"],
        "Installed historical PyG PGExplainer source mismatch",
    )
    require(
        torch.version.cuda == "12.8" and torch.cuda.is_available(),
        "CUDA12.8/Torch availability gate failed",
    )
    require(
        torch.cuda.device_count() == (1 if worker else 2), "Incorrect visible CUDA device count"
    )
    return {
        "versions": versions,
        "numerical": numerical_settings(torch),
        "cuda_build": torch.version.cuda,
        "python": platform.python_version(),
        "os": platform.platform(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def contract(protocol, runtime, hardware):
    return {
        "protocol_identity": identity(protocol),
        "versions": runtime["versions"],
        "python": runtime["python"],
        "cuda_build": runtime["cuda_build"],
        "numerical": runtime["numerical"],
        "gpu_models_by_role": {
            b: hardware["gpus"][i]["name"] for b, i in [("graphsage", 0), ("gcn", 1)]
        },
        "driver_versions": sorted({x["driver_version"] for x in hardware["gpus"]}),
        "source_identity": protocol["execution_identity"],
        "scientific_origin_commit": protocol["scientific_origin_commit"],
        "runner_git_fallback_without_dotgit": "",
    }


def run(session):
    protocol = read_json(HERE / "protocol.json")
    file_gate(protocol)
    runtime = runtime_gate(protocol)
    WORK.mkdir(parents=True, exist_ok=True)
    hardware = snapshot(WORK)
    hardware_gate(hardware, required_free_bytes=12 * GIB)
    result = {
        "status": "PREFLIGHT_PASS",
        "session": session,
        "runtime": runtime,
        "hardware": hardware,
        "runtime_contract": contract(protocol, runtime, hardware),
        "protocol_identity": identity(protocol),
        "model_forwards": 0,
        "training_runs": 0,
        "required_output_disk_bytes": 3 * GIB,
        "resources_target_not_guaranteed": True,
    }
    atomic_json(WORK / "sessions" / session / "preflight.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=["A", "B"], required=True)
    args = parser.parse_args()
    print(run(args.session)["status"])
