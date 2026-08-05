from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit if commit and "HEAD" not in commit else None


def collect_environment(compute_target: str) -> dict[str, Any]:
    import matplotlib
    import networkx
    import numpy
    import pandas
    import scipy
    import sklearn
    import torch
    import torch_geometric

    cuda_available = torch.cuda.is_available()
    return {
        "compute_target": compute_target,
        "cuda_available": cuda_available,
        "git_commit": _git_commit(),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "matplotlib_version": matplotlib.__version__,
        "networkx_version": networkx.__version__,
        "numpy_version": numpy.__version__,
        "os": platform.system(),
        "pandas_version": pandas.__version__,
        "pip_version": _version("pip"),
        "platform": platform.platform(),
        "python_executable": str(Path(sys.executable).name),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_geometric_version": torch_geometric.__version__,
        "torch_version": torch.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a sanitized environment report.")
    parser.add_argument("--output", default="docs/environment_report.json")
    parser.add_argument("--compute-target", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_environment(compute_target=args.compute_target)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
