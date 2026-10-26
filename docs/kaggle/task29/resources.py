"""Read hardware/process metrics without GPU synchronization or numerical changes."""

import csv
import io
import os
import shutil
import subprocess
from pathlib import Path

from common import GIB, require


def nvidia(fields, compute=False):
    command = [
        "nvidia-smi",
        "--query-compute-apps=" + ",".join(fields)
        if compute
        else "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, timeout=15)
    return [
        dict(zip(fields, [x.strip() for x in row], strict=True))
        for row in csv.reader(io.StringIO(output))
        if row
    ]


def memory():
    import psutil

    value = psutil.virtual_memory()
    limits = [value.total]
    used = value.total - value.available
    for maximum, current in [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ]:
        if Path(maximum).is_file():
            content = Path(maximum).read_text().strip()
            if content != "max" and int(content) < value.total:
                limits.append(int(content))
                used = int(Path(current).read_text().strip())
    return {
        "ram_limit_bytes": min(limits),
        "ram_used_bytes": used,
        "ram_available_bytes": max(0, min(limits) - used),
    }


def snapshot(work, pids=None):
    disk = shutil.disk_usage(work)
    return {
        "gpus": sorted(
            nvidia(
                [
                    "index",
                    "uuid",
                    "name",
                    "driver_version",
                    "memory.total",
                    "memory.used",
                    "utilization.gpu",
                    "temperature.gpu",
                ]
            ),
            key=lambda r: int(r["index"]),
        ),
        "gpu_processes": nvidia(["gpu_uuid", "pid", "used_gpu_memory"], compute=True),
        "memory": memory(),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "worker_pids": pids or {},
    }


def hardware_gate(data, required_free_bytes):
    require(
        len(data["gpus"]) == 2 and {x["index"] for x in data["gpus"]} == {"0", "1"},
        "Exactly two physical GPUs required",
    )
    require(len({x["uuid"] for x in data["gpus"]}) == 2, "Duplicate GPU UUID")
    for gpu in data["gpus"]:
        require(
            "T4" in gpu["name"] and float(gpu["memory.total"]) * 1024**2 >= 14 * GIB,
            "Tesla T4 with at least14GiB required on each GPU",
        )
        require(
            gpu["driver_version"] and gpu["uuid"].startswith("GPU-"), "GPU/driver identity missing"
        )
    require(data["memory"]["ram_limit_bytes"] >= 28 * GIB, "Effective RAM limit below28GiB")
    require(data["memory"]["ram_available_bytes"] >= 10 * GIB, "Insufficient available RAM")
    require(data["disk_free_bytes"] >= required_free_bytes, "Insufficient working disk budget")


def numerical_settings(torch):
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "torch_build_config": torch.__config__.show(),
    }
