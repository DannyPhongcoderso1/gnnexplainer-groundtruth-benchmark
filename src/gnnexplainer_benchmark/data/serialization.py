from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from gnnexplainer_benchmark.utils.paths import resolve_repository_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_data_artifact(data: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return sha256_file(path)


def load_data_artifact(path: Path) -> Any:
    return torch.load(resolve_repository_path(path), map_location="cpu", weights_only=False)


def save_metadata(metadata: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(resolve_repository_path(path).read_text(encoding="utf-8"))
