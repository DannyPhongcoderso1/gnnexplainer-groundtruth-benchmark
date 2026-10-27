"""Resolve former publication paths without rewriting execution provenance."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _mapping(root: Path) -> dict[str, str]:
    catalog = root / "docs/provenance/repository_paths.json"
    if not catalog.is_file():
        return {}
    return json.loads(catalog.read_text(encoding="utf-8"))["paths"]


def resolve_repository_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Map an exact former repository path; leave unrelated caller paths intact."""
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    supplied = Path(path)
    if supplied.is_absolute():
        try:
            relative = supplied.relative_to(root).as_posix()
        except ValueError:
            return supplied
    else:
        relative = supplied.as_posix()
    destination = _mapping(root).get(relative)
    if destination is None:
        return supplied if supplied.is_absolute() else root / supplied
    resolved = (root / destination).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Published path escapes repository root")
    return resolved
