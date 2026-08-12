from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class RuntimeTimer:
    runtime_seconds: float | None = None


@contextmanager
def measure_runtime() -> Iterator[RuntimeTimer]:
    timer = RuntimeTimer()
    start = perf_counter()
    try:
        yield timer
    finally:
        timer.runtime_seconds = perf_counter() - start
