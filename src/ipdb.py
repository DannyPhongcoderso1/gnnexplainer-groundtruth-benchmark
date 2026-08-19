"""Compatibility shim for the vendored GraphXAI debug-only import."""


def set_trace() -> None:
    return None
