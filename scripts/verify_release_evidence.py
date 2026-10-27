"""Historical command/import compatibility; implementation: scripts/verify/verify_release_evidence.py."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_root), str(_root / "src")]

from scripts.verify.verify_release_evidence import *  # noqa: E402,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("scripts.verify.verify_release_evidence", run_name="__main__")
