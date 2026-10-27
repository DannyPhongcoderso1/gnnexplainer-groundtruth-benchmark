"""Historical command/import compatibility; implementation: scripts/train/run_pgexplainer_graphsage_stability.py."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_root), str(_root / "src")]

from scripts.train.run_pgexplainer_graphsage_stability import *  # noqa: E402,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("scripts.train.run_pgexplainer_graphsage_stability", run_name="__main__")
