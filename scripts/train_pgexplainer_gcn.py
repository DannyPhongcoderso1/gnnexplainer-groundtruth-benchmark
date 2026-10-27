"""Historical command/import compatibility; implementation: scripts/train/train_pgexplainer_gcn.py."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_root), str(_root / "src")]

from scripts.train.train_pgexplainer_gcn import *  # noqa: E402,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("scripts.train.train_pgexplainer_gcn", run_name="__main__")
