"""Historical command/import compatibility; implementation: scripts/data/freeze_ba_house_2x2_targets.py."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_root), str(_root / "src")]

from scripts.data.freeze_ba_house_2x2_targets import *  # noqa: E402,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("scripts.data.freeze_ba_house_2x2_targets", run_name="__main__")
