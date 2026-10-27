"""Historical CLI compatibility for the MIT Elliptic preprocessing pipeline."""

import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root), str(root / "src")]
if __name__ == "__main__":
    runpy.run_module("scripts.data.preprocess_elliptic", run_name="__main__")
