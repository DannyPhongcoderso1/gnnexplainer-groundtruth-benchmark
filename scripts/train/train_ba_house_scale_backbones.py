"""Historical BA scale protocol; delegates to its provenance-pinned producer."""

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.extensions.train_ba_house_scale_backbones import main

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
