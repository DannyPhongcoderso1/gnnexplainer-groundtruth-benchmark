# ruff: noqa: E402
from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
from pathlib import Path

from gnnexplainer_benchmark.explainers.pgexplainer_training import train_pgexplainer_gcn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PGExplainer on the frozen GCN/BA-House benchmark."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house/ba500/pgexplainer/gcn/primary_seed42.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/explainers/pgexplainer_gcn_ba_house_seed42_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_pgexplainer_gcn(
        device=args.device,
        artifact_path=args.artifact,
        manifest_path=args.manifest,
    )
    print(
        "PGExplainer GCN training complete | "
        f"epochs={result.epochs_completed} | "
        f"final_loss={result.final_loss:.6f} | "
        f"training_runtime_seconds={result.training_runtime_seconds:.3f} | "
        f"model_parameters_unchanged={result.model_parameters_unchanged} | "
        f"checkpoint_reload_ok={result.checkpoint_reload_ok} | "
        f"validation_success={result.validation_success_count} | "
        f"validation_failure={result.validation_failure_count}"
    )
    print(f"artifact={result.artifact_path}")
    print(f"manifest={result.manifest_path}")
    print(f"training_log={result.training_log_path}")
    print(f"validation_health={result.validation_health_path}")
    print(f"validation_summary={result.validation_summary_path}")


if __name__ == "__main__":
    main()
