# ruff: noqa: E402
from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gnnexplainer_benchmark.data.serialization import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze BA-House 2x2 target intersection.")
    parser.add_argument(
        "--parity",
        type=Path,
        default=Path("results/metrics/model_target_parity_gcn_graphsage.csv"),
    )
    parser.add_argument(
        "--frozen-gcn-targets",
        type=Path,
        default=Path("results/metrics/explainer_test_targets.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/ba_house/ba500/targets/ba_house_2x2_targets.csv"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("configs/ba_house/ba500/targets/ba_house_2x2_targets_manifest.json"),
    )
    parser.add_argument("--min-targets", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = read_csv(args.frozen_gcn_targets)
    parity_by_node = {int(row["node_id"]): row for row in read_csv(args.parity)}
    selected_rows = []
    for row in targets:
        node_id = int(row["node_id"])
        parity = parity_by_node[node_id]
        if is_eligible(parity):
            selected_rows.append(row)
    if len(selected_rows) < args.min_targets:
        raise RuntimeError(
            f"BA-House 2x2 intersection too small: {len(selected_rows)} < {args.min_targets}"
        )
    write_csv(args.output, selected_rows, list(targets[0]))
    manifest = {
        "status": "success",
        "rule": (
            "original frozen GCN target AND GCN correct AND GraphSAGE correct "
            "AND GT explanation eligible AND valid candidate mapping for both backbones"
        ),
        "source_frozen_gcn_targets": str(args.frozen_gcn_targets),
        "source_frozen_gcn_targets_sha256": sha256_file(args.frozen_gcn_targets),
        "source_parity_table": str(args.parity),
        "source_parity_table_sha256": sha256_file(args.parity),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "target_count": len(selected_rows),
        "node_ids": [int(row["node_id"]) for row in selected_rows],
        "frozen_before_graphsage_explanations": True,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "BA-House 2x2 targets frozen | "
        f"count={len(selected_rows)} | sha256={manifest['output_sha256']}"
    )
    print(f"output={args.output}")
    print(f"manifest={args.manifest_output}")


def is_eligible(row: dict[str, str]) -> bool:
    return all(
        parse_bool(row[field])
        for field in [
            "gcn_correct",
            "graphsage_correct",
            "gt_explanation_eligible",
            "candidate_mapping_valid_for_gcn",
            "candidate_mapping_valid_for_graphsage",
            "eligible_for_ba_house_2x2",
        ]
    )


def parse_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
