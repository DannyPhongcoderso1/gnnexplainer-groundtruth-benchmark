from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch_geometric

from gnnexplainer_benchmark.extensions.ba_house_scale import (
    DETERMINISM_TENSORS,
    SCIENTIFIC_TENSORS,
    frozen_generation_config,
    generate_ba_house_scale,
    structural_audit,
    structural_receipt,
    structure,
)

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/extension/ba_house_scale/phase1_generation"
ART = ROOT / "artifacts/extensions/ba_house_scale"
CONFIG = ROOT / "configs/extensions/ba_house_scale"
PHASE1 = ROOT / "docs/extension/phase1_pipeline_audit"
VARIANTS = [
    ("BA500_EXT", "ba500_ext", 300, 40),
    ("BA1000_EXT", "ba1000_ext", 600, 80),
    ("BA1500_EXT", "ba1500_ext", 900, 120),
]
BASELINE_SHA = "4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94"


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def exclusive_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def exclusive_tensor(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
        "creation_timestamp_utc": datetime.fromtimestamp(
            path.stat().st_ctime, UTC
        ).isoformat(),
    }


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def output_paths() -> list[Path]:
    paths = [
        DOC / name
        for name in (
            "01_generation_report.md",
            "01_generation_report.json",
            "02_scale_parity_report.md",
            "02_scale_parity_report.json",
            "03_structural_clean_pool.csv",
            "03_structural_clean_pool.json",
            "04_verifier_receipt.json",
            "protected_baseline_before.json",
            "protected_baseline_after.json",
            "generation_log.json",
            "command_receipt.json",
        )
    ]
    paths.append(ART / "ba_house_scale_dataset_manifest.json")
    for _, slug, _, _ in VARIANTS:
        paths.extend(
            [
                CONFIG / f"{slug}.json",
                ART / slug / f"{slug}.pt",
                ART / slug / f"{slug}_manifest.json",
                ART / slug / "structural_gt_diagnostics.pt",
            ]
        )
    return paths


def guard_outputs() -> None:
    existing = [file_record(path) for path in output_paths() if path.exists()]
    if existing:
        print(json.dumps(existing, indent=2))
        raise RuntimeError("BA_HOUSE_SCALE_PHASE1_OUTPUT_ALREADY_EXISTS")


def before_manifest() -> dict[str, Any]:
    reference_path = PHASE1 / "protected_baseline_manifest_after.json"
    reference = read_json(reference_path)
    rows = []
    for expected in reference["files"]:
        path = ROOT / expected["path"]
        actual = {"path": expected["path"], "bytes": path.stat().st_size, "sha256": sha(path)}
        if actual != expected:
            raise RuntimeError("BA_HOUSE_SCALE_BASELINE_PRESERVATION_FAILURE: " + str(path))
        rows.append(actual)
    # Phase 1 reports are now frozen too; do not use its old builder/verifier.
    for path in sorted(PHASE1.iterdir()):
        if path.is_file():
            rows.append(
                {
                    key: value
                    for key, value in file_record(path).items()
                    if key != "creation_timestamp_utc"
                }
            )
    rows.sort(key=lambda row: row["path"])
    result = {
        "created_at_utc": timestamp(),
        "reference": file_record(reference_path),
        "scope": "Exact Phase1 protected file list plus all Phase1 audit files; "
        "new extensions excluded",
        "file_count": len(rows),
        "phase1_reference_match": True,
        "files": rows,
    }
    exclusive_json(DOC / "protected_baseline_before.json", result)
    return result


def after_manifest(before: dict[str, Any]) -> None:
    rows = [
        {
            "path": row["path"],
            "bytes": (ROOT / row["path"]).stat().st_size,
            "sha256": sha(ROOT / row["path"]),
        }
        for row in before["files"]
    ]
    unchanged = rows == before["files"]
    exclusive_json(
        DOC / "protected_baseline_after.json",
        {
            "created_at_utc": timestamp(),
            "file_count": len(rows),
            "files": rows,
            "exact_equality": unchanged,
            "verdict": "BA_HOUSE_SCALE_BASELINE_PRESERVATION_PASS"
            if unchanged
            else "BA_HOUSE_SCALE_BASELINE_PRESERVATION_FAIL",
        },
    )
    if not unchanged:
        raise RuntimeError("BA_HOUSE_SCALE_BASELINE_PRESERVATION_FAILURE")


def check_semantics(stats: dict[str, Any], background: int, houses: int) -> None:
    total = background + 5 * houses
    wanted = {"0": background, "1": 2 * houses, "2": 2 * houses, "3": houses}
    valid = (
        stats["num_nodes"] == total
        and stats["background_nodes"] == background
        and stats["motif_instances"] == houses
        and stats["motif_nodes"] == 5 * houses
        and stats["motif_fraction"] == 0.4
        and stats["background_fraction"] == 0.6
        and stats["features_all_ones"]
        and stats["feature_dim"] == 10
        and stats["class_counts"] == {key: value for key, value in wanted.items()}
        and stats["internal_house_undirected_edges"] == 6 * houses
        and stats["global_gt_positive_directed_entries"] == 12 * houses
        and stats["global_node_gt_matches_motifs"]
        and stats["global_edge_gt_matches_internal"]
        and stats["bridge_undirected_edges"] == houses
        and stats["bridge_gt_positive_count"] == 0
        and stats["background_gt_positive_count"] == 0
        and stats["other_directed_entries"] == 0
        and stats["self_loops"] == stats["duplicate_ordered_entries"] == 0
        and stats["reciprocal_complete"]
        and stats["split_disjoint_and_complete"]
        and stats["split_counts"]
        == {"train": int(total * 0.6), "validation": int(total * 0.2), "test": int(total * 0.2)}
    )
    if not valid:
        raise RuntimeError("BA_HOUSE_SCALE_DATASET_PHASE_INVALID: structural semantics")


def table(title: str, columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"title": title, "columns": columns, "rows": rows}


def md_table(value: dict[str, Any]) -> str:
    def cell(item: Any) -> str:
        return (json.dumps(item) if isinstance(item, (dict, list)) else str(item)).replace(
            "|", " / "
        )

    lines = [
        "## " + value["title"],
        "",
        "| " + " | ".join(value["columns"]) + " |",
        "| " + " | ".join(["---"] * len(value["columns"])) + " |",
    ]
    lines += ["| " + " | ".join(cell(item) for item in row) + " |" for row in value["rows"]]
    return "\n".join(lines)


def main() -> None:
    guard_outputs()
    before = before_manifest()
    if sha(ROOT / "artifacts/data/ba_house.pt") != BASELINE_SHA:
        raise RuntimeError("BA_HOUSE_SCALE_BASELINE_PRESERVATION_FAILURE")
    torch.set_num_threads(1)
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "pyg": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "dataset_device": "cpu",
        "git_commit": git("rev-parse", "HEAD"),
        "git_diff_status_before_generation": git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ),
    }
    scientific = read_json(PHASE1 / "01_ba_house_pipeline_audit.json")
    frozen_versions = read_json(ROOT / "artifacts/data/ba_house_metadata.json")["versions"]
    reference = torch.load(
        ROOT / "artifacts/data/ba_house.pt", map_location="cpu", weights_only=False
    )
    started = timestamp()
    command = {
        "started_at_utc": started,
        "working_directory": str(ROOT),
        "argv": sys.argv,
        "python_executable": sys.executable,
        "allowed_activity": "Dataset generation and structural checks only; "
        "no models, targets, explainers or XAI metrics",
    }
    exclusive_json(DOC / "command_receipt.json", command)
    manifests, all_records, events = [], [], []
    capture_out, capture_err = io.StringIO(), io.StringIO()
    control = None
    for variant, slug, background, houses in VARIANTS:
        parameters = dict(
            background_nodes=background, num_house_motifs=houses, ba_m=5, feature_dim=10, seed=42
        )
        config = {
            "dataset_variant": variant,
            "parameters": parameters,
            "frozen_generation_config": frozen_generation_config(**parameters),
            "track": "TRACK A ONLY - CLEAN SCALE ROBUSTNESS",
            "primary_gt": "GLOBAL edge GT restricted to target 3-hop candidates",
            "same_seed_nesting_claim": False,
            "no_common_correct_or_final_targets_in_phase_BA1": True,
        }
        config_path = CONFIG / f"{slug}.json"
        exclusive_json(config_path, config)
        with contextlib.redirect_stdout(capture_out), contextlib.redirect_stderr(capture_err):
            data = generate_ba_house_scale(**parameters)
            independent_copy = generate_ba_house_scale(**parameters)
        deterministic = {
            key: torch.equal(getattr(data, key), getattr(independent_copy, key))
            for key in DETERMINISM_TENSORS
        }
        if not all(deterministic.values()):
            raise RuntimeError("BA_HOUSE_SCALE_DATASET_PHASE_INVALID: determinism " + variant)
        stats = structure(data)
        check_semantics(stats, background, houses)
        records, diagnostic_masks = structural_audit(data, variant)
        receipt = structural_receipt(records)
        if (
            not receipt["finite_complete"]
            or not receipt["decomposition_valid"]
            or receipt["structurally_clean"] == 0
        ):
            raise RuntimeError("BA_HOUSE_SCALE_DATASET_PHASE_INVALID: structural pool " + variant)
        dataset_path = ART / slug / f"{slug}.pt"
        diagnostic_path = ART / slug / "structural_gt_diagnostics.pt"
        exclusive_tensor(dataset_path, data)
        exclusive_tensor(diagnostic_path, diagnostic_masks)
        manifest = {
            "created_at_utc": timestamp(),
            "dataset_variant": variant,
            "dataset": file_record(dataset_path),
            "config_file": file_record(config_path),
            "diagnostic_masks": file_record(diagnostic_path),
            "generation_config": config,
            "seed": 42,
            "environment": environment,
            "structure": stats,
            "structural_pool": receipt,
            "determinism_tensor_parity": deterministic,
            "determinism_verdict": variant + "_DETERMINISM_PASS",
            "metadata_semantics": {
                "motif_instance_id": "background0; successive five-node blocks instance1..H",
                "motif_role_in_house": "background=-1; local position0..4; "
                "roles0,1 -> class1; roles2,3 -> class2; role4 -> class3",
                "scientific_gt_unchanged": True,
                "diagnostic_masks_are_not_an_alternate_benchmark": True,
            },
            "same_seed_does_not_imply_graph_nesting": True,
            "no_model_related_activity": True,
        }
        if variant == "BA500_EXT":
            exact = {
                key: torch.equal(getattr(data, key), getattr(reference, key))
                for key in SCIENTIFIC_TENSORS
            }
            verdict = (
                "BA500_EXT_EXACT_DATA_REPRODUCTION"
                if all(exact.values())
                else "BA500_EXT_SEMANTIC_DATA_REPRODUCTION"
            )
            control = {
                "frozen_dataset": file_record(ROOT / "artifacts/data/ba_house.pt"),
                "extension_dataset": file_record(dataset_path),
                "tensor_parity": exact,
                "verdict": verdict,
                "saved_generation_versions": frozen_versions,
                "current_environment": environment,
                "exact_mismatch_reason": None
                if all(exact.values())
                else "Unresolved exact identity mismatch: inspect RNG sequence and environment; "
                "structural invariants passed, no forced identity claim.",
                "serialized_sha_equality_required": False,
                "serialization_note": "Exact reproduction means scientific tensor equality; "
                "extension-only motif metadata changes serialized bytes.",
                "old_target_selection_reproduced": False,
                "phase1_reference_verdict": scientific["verdict"],
            }
            manifest["frozen500_control"] = control
        manifest_path = ART / slug / f"{slug}_manifest.json"
        exclusive_json(manifest_path, manifest)
        manifests.append({**manifest, "manifest_file": file_record(manifest_path)})
        all_records.extend(records)
        event = {
            "variant": variant,
            "completed_at_utc": timestamp(),
            "dataset": file_record(dataset_path),
            "determinism": manifest["determinism_verdict"],
            "structurally_clean": receipt["structurally_clean"],
        }
        events.append(event)
        print(json.dumps(event), flush=True)
    columns = list(all_records[0])
    with (DOC / "03_structural_clean_pool.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_records)
    structural = {
        "protocol": "TRACK A ONLY; structural pool, no common-correct or final target selection",
        "motif_count_definition": "Distinct instances contributing positive GLOBAL GT entries "
        "inside induced3hop candidates",
        "primary_gt": "global_edge_mask restricted to candidate; "
        "own/non-own masks are diagnostics only",
        "variants": {m["dataset_variant"]: m["structural_pool"] for m in manifests},
        "records": all_records,
    }
    exclusive_json(DOC / "03_structural_clean_pool.json", structural)
    tables = []
    tables.append(
        table(
            "BA-S1 Generation Semantics",
            [
                "Variant",
                "Nodes",
                "Background",
                "HouseInstances",
                "MotifNodes",
                "MotifFraction",
                "FeatureDim",
                "BA_m",
                "Seed",
            ],
            [
                [
                    m["dataset_variant"],
                    m["structure"]["num_nodes"],
                    m["structure"]["background_nodes"],
                    m["structure"]["motif_instances"],
                    m["structure"]["motif_nodes"],
                    0.4,
                    10,
                    5,
                    42,
                ]
                for m in manifests
            ],
        )
    )
    tables.append(
        table(
            "BA-S2 Graph Realization",
            [
                "Variant",
                "UndirectedEdges",
                "DirectedEntries",
                "AverageDegree",
                "MedianDegree",
                "MinDegree",
                "MaxDegree",
                "SelfLoops",
                "DuplicateOrderedEntries",
                "ReciprocalComplete",
            ],
            [
                [m["dataset_variant"]]
                + [
                    m["structure"][key]
                    for key in (
                        "undirected_edges",
                        "directed_entries",
                        "average_degree",
                        "median_degree",
                        "min_degree",
                        "max_degree",
                        "self_loops",
                        "duplicate_ordered_entries",
                        "reciprocal_complete",
                    )
                ]
                for m in manifests
            ],
        )
    )
    tables.append(
        table(
            "BA-S3 Labels And Splits",
            ["Variant", "Class0", "Class1", "Class2", "Class3", "Train", "Validation", "Test"],
            [
                [m["dataset_variant"]]
                + [m["structure"]["class_counts"][str(c)] for c in range(4)]
                + list(m["structure"]["split_counts"].values())
                for m in manifests
            ],
        )
    )
    tables.append(
        table(
            "BA-S4 Ground Truth",
            [
                "Variant",
                "InternalHouseUndirectedEdges",
                "PositiveGTDirectedEntries",
                "BridgeUndirectedEdges",
                "BridgeGTPositiveCount",
            ],
            [
                [m["dataset_variant"]]
                + [
                    m["structure"][key]
                    for key in (
                        "internal_house_undirected_edges",
                        "global_gt_positive_directed_entries",
                        "bridge_undirected_edges",
                        "bridge_gt_positive_count",
                    )
                ]
                for m in manifests
            ],
        )
    )
    tables.append(
        table(
            "BA-S5 Structural Clean Pool",
            [
                "Variant",
                "MotifNodes",
                "StructurallyClean",
                "CleanFraction",
                "MeanCandidateEdgesClean",
                "MedianCandidateEdgesClean",
                "MeanGTPrevalenceClean",
            ],
            [
                [
                    m["dataset_variant"],
                    m["structure"]["motif_nodes"],
                    m["structural_pool"]["structurally_clean"],
                    m["structural_pool"]["clean_fraction"],
                    m["structural_pool"]["clean_distributions"]["candidate_directed_edge_count"][
                        "mean"
                    ],
                    m["structural_pool"]["clean_distributions"]["candidate_directed_edge_count"][
                        "median"
                    ],
                    m["structural_pool"]["clean_distributions"]["global_gt_prevalence"]["mean"],
                ]
                for m in manifests
            ],
        )
    )
    distributions = table(
        "Candidate Complexity And Full Structural Distributions",
        [
            "Variant",
            "Population",
            "Metric",
            "Count",
            "Mean",
            "Std",
            "Min",
            "Q25",
            "Median",
            "Q75",
            "Max",
        ],
        [
            [m["dataset_variant"], population, metric] + list(values.values())
            for m in manifests
            for population in ("clean_distributions", "all_motif_distributions")
            for metric, values in m["structural_pool"][population].items()
        ],
    )
    common_notes = [
        "Same seed42 is intentional; graph sizes are separate realizations. "
        "No node-level nesting or matched node IDs across scales is asserted.",
        "Primary GT remains the GLOBAL edge mask restricted to the target's induced "
        "3-hop candidate edges. Own/non-own masks are diagnostic only.",
        "Structural eligibility uses instances contributing positive GT, and requires "
        "own instance only, globalGT=ownGT=12 and nonOwnGT=0.",
        "These pools do not certify common-correctness, predictive performance or the "
        "availability of a fixed final target count. Phase BA-3 must check those later.",
        "Candidate counts and GT prevalence are descriptive covariates within Track A only. "
        "No second track, alternate GT or multi-explanation protocol is created.",
        "No backbone load/train/inference, PG pool, explainer, Random baseline, AUROC/AP, "
        "fidelity or stability experiment was performed.",
        "Std is sample ddof1; quartiles use linear interpolation. "
        "Independent verification/final pass gates are in 04_verifier_receipt.json.",
    ]
    report = {
        "phase": "BA-1",
        "generation_completed": True,
        "notes": common_notes,
        "variants": manifests,
        "ba500_control": control,
        "tables": tables,
        "independent_verifier_receipt": "04_verifier_receipt.json",
    }
    exclusive_json(DOC / "01_generation_report.json", report)
    exclusive_text(
        DOC / "01_generation_report.md",
        "# BA-House Scale Generation Report\n\n"
        + "\n\n".join(common_notes)
        + "\n\n"
        + "\n\n".join(md_table(t) for t in tables)
        + "\n\n## Dataset Hashes\n\n"
        + "\n\n".join(
            f"{m['dataset_variant']}: `{m['dataset']['path']}`; SHA256 `{m['dataset']['sha256']}`."
            for m in manifests
        )
        + "\n",
    )
    parity = {
        "tables": tables,
        "distributions": distributions,
        "ba500_control": control,
        "notes": common_notes,
        "per_class_splits": {
            m["dataset_variant"]: m["structure"]["split_class_counts"] for m in manifests
        },
    }
    exclusive_json(DOC / "02_scale_parity_report.json", parity)
    control_table = table(
        "BA500 Frozen Scientific Tensor Parity",
        ["Tensor", "ExactEquality"],
        list(control["tensor_parity"].items()),
    )
    exclusive_text(
        DOC / "02_scale_parity_report.md",
        "# BA-House Scale Parity And Complexity\n\n"
        + "\n\n".join(common_notes)
        + "\n\n"
        + "\n\n".join(md_table(t) for t in [*tables, distributions, control_table])
        + "\n\nBA500 control: `"
        + control["verdict"]
        + "`. Scientific tensor parity is distinct from serialized file identity.\n\n"
        + md_table(
            table(
                "Per-Class Splits",
                ["Variant", "Split", "Class0", "Class1", "Class2", "Class3"],
                [
                    [m["dataset_variant"], split] + [counts[str(c)] for c in range(4)]
                    for m in manifests
                    for split, counts in m["structure"]["split_class_counts"].items()
                ],
            )
        )
        + "\n",
    )
    exclusive_json(
        DOC / "generation_log.json",
        {
            "started_at_utc": started,
            "finished_at_utc": timestamp(),
            "events": events,
            "stdout": capture_out.getvalue()
            + "\n".join(json.dumps(event) for event in events)
            + "\n"
            + control["verdict"]
            + "\nBA_HOUSE_SCALE_GENERATION_COMPLETE_AWAITING_INDEPENDENT_VERIFIER\n",
            "stderr": capture_err.getvalue(),
            "execution_boundary": "Exactly three saved dataset variants; "
            "two in-memory generations each; no model-related calls",
        },
    )
    after_manifest(before)
    master = {
        "created_at_utc": timestamp(),
        "phase": "BA-1",
        "track": "TRACK A ONLY - CLEAN SCALE ROBUSTNESS",
        "environment": environment,
        "variants": manifests,
        "ba500_control": control,
        "baseline_preservation": "BA_HOUSE_SCALE_BASELINE_PRESERVATION_PASS",
        "verification_status": "Generation gates passed; final independent verifier required",
        "independent_verifier_receipt": (DOC / "04_verifier_receipt.json")
        .relative_to(ROOT)
        .as_posix(),
        "file_hashes": [file_record(p) for p in output_paths() if p.exists()],
        "source_hashes": [
            file_record(ROOT / p)
            for p in (
                "src/gnnexplainer_benchmark/extensions/__init__.py",
                "src/gnnexplainer_benchmark/extensions/ba_house_scale.py",
                "scripts/extensions/generate_ba_house_scale.py",
                "scripts/extensions/verify_ba_house_scale.py",
            )
        ],
        "self_hash_note": "Master self-hash and final verifier receipt are excluded "
        "to avoid a hash cycle; verifier hashes the master separately.",
    }
    exclusive_json(ART / "ba_house_scale_dataset_manifest.json", master)
    print(control["verdict"])
    print("BA_HOUSE_SCALE_GENERATION_COMPLETE_AWAITING_INDEPENDENT_VERIFIER")


if __name__ == "__main__":
    main()
