import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_task25_outputs() -> None:
    if (ROOT / "results/metrics/shapeggen_2x2/paired_matrix.csv").exists():
        return
    subprocess.run(
        [sys.executable, "scripts/build_shapeggen_2x2_cross_synthetic.py"],
        cwd=ROOT,
        check=True,
    )


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_shapeggen_2x2_matrix_excludes_random_and_has_four_cells() -> None:
    ensure_task25_outputs()
    rows = read_csv("results/metrics/shapeggen_2x2/main.csv")
    cells = {(row["backbone"], row["explainer"]) for row in rows}

    assert len(rows) == 4
    assert cells == {
        ("GCN", "GNNExplainer"),
        ("GCN", "PGExplainer"),
        ("GraphSAGE", "GNNExplainer"),
        ("GraphSAGE", "PGExplainer"),
    }
    assert "Random" not in {row["explainer"] for row in rows}
    assert {row["n_nodes"] for row in rows} == {"167"}


def test_shape_paired_matrix_uses_frozen_common_final_targets() -> None:
    ensure_task25_outputs()
    paired = read_csv("results/metrics/shapeggen_2x2/paired_matrix.csv")
    frozen_targets = read_csv("configs/shapeggen/v1/shapeggen_2x2_targets.csv")

    assert len(paired) == 167
    assert [row["node_id"] for row in paired] == [row["node_id"] for row in frozen_targets]
    assert len({row["node_id"] for row in paired}) == 167
    assert all(row["gcn_gnnexplainer_edge_auroc"] for row in paired)
    assert all(row["graphsage_pgexplainer_edge_ap"] for row in paired)


def test_random_is_node_level_and_outside_2x2() -> None:
    ensure_task25_outputs()
    random_rows = read_csv("results/metrics/shapeggen_2x2/random_node_level_summary.csv")
    main_rows = read_csv("results/metrics/shapeggen_2x2/main.csv")

    assert random_rows
    assert {row["aggregation"] for row in random_rows} == {
        "raw seed rows -> mean within node -> summarize across nodes"
    }
    assert "Random" not in {row["explainer"] for row in main_rows}


def test_cross_synthetic_outputs_do_not_pool_node_rows() -> None:
    ensure_task25_outputs()
    freeze = json.loads((ROOT / "docs/cross_synthetic_freeze.json").read_text(encoding="utf-8"))
    consistency = read_csv("results/metrics/cross_synthetic/conclusion_consistency_matrix.csv")
    context = read_csv("results/metrics/cross_synthetic/dataset_context.csv")

    assert freeze["reran_experiments"] is False
    assert freeze["protocol"]["cross_dataset_pooling"] == "not used for primary analysis"
    assert freeze["protocol"]["random_position"] == "outside_2x2_matrix"
    assert len(consistency) == 7
    assert {row["dataset"] for row in context} == {"BA-House", "ShapeGGen"}


def test_task25_reports_avoid_forbidden_causal_backbone_wording() -> None:
    ensure_task25_outputs()
    report_paths = [
        ROOT / "docs/provenance/historical_reports/shapeggen_2x2_report.md",
        ROOT / "docs/provenance/historical_reports/cross_synthetic_ba_house_shapeggen_report.md",
    ]
    forbidden = [
        "backbone effect",
        "causal backbone",
        "universally best",
        "pseudo-dataset",
        "faithfulness",
        "fidelity",
    ]

    for path in report_paths:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            assert phrase not in text
