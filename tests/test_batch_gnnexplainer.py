from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gnnexplainer_benchmark.data.target_selection import ExplainerTarget
from gnnexplainer_benchmark.evaluation.batch_gnnexplainer import (
    PER_NODE_FIELDS,
    BatchConfig,
    batch_signature,
    can_resume_node,
    failure_row,
    raw_explanation_matches,
    read_explainer_targets,
    summarize_metric_rows,
    validate_fidelity_alias,
    validate_probability_diagnostics,
    validate_runtime_reporting,
    validate_target_order_and_uniqueness,
    write_per_node_metrics,
)
from gnnexplainer_benchmark.evaluation.budget import selected_edge_count_for_ratio


def make_target(node_id: int) -> ExplainerTarget:
    return ExplainerTarget(
        node_id=node_id,
        label=1,
        prediction=1,
        confidence=0.5,
        split="test",
        is_ground_truth_motif=True,
    )


def test_target_order_is_deterministic_and_unique() -> None:
    validate_target_order_and_uniqueness([make_target(1), make_target(2)])

    with pytest.raises(ValueError, match="Duplicate"):
        validate_target_order_and_uniqueness([make_target(1), make_target(1)])

    with pytest.raises(ValueError, match="sorted"):
        validate_target_order_and_uniqueness([make_target(2), make_target(1)])


def test_read_explainer_targets_rejects_duplicate_nodes(tmp_path: Path) -> None:
    path = tmp_path / "targets.csv"
    path.write_text(
        "node_id,label,prediction,confidence,split,is_ground_truth_motif\n"
        "1,1,1,0.5,test,True\n"
        "1,1,1,0.4,test,True\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        read_explainer_targets(path)


def test_per_node_metrics_schema_is_written(tmp_path: Path) -> None:
    row = failure_row(make_target(1), seed=42, error=RuntimeError("boom"))
    path = tmp_path / "per_node_metrics.csv"

    write_per_node_metrics(path, [row])

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PER_NODE_FIELDS
        rows = list(reader)
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "boom"


def test_summary_metrics_include_distribution_statistics() -> None:
    rows = [
        {
            "status": "success",
            "edge_auroc": 0.8,
            "edge_ap": 0.2,
            "fid_plus": 0,
            "fid_minus": 1,
            "one_minus_fid_minus": 0,
            "remove_probability_delta": -0.1,
            "retain_probability": 0.3,
            "retain_probability_delta": -0.02,
            "fidelity_legacy": -0.1,
            "sparsity": 0.9,
            "runtime_seconds": 1.0,
        },
        {
            "status": "success",
            "edge_auroc": 1.0,
            "edge_ap": 0.4,
            "fid_plus": 1,
            "fid_minus": 0,
            "one_minus_fid_minus": 1,
            "remove_probability_delta": 0.1,
            "retain_probability": 0.5,
            "retain_probability_delta": -0.04,
            "fidelity_legacy": 0.1,
            "sparsity": 0.8,
            "runtime_seconds": 3.0,
        },
        {
            "status": "failed",
            "edge_auroc": None,
            "edge_ap": None,
            "fid_plus": None,
            "fid_minus": None,
            "one_minus_fid_minus": None,
            "remove_probability_delta": None,
            "retain_probability": None,
            "retain_probability_delta": None,
            "fidelity_legacy": None,
            "sparsity": None,
            "runtime_seconds": None,
        },
    ]

    summary = summarize_metric_rows(rows)
    by_metric = {row["metric"]: row for row in summary}

    assert by_metric["edge_auroc"]["n"] == 2
    assert by_metric["edge_auroc"]["mean"] == pytest.approx(0.9)
    assert by_metric["edge_auroc"]["median"] == pytest.approx(0.9)
    assert by_metric["fid_plus"]["mean"] == pytest.approx(0.5)
    assert by_metric["fid_minus"]["mean"] == pytest.approx(0.5)
    assert by_metric["one_minus_fid_minus"]["mean"] == pytest.approx(0.5)
    assert by_metric["fidelity_legacy"]["mean"] == by_metric["remove_probability_delta"]["mean"]
    assert by_metric["runtime_seconds"]["min"] == 1.0
    assert by_metric["runtime_seconds"]["max"] == 3.0


def test_fidelity_alias_validation_enforces_remove_delta_equivalence() -> None:
    validate_fidelity_alias([{"fidelity": 0.1, "remove_delta": 0.1}])

    with pytest.raises(ValueError, match="fidelity must equal remove_delta"):
        validate_fidelity_alias([{"fidelity": 0.1, "remove_delta": 0.2}])


def test_probability_diagnostic_validation_enforces_legacy_equivalence() -> None:
    validate_probability_diagnostics(
        [{"fidelity_legacy": 0.1, "remove_probability_delta": 0.1}]
    )

    with pytest.raises(ValueError, match="fidelity_legacy must equal"):
        validate_probability_diagnostics(
            [{"fidelity_legacy": 0.1, "remove_probability_delta": 0.2}]
        )


def test_runtime_reporting_rejects_zero_generation_runtime_for_reused_masks() -> None:
    with pytest.raises(ValueError, match="0.0 generation runtime"):
        validate_runtime_reporting(
            [
                {
                    "explanation_source": "reused_existing_raw_mask",
                    "generation_runtime_seconds": 0.0,
                    "runtime_seconds": 0.0,
                }
            ]
        )


def test_runtime_reporting_allows_null_generation_runtime_for_reused_masks() -> None:
    validate_runtime_reporting(
        [
            {
                "explanation_source": "reused_existing_raw_mask",
                "generation_runtime_seconds": "",
                "runtime_seconds": "",
            }
        ]
    )


def test_batch_resume_uses_signature(tmp_path: Path) -> None:
    config = BatchConfig(
        checkpoint_path=Path("checkpoint.pt"),
        target_file=Path("targets.csv"),
        explainer_config={"explainer": {"algorithm": "test"}},
        metrics_config={"evaluation": {"top_k_edges": 25}},
        seed=42,
        selected_edge_ratio=0.2,
        output_root=tmp_path,
        metrics_output_dir=tmp_path,
        device="cpu",
        dataset_checksum="abc",
        num_hops=3,
    )
    target = make_target(1)
    signature = batch_signature(config, target)
    output_dir = tmp_path / "node_1"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        f'{{"batch_run_signature": "{signature}"}}\n',
        encoding="utf-8",
    )
    (output_dir / "edge_mask.pt").write_bytes(b"edge")
    (output_dir / "node_mask.pt").write_bytes(b"node")

    assert can_resume_node(output_dir, signature) is True
    assert can_resume_node(output_dir, "different") is False


def test_raw_explanation_matches_requires_same_checkpoint_config_and_target(tmp_path: Path) -> None:
    config = BatchConfig(
        checkpoint_path=Path("checkpoint.pt"),
        target_file=Path("targets.csv"),
        explainer_config={"explainer": {"algorithm": "test", "seed": 42}},
        metrics_config={"evaluation": {"selected_edge_ratio": 0.2}},
        seed=42,
        selected_edge_ratio=0.2,
        output_root=tmp_path,
        metrics_output_dir=tmp_path,
        device="cpu",
        dataset_checksum="abc",
        num_hops=3,
    )
    metadata = {
        "raw_masks_saved": True,
        "dataset_checksum": "abc",
        "checkpoint": {"path": "checkpoint.pt"},
        "explainer_config": {"explainer": {"algorithm": "test", "seed": 42}},
        "num_hops": 3,
        "target": {"node_id": 1, "label": 1, "prediction": 1},
    }

    assert raw_explanation_matches(metadata, make_target(1), config) is True

    metadata["checkpoint"]["path"] = "other.pt"
    assert raw_explanation_matches(metadata, make_target(1), config) is False


def test_ratio_budget_rounding_rule_for_phase4c() -> None:
    assert selected_edge_count_for_ratio(24, 0.2) == 5
    assert selected_edge_count_for_ratio(26, 0.2) == 6
    assert selected_edge_count_for_ratio(866, 0.2) == 174
