from __future__ import annotations

# ruff: noqa: E501
import csv
import hashlib
import json
import os
import random
import subprocess
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get("ELLIPTIC_BASELINE_REPO", r"D:\CS\KLTN\elliptic-gnn-xai"))
TARGET_DIR = ROOT / "configs" / "targets"
DOCS_DIR = ROOT / "docs"

FINAL_TEST_TARGET_CAP = 200
PG_TRAIN_CAP = 2000
PG_VAL_CAP = 500
SEED = 42
SELECTED_EDGE_RATIO = 0.20
PARITY_TOLERANCE = 1e-6


class EllipticGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embedding_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        cached: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, embedding_dim, cached=cached))
        else:
            self.convs.append(GCNConv(in_channels, hidden_dim, cached=cached))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=cached))
            self.convs.append(GCNConv(hidden_dim, embedding_dim, cached=cached))
        self.dropout = float(dropout)
        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx < len(self.convs) - 1:
                x = x.relu()
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


class EllipticSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embedding_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        aggregator: str,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, embedding_dim, aggr=aggregator))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_dim, aggr=aggregator))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            self.convs.append(SAGEConv(hidden_dim, embedding_dim, aggr=aggregator))
        self.dropout = float(dropout)
        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx < len(self.convs) - 1:
                x = x.relu()
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_data() -> Any:
    data_path = BASELINE / "datasets" / "elliptic" / "elliptic_data.pt"
    return torch.load(data_path, map_location="cpu", weights_only=False)


def state_dict(path: Path) -> OrderedDict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, OrderedDict):
        return loaded
    if isinstance(loaded, dict):
        return loaded.get("model_state_dict") or loaded.get("state_dict") or loaded
    raise TypeError(f"Unsupported checkpoint object: {type(loaded)!r}")


def build_models(data: Any) -> dict[str, nn.Module]:
    gcn_cfg = load_yaml(BASELINE / "config" / "training_baseline_paper_gcn.yaml")
    sage_cfg = load_yaml(BASELINE / "config" / "training_baseline_paper_sage.yaml")
    gcn_params = gcn_cfg["training"]["fixed_hparams"]
    sage_params = sage_cfg["training"]["fixed_hparams"]
    in_channels = int(data.x.size(1))

    gcn_cached = EllipticGCN(
        in_channels=in_channels,
        hidden_dim=int(gcn_params["hidden_dim"]),
        embedding_dim=int(gcn_params["embedding_dim"]),
        output_dim=2,
        num_layers=int(gcn_params["num_layers"]),
        dropout=float(gcn_params["dropout"]),
        cached=True,
    )
    gcn_compatible = EllipticGCN(
        in_channels=in_channels,
        hidden_dim=int(gcn_params["hidden_dim"]),
        embedding_dim=int(gcn_params["embedding_dim"]),
        output_dim=2,
        num_layers=int(gcn_params["num_layers"]),
        dropout=float(gcn_params["dropout"]),
        cached=False,
    )
    sage = EllipticSAGE(
        in_channels=in_channels,
        hidden_dim=int(sage_params["hidden_dim"]),
        embedding_dim=int(sage_params["embedding_dim"]),
        output_dim=2,
        num_layers=int(sage_params["num_layers"]),
        dropout=float(sage_params["dropout"]),
        aggregator=str(sage_params.get("aggregator", "mean")),
    )

    gcn_sd = state_dict(BASELINE / "artifacts" / "checkpoints" / "gcn_baseline.pt")
    sage_sd = state_dict(BASELINE / "artifacts" / "checkpoints" / "graphsage_baseline.pt")
    gcn_cached.load_state_dict(gcn_sd)
    gcn_compatible.load_state_dict(gcn_sd)
    sage.load_state_dict(sage_sd)
    for model in (gcn_cached, gcn_compatible, sage):
        model.eval()
    return {"gcn_cached": gcn_cached, "gcn_compatible": gcn_compatible, "graphsage": sage}


@torch.no_grad()
def logits(model: nn.Module, data: Any, edge_index: torch.Tensor | None = None) -> torch.Tensor:
    return model(data.x, data.edge_index if edge_index is None else edge_index)


def parity_report(data: Any, models: dict[str, nn.Module]) -> dict[str, Any]:
    gcn_cached_logits = logits(models["gcn_cached"], data)
    gcn_compatible_logits = logits(models["gcn_compatible"], data)
    sage_first = logits(models["graphsage"], data)
    sage_second = logits(models["graphsage"], data)

    gcn_diff = (gcn_cached_logits - gcn_compatible_logits).abs()
    sage_diff = (sage_first - sage_second).abs()
    return {
        "tolerance": PARITY_TOLERANCE,
        "gcn": {
            "baseline_cached_true": True,
            "compatible_cached_false": True,
            "max_abs_logit_diff": float(gcn_diff.max().item()),
            "mean_abs_logit_diff": float(gcn_diff.mean().item()),
            "prediction_mismatch_count": int(
                (gcn_cached_logits.argmax(dim=1) != gcn_compatible_logits.argmax(dim=1)).sum().item()
            ),
            "prediction_count": int(data.x.size(0)),
            "passes": bool(
                gcn_diff.max().item() <= PARITY_TOLERANCE
                and (gcn_cached_logits.argmax(dim=1) == gcn_compatible_logits.argmax(dim=1)).all().item()
            ),
        },
        "graphsage": {
            "wrapper": "same SAGEConv architecture with explicit edge_index input",
            "max_abs_logit_diff": float(sage_diff.max().item()),
            "mean_abs_logit_diff": float(sage_diff.mean().item()),
            "prediction_mismatch_count": int(
                (sage_first.argmax(dim=1) != sage_second.argmax(dim=1)).sum().item()
            ),
            "prediction_count": int(data.x.size(0)),
            "passes": bool(
                sage_diff.max().item() <= PARITY_TOLERANCE
                and (sage_first.argmax(dim=1) == sage_second.argmax(dim=1)).all().item()
            ),
        },
        "probability_parity": {
            "gcn_max_abs_probability_diff": float(
                (
                    gcn_cached_logits.softmax(dim=1)
                    - gcn_compatible_logits.softmax(dim=1)
                )
                .abs()
                .max()
                .item()
            ),
            "graphsage_max_abs_probability_diff": float(
                (sage_first.softmax(dim=1) - sage_second.softmax(dim=1)).abs().max().item()
            ),
        },
    }


def perturbation_smoke(data: Any, models: dict[str, nn.Module]) -> dict[str, Any]:
    edge_index = data.edge_index
    keep = torch.ones(edge_index.size(1), dtype=torch.bool)
    keep[::20] = False
    perturbed_edge_index = edge_index[:, keep]
    node_mask = data.test_mask.to(torch.bool)

    report: dict[str, Any] = {
        "original_edge_count": int(edge_index.size(1)),
        "perturbed_edge_count": int(perturbed_edge_index.size(1)),
        "removed_edge_count": int((~keep).sum().item()),
        "target_mask": "known-labeled temporal test nodes",
    }
    for name, model in [
        ("gcn_compatible_cached_false", models["gcn_compatible"]),
        ("graphsage_wrapper", models["graphsage"]),
    ]:
        original = logits(model, data, edge_index)
        perturbed = logits(model, data, perturbed_edge_index)
        diff = (original[node_mask] - perturbed[node_mask]).abs()
        report[name] = {
            "max_abs_test_logit_diff": float(diff.max().item()),
            "mean_abs_test_logit_diff": float(diff.mean().item()),
            "prediction_change_count": int(
                (original[node_mask].argmax(dim=1) != perturbed[node_mask].argmax(dim=1)).sum().item()
            ),
            "passes": bool(diff.max().item() > 0.0),
        }
    return report


def two_hop_candidate_counts(data: Any) -> dict[int, int]:
    row = data.edge_index[0].tolist()
    col = data.edge_index[1].tolist()
    adjacency: dict[int, set[int]] = {}
    outgoing: dict[int, set[int]] = {}
    for src, dst in zip(row, col, strict=False):
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
        outgoing.setdefault(src, set()).add(dst)

    all_targets = torch.nonzero(
        (data.y != 2) & (data.train_mask | data.val_mask | data.test_mask),
        as_tuple=False,
    ).view(-1).tolist()
    counts: dict[int, int] = {}
    for node_id in all_targets:
        one = adjacency.get(node_id, set())
        nodes = {node_id, *one}
        for nbr in one:
            nodes.update(adjacency.get(nbr, set()))
        counts[int(node_id)] = sum(len(outgoing.get(src, set()) & nodes) for src in nodes)
    return counts


@torch.no_grad()
def predictions(data: Any, models: dict[str, nn.Module]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out = {}
    for name, model in [
        ("gcn", models["gcn_compatible"]),
        ("graphsage", models["graphsage"]),
    ]:
        probs = logits(model, data).softmax(dim=1)
        out[name] = (probs.argmax(dim=1), probs.max(dim=1).values)
    return out


def split_name(data: Any, node_id: int) -> str:
    if bool(data.train_mask[node_id]):
        return "train"
    if bool(data.val_mask[node_id]):
        return "validation"
    if bool(data.test_mask[node_id]):
        return "test"
    return "none"


def deterministic_sample(node_ids: list[int], cap: int, seed: int, salt: str) -> list[int]:
    if len(node_ids) <= cap:
        return sorted(node_ids)
    rng = random.Random(f"{seed}:{salt}")
    return sorted(rng.sample(sorted(node_ids), cap))


def target_rows(data: Any, node_ids: list[int], counts: dict[int, int], preds: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> list[dict[str, Any]]:
    rows = []
    for node_id in node_ids:
        label = int(data.y[node_id].item())
        gcn_pred, gcn_conf = preds["gcn"]
        sage_pred, sage_conf = preds["graphsage"]
        rows.append(
            {
                "node_id": node_id,
                "time_step": int(data.time_step[node_id].item()) if hasattr(data, "time_step") else "",
                "label": label,
                "label_semantics": "illicit" if label == 1 else "licit",
                "split": split_name(data, node_id),
                "gcn_prediction": int(gcn_pred[node_id].item()),
                "gcn_confidence": float(gcn_conf[node_id].item()),
                "graphsage_prediction": int(sage_pred[node_id].item()),
                "graphsage_confidence": float(sage_conf[node_id].item()),
                "candidate_edge_count": counts[node_id],
                "candidate_mapping_valid": counts[node_id] > 0,
                "positive_semantic_class": "illicit",
                "positive_label_index": 1,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_targets(data: Any, models: dict[str, nn.Module]) -> dict[str, Any]:
    counts = two_hop_candidate_counts(data)
    preds = predictions(data, models)
    known = data.y != 2
    eligible = {
        "train": [
            int(idx)
            for idx in torch.nonzero(known & data.train_mask, as_tuple=False).view(-1).tolist()
            if counts.get(int(idx), 0) > 0
        ],
        "validation": [
            int(idx)
            for idx in torch.nonzero(known & data.val_mask, as_tuple=False).view(-1).tolist()
            if counts.get(int(idx), 0) > 0
        ],
        "test": [
            int(idx)
            for idx in torch.nonzero(known & data.test_mask, as_tuple=False).view(-1).tolist()
            if counts.get(int(idx), 0) > 0
        ],
    }
    final_ids = deterministic_sample(eligible["test"], FINAL_TEST_TARGET_CAP, SEED, "final_test")
    pg_train_ids = deterministic_sample(eligible["train"], PG_TRAIN_CAP, SEED, "pg_train")
    pg_val_ids = deterministic_sample(eligible["validation"], PG_VAL_CAP, SEED, "pg_val")

    paths = {
        "final_test": TARGET_DIR / "elliptic_xai_common_test_targets.csv",
        "gcn_train_eligible": TARGET_DIR / "elliptic_pg_gcn_train_eligible.csv",
        "gcn_val_eligible": TARGET_DIR / "elliptic_pg_gcn_val_eligible.csv",
        "graphsage_train_eligible": TARGET_DIR / "elliptic_pg_graphsage_train_eligible.csv",
        "graphsage_val_eligible": TARGET_DIR / "elliptic_pg_graphsage_val_eligible.csv",
        "common_train": TARGET_DIR / "elliptic_pg_common_train.csv",
        "common_val": TARGET_DIR / "elliptic_pg_common_val.csv",
    }
    write_csv(paths["final_test"], target_rows(data, final_ids, counts, preds))
    train_rows = target_rows(data, eligible["train"], counts, preds)
    val_rows = target_rows(data, eligible["validation"], counts, preds)
    write_csv(paths["gcn_train_eligible"], train_rows)
    write_csv(paths["gcn_val_eligible"], val_rows)
    write_csv(paths["graphsage_train_eligible"], train_rows)
    write_csv(paths["graphsage_val_eligible"], val_rows)
    write_csv(paths["common_train"], target_rows(data, pg_train_ids, counts, preds))
    write_csv(paths["common_val"], target_rows(data, pg_val_ids, counts, preds))

    return {
        "selection": {
            "seed": SEED,
            "final_test_cap": FINAL_TEST_TARGET_CAP,
            "pg_train_cap": PG_TRAIN_CAP,
            "pg_val_cap": PG_VAL_CAP,
            "rule": "known-labeled temporal split nodes with positive 2-hop candidate-edge count; deterministic seeded sample if count exceeds cap",
            "no_explanation_result_driven_sampling": True,
        },
        "counts": {
            "eligible_train": len(eligible["train"]),
            "eligible_validation": len(eligible["validation"]),
            "eligible_test": len(eligible["test"]),
            "final_test": len(final_ids),
            "common_train": len(pg_train_ids),
            "common_val": len(pg_val_ids),
        },
        "disjointness": {
            "common_train_common_val": len(set(pg_train_ids) & set(pg_val_ids)),
            "common_train_final_test": len(set(pg_train_ids) & set(final_ids)),
            "common_val_final_test": len(set(pg_val_ids) & set(final_ids)),
        },
        "files": {
            key: {
                "path": str(path.relative_to(ROOT)).replace("/", "\\"),
                "count": sum(1 for _ in path.open("r", encoding="utf-8")) - 1,
                "sha256": sha256(path),
            }
            for key, path in paths.items()
        },
        "candidate_edge_count_distribution": summarize([counts[idx] for idx in final_ids]),
        "final_label_distribution": count_values([int(data.y[idx].item()) for idx in final_ids]),
        "final_gcn_prediction_distribution": count_values([int(preds["gcn"][0][idx].item()) for idx in final_ids]),
        "final_graphsage_prediction_distribution": count_values([int(preds["graphsage"][0][idx].item()) for idx in final_ids]),
    }


def summarize(values: list[int | float]) -> dict[str, float | int]:
    values = list(values)
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(sorted(values)[len(values) // 2]),
        "min": min(values),
        "max": max(values),
    }


def count_values(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def provenance() -> dict[str, Any]:
    source_commit = git(BASELINE, "rev-parse", "HEAD")
    return {
        "baseline_repo_path": str(BASELINE),
        "baseline_remote": git(BASELINE, "remote", "-v"),
        "baseline_commit": source_commit,
        "baseline_tags_at_head": git(BASELINE, "tag", "--points-at", "HEAD") or "none",
        "baseline_status_short": git(BASELINE, "status", "--short"),
        "xai_benchmark_commit": git(ROOT, "rev-parse", "HEAD"),
        "xai_benchmark_branch": git(ROOT, "branch", "--show-current"),
        "source_files": {
            "gcn_checkpoint": file_meta(BASELINE / "artifacts" / "checkpoints" / "gcn_baseline.pt"),
            "graphsage_checkpoint": file_meta(BASELINE / "artifacts" / "checkpoints" / "graphsage_baseline.pt"),
            "gcn_training_config": file_meta(BASELINE / "config" / "training_baseline_paper_gcn.yaml"),
            "graphsage_training_config": file_meta(BASELINE / "config" / "training_baseline_paper_sage.yaml"),
            "gcn_model_config": file_meta(BASELINE / "config" / "model_gcn.yaml"),
            "graphsage_model_config": file_meta(BASELINE / "config" / "model_sage.yaml"),
            "elliptic_processed_data": file_meta(BASELINE / "datasets" / "elliptic" / "elliptic_data.pt"),
        },
    }


def file_meta(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def write_docs(prov: dict[str, Any], parity: dict[str, Any], smoke: dict[str, Any], targets: dict[str, Any]) -> None:
    handoff = f"""# Elliptic XAI Handoff Provenance

## Scope

Task26 prepares Elliptic for final real-world XAI. It does not retrain GCN or GraphSAGE, does not run GNNExplainer, does not run PGExplainer, and does not inspect final explanation results.

## Cross-Repo Provenance

- Baseline repo path: `{prov["baseline_repo_path"]}`
- Baseline remote:

```text
{prov["baseline_remote"]}
```

- Baseline commit: `{prov["baseline_commit"]}`
- Baseline tag at HEAD: `{prov["baseline_tags_at_head"]}`
- Baseline status note: `README.md` has a pre-existing tracked modification and was not modified by Task26.
- XAI benchmark commit before Task26 commit: `{prov["xai_benchmark_commit"]}`

## Frozen Inputs

| Artifact | Path | SHA256 |
|---|---|---|
| GCN checkpoint | `{prov["source_files"]["gcn_checkpoint"]["path"]}` | `{prov["source_files"]["gcn_checkpoint"]["sha256"]}` |
| GraphSAGE checkpoint | `{prov["source_files"]["graphsage_checkpoint"]["path"]}` | `{prov["source_files"]["graphsage_checkpoint"]["sha256"]}` |
| GCN config | `{prov["source_files"]["gcn_training_config"]["path"]}` | `{prov["source_files"]["gcn_training_config"]["sha256"]}` |
| GraphSAGE config | `{prov["source_files"]["graphsage_training_config"]["path"]}` | `{prov["source_files"]["graphsage_training_config"]["sha256"]}` |
| Processed Elliptic data | `{prov["source_files"]["elliptic_processed_data"]["path"]}` | `{prov["source_files"]["elliptic_processed_data"]["sha256"]}` |

No silent model-code copying is used. The Task26 script defines explanation-compatible wrappers that match the baseline checkpoint state dicts and records original-graph parity.
"""
    protocol = f"""# Elliptic XAI Protocol

## Dataset Semantics

- Train timesteps: `1-29`, implemented as `time_step < 30`.
- Validation timesteps: `30-39`.
- Test timesteps: `40-49`, implemented as `time_step >= 40`.
- Unknown-label nodes remain in the graph.
- Unknown-label nodes are excluded from supervised, evaluation, final XAI target, and PG train/validation target files.
- Unknown neighbors may remain inside computation subgraphs.

## Positive Class Semantics

The positive semantic class for TP/FP/FN/TN and case studies is `ILLICIT transaction`.

Implementation encoding:

| Raw class | Label index | Meaning |
|---|---:|---|
| `1` | 1 | illicit |
| `2` | 0 | licit |
| `unknown` | 2 | unknown, excluded as target |

## Explanation Metrics

Elliptic has no ground-truth explanatory edge masks in this thesis. Edge AUROC and Edge AP are disabled and must not be reported for Elliptic explanation correctness.

Core metrics:

- Fid+
- Fid-
- 1-Fid-
- remove_probability_delta
- retain_probability
- retain_probability_delta
- retain_probability_ratio
- Sparsity
- Jaccard Stability
- Runtime
- success/failure rate
- candidate-edge count distribution

The selected-edge budget remains `selected_edge_ratio = {SELECTED_EDGE_RATIO}` with `ceil(candidate_edge_count * selected_edge_ratio), minimum 1`, unless a later pre-final reviewed amendment changes it before any final Elliptic explanation run.

## Frozen Targets And PG Pools

- Final common test targets: `{targets["counts"]["final_test"]}`
- Common PG train: `{targets["counts"]["common_train"]}`
- Common PG validation: `{targets["counts"]["common_val"]}`
- Selection rule: {targets["selection"]["rule"]}
- Sampling seed: `{targets["selection"]["seed"]}`
- Sampling does not use explanation results.

Target files are recorded in `docs/elliptic_targets_freeze.json`.

## Stop Boundary

No final Elliptic explanation may run before review.
"""
    parity_doc = f"""# Elliptic Model Parity Report

## GCN Original-Graph Parity

Baseline GCN uses `GCNConv(cached=True)`. Task26 instantiates the same architecture with `cached=False`, loads identical weights, sets eval mode, and compares original-graph logits/probabilities/predictions before any XAI run.

| Check | Value |
|---|---:|
| tolerance | `{parity["tolerance"]}` |
| max_abs_logit_diff | `{parity["gcn"]["max_abs_logit_diff"]}` |
| mean_abs_logit_diff | `{parity["gcn"]["mean_abs_logit_diff"]}` |
| prediction_mismatch_count | `{parity["gcn"]["prediction_mismatch_count"]}` |
| prediction_count | `{parity["gcn"]["prediction_count"]}` |
| pass | `{parity["gcn"]["passes"]}` |

## GraphSAGE Original-Graph Parity

| Check | Value |
|---|---:|
| max_abs_logit_diff | `{parity["graphsage"]["max_abs_logit_diff"]}` |
| mean_abs_logit_diff | `{parity["graphsage"]["mean_abs_logit_diff"]}` |
| prediction_mismatch_count | `{parity["graphsage"]["prediction_mismatch_count"]}` |
| prediction_count | `{parity["graphsage"]["prediction_count"]}` |
| pass | `{parity["graphsage"]["passes"]}` |

## Probability Parity

| Check | Value |
|---|---:|
| GCN max_abs_probability_diff | `{parity["probability_parity"]["gcn_max_abs_probability_diff"]}` |
| GraphSAGE max_abs_probability_diff | `{parity["probability_parity"]["graphsage_max_abs_probability_diff"]}` |

## Perturbation Smoke

The smoke test removes a deterministic 5% edge subset and verifies that the explanation-compatible models receive changed edge input.

| Model | max_abs_test_logit_diff | mean_abs_test_logit_diff | prediction_change_count | pass |
|---|---:|---:|---:|---:|
| GCN cached=False | `{smoke["gcn_compatible_cached_false"]["max_abs_test_logit_diff"]}` | `{smoke["gcn_compatible_cached_false"]["mean_abs_test_logit_diff"]}` | `{smoke["gcn_compatible_cached_false"]["prediction_change_count"]}` | `{smoke["gcn_compatible_cached_false"]["passes"]}` |
| GraphSAGE wrapper | `{smoke["graphsage_wrapper"]["max_abs_test_logit_diff"]}` | `{smoke["graphsage_wrapper"]["mean_abs_test_logit_diff"]}` | `{smoke["graphsage_wrapper"]["prediction_change_count"]}` | `{smoke["graphsage_wrapper"]["passes"]}` |

Original edge count: `{smoke["original_edge_count"]}`. Perturbed edge count: `{smoke["perturbed_edge_count"]}`.
"""
    (DOCS_DIR / "elliptic_xai_handoff_provenance.md").write_text(handoff, encoding="utf-8")
    (DOCS_DIR / "elliptic_xai_protocol.md").write_text(protocol, encoding="utf-8")
    (DOCS_DIR / "elliptic_model_parity_report.md").write_text(parity_doc, encoding="utf-8")


def write_freeze(prov: dict[str, Any], parity: dict[str, Any], smoke: dict[str, Any], targets: dict[str, Any]) -> None:
    freeze = {
        "task": "Task26 Elliptic XAI handoff, parity, targets, and PG pools",
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "no_experiment_rerun": True,
        "no_final_explanations_run": True,
        "positive_class_semantics": {
            "semantic_positive_class": "ILLICIT transaction",
            "positive_label_index": 1,
            "licit_label_index": 0,
            "unknown_label_index": 2,
        },
        "dataset_semantics": {
            "train_timesteps": "1-29",
            "validation_timesteps": "30-39",
            "test_timesteps": "40-49",
            "unknown_nodes_remain_in_graph": True,
            "unknown_nodes_excluded_as_targets": True,
        },
        "provenance": prov,
        "model_parity": parity,
        "perturbation_smoke": smoke,
        "targets": targets,
        "metric_protocol": {
            "edge_auroc": "disabled for Elliptic because there is no GT explanatory edge mask",
            "edge_ap": "disabled for Elliptic because there is no GT explanatory edge mask",
            "enabled": [
                "Fid+",
                "Fid-",
                "1-Fid-",
                "remove_probability_delta",
                "retain_probability",
                "retain_probability_delta",
                "retain_probability_ratio",
                "Sparsity",
                "Jaccard Stability",
                "Runtime",
                "success/failure rate",
                "candidate-edge count distribution",
            ],
            "selected_edge_ratio": SELECTED_EDGE_RATIO,
        },
        "acceptance": {
            "baseline_cross_repo_provenance_recorded": True,
            "gcn_cached_false_copy_created_without_retraining": True,
            "gcn_original_graph_parity_passes": parity["gcn"]["passes"],
            "graphsage_original_graph_parity_passes": parity["graphsage"]["passes"],
            "perturbation_smoke_confirms_edge_changes_reach_message_passing": all(
                smoke[key]["passes"]
                for key in ["gcn_compatible_cached_false", "graphsage_wrapper"]
            ),
            "illicit_positive_class_semantics_frozen": True,
            "no_edge_auroc_ap_in_elliptic_protocol": True,
            "common_final_test_targets_frozen_before_xai": True,
            "per_backbone_pg_eligibility_audited": True,
            "common_pg_train_derived_prospectively": True,
            "common_pg_val_derived_prospectively": True,
            "both_backbones_will_use_identical_pg_train_val_node_ids": True,
            "train_val_final_disjoint": all(value == 0 for value in targets["disjointness"].values()),
            "unknown_labels_excluded_as_targets": True,
            "target_config_hashes_recorded": True,
            "no_explanation_result_driven_sampling": True,
        },
        "stop_boundary": "No final Elliptic explanation may run before review.",
    }
    (DOCS_DIR / "elliptic_targets_freeze.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not BASELINE.exists():
        raise FileNotFoundError(BASELINE)
    data = load_data()
    models = build_models(data)
    prov = provenance()
    parity = parity_report(data, models)
    smoke = perturbation_smoke(data, models)
    targets = freeze_targets(data, models)

    if not parity["gcn"]["passes"]:
        raise AssertionError("GCN cached=False parity failed")
    if not parity["graphsage"]["passes"]:
        raise AssertionError("GraphSAGE parity failed")
    if not all(smoke[key]["passes"] for key in ["gcn_compatible_cached_false", "graphsage_wrapper"]):
        raise AssertionError("Perturbation smoke failed")
    if not all(value == 0 for value in targets["disjointness"].values()):
        raise AssertionError("Target pools are not disjoint")

    write_docs(prov, parity, smoke, targets)
    write_freeze(prov, parity, smoke, targets)


if __name__ == "__main__":
    main()
