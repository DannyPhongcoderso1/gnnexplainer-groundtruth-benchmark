# Phase 4 Evaluation Report

## 1. Dataset Và Model Checkpoint

Dataset benchmark hiện tại là BA-House synthetic graph dùng cho đánh giá XAI có ground-truth motif.

- Artifact: `artifacts/data/ba_house.pt`
- Metadata: `artifacts/data/ba_house_metadata.json`
- Dataset SHA256: `4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94`
- Nodes: `500`
- Edges: `3398`
- Node features: `10`
- Label distribution: `{0: 300, 1: 80, 2: 80, 3: 40}`
- Ground-truth motif nodes: `200`
- Ground-truth motif edges: `480`
- Split: train `300`, validation `100`, test `100`
- Dataset seed: `42`

Frozen GCN checkpoint:

- Checkpoint: `artifacts/checkpoints/ba_house_gcn_best.pt`
- Checkpoint SHA256: `4C81814172650E20B6EDEE5FC97BC66CAE46AC1E8BA76BAEC2226C59A7D1D507`
- Model config: `configs/model/gcn.yaml`
- Model type: GCN with PyG `GCNConv`
- Hidden channels: `32`
- Layers: `3`
- Dropout: `0.2`
- `cached=false`
- Training seed: `42`

The test suite includes checkpoint reload and model-freeze checks. Phase 4C and Phase 4D manifests report `model_weights_unchanged=true`.

## Code And Environment Freeze

Repository state recorded for this Phase 4 report:

- Git commit hash: `26a5fc11122c52b45557a2ea8cf7ed21d9f581ed`
- Git branch: `feature/single-node-gnnexplainer`
- Git tag at HEAD: `phase4c-gnnexplainer-v2`

Runtime environment recorded from the active virtual environment:

- Python: `3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]`
- PyTorch: `2.11.0+cu128`
- PyTorch Geometric: `2.8.0.post1`
- CUDA runtime reported by PyTorch: `12.8`
- CUDA available: `True`
- Device/GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`

## 2. Target Selection

Frozen explainer target files:

- Validation targets: `results/metrics/explainer_validation_targets.csv`
- Test targets: `results/metrics/explainer_test_targets.csv`

Validation target nodes:

```text
311, 380, 385, 450
```

Test target nodes:

```text
306, 315, 316, 355, 365, 375, 376, 411, 445, 465, 486, 491
```

Validation and test explainer targets have no node overlap. Target order is deterministic and duplicate target ids are rejected by tests.

## 3. Metric Definitions

All Phase 4 metrics use the shared evaluation code under `src/gnnexplainer_benchmark/evaluation/`.

- Edge AUROC: AUROC between continuous candidate-edge explanation scores and binary ground-truth edge labels.
- Edge AP: average precision over the same continuous candidate-edge scores and binary ground-truth edge labels.
- Candidate-edge scope: computation subgraph around the target node using the model hop count.
- Sparsity: `1 - selected_edge_count / candidate_edge_count`.
- Runtime: recorded separately where possible as generation runtime and evaluation runtime.
- Jaccard stability: mean pairwise Jaccard similarity of selected edge sets across seeds.

Fidelity fields:

- `remove_probability_delta = original_probability - remove_probability`
- `retain_probability_delta = original_probability - retain_probability`
- `fidelity_legacy = remove_probability_delta`
- `fid_plus = 1` if removing selected edges changes the original prediction, otherwise `0`
- `fid_minus = 1` if retaining only selected edges changes the original prediction, otherwise `0`
- `one_minus_fid_minus = 1 - fid_minus`

## 4. Candidate-Edge Scope Và Budget Protocol

Frozen evaluation protocol:

```yaml
edge_ranking_metric_scope: continuous_scores_over_full_candidate_edge_set
budget_protocol: fixed_candidate_edge_ratio
selected_edge_ratio: 0.2
rounding_rule: ceil(candidate_edge_count * selected_edge_ratio), minimum 1
```

Edge AUROC and Edge AP are computed from continuous scores over the full candidate-edge set. Fidelity, retain behavior, sparsity, and stability use the selected edge set produced by the frozen 20% candidate-edge budget.

The same selected edge set is used for remove-side and retain-side perturbations.

## 5. GNNExplainer Result Summary

Frozen Phase 4C GNNExplainer results:

- Per-node metrics: `results/metrics/frozen_phase4c_v2/per_node_metrics.csv`
- Summary: `results/metrics/frozen_phase4c_v2/summary_metrics.csv`
- Manifest: `results/metrics/frozen_phase4c_v2/batch_gnnexplainer_manifest.json`
- Raw explanation root: `results/explanations/frozen_phase4c_v2/`

Batch result:

- Targets: `12`
- Success: `12`
- Failed: `0`
- Raw explanations reused: `12`
- Raw explanations generated: `0`
- Seed: `42`
- Device: `cuda`

Main metrics:

| Metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Edge AUROC | 0.870617 | 0.077797 | 0.892918 | 0.678571 | 0.935044 |
| Edge AP | 0.527731 | 0.236662 | 0.584178 | 0.106239 | 0.781677 |
| Fid+ | 0.666667 | 0.492366 | 1.000000 | 0.000000 | 1.000000 |
| Fid- | 0.250000 | 0.452267 | 0.000000 | 0.000000 | 1.000000 |
| 1-Fid- | 0.750000 | 0.452267 | 1.000000 | 0.000000 | 1.000000 |
| Fidelity legacy | -0.000357 | 0.006904 | 0.002986 | -0.015010 | 0.007155 |
| Sparsity | 0.793694 | 0.008901 | 0.797033 | 0.769231 | 0.800000 |
| Generation runtime seconds | 0.129505 | 0.177384 | 0.075328 | 0.068638 | 0.692148 |

## 6. Stability Summary

GNNExplainer multi-seed stability:

- Output: `results/metrics/phase4d_stability/gnnexplainer_stability.csv`
- Summary: `results/metrics/phase4d_stability/gnnexplainer_stability_summary.csv`
- Seeds: `0, 1, 2, 3, 4`
- Targets: `12`
- Success nodes: `12`
- Failed nodes: `0`
- Pairwise comparisons per target: `10`

Result:

| Metric | n | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Mean pairwise Jaccard | 12 | 0.441272 | 0.279181 | 0.368264 | 0.159524 | 0.977273 |

Random Jaccard stability:

- Pairwise output: `results/metrics/phase4d_random_baseline/random_stability_pairwise.csv`
- Per-node output: `results/metrics/phase4d_random_baseline/random_stability_per_node.csv`
- Summary: `results/metrics/phase4d_random_baseline/random_stability_summary.csv`
- Seeds: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Pairwise comparisons per target: `45`
- Total pairwise comparisons: `540`
- Targets success: `12`
- Failed: `0`

Result:

| Metric | n targets | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Random mean pairwise Jaccard | 12 | 0.126536 | 0.017376 | 0.123461 | 0.108960 | 0.169899 |

## 7. Random Baseline Summary

Random baseline uses the same target nodes, candidate-edge scope, selected-edge ratio, rounding rule, model checkpoint, and metric implementation as the GNNExplainer evaluation.

- Output root: `results/explanations/phase4d_random_baseline/`
- Per-node metrics: `results/metrics/phase4d_random_baseline/random_baseline_per_node.csv`
- Summary: `results/metrics/phase4d_random_baseline/random_baseline_summary.csv`
- Seeds: `0..9`
- Runs: `120`
- Success: `120`
- Failed: `0`

Main metrics:

| Metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Edge AUROC | 0.496727 | 0.102063 | 0.500951 | 0.196970 | 0.754630 |
| Edge AP | 0.277178 | 0.233067 | 0.216445 | 0.012323 | 0.726835 |
| Fid+ | 0.333333 | 0.473381 | 0.000000 | 0.000000 | 1.000000 |
| Fid- | 0.625000 | 0.486153 | 1.000000 | 0.000000 | 1.000000 |
| 1-Fid- | 0.375000 | 0.486153 | 0.000000 | 0.000000 | 1.000000 |
| Fidelity legacy | -0.000994 | 0.006809 | -0.000784 | -0.027685 | 0.020590 |
| Sparsity | 0.793694 | 0.008557 | 0.797033 | 0.769231 | 0.800000 |
| Evaluation runtime seconds | 0.021375 | 0.022475 | 0.018677 | 0.017017 | 0.262515 |

Current comparison table:

- `results/metrics/phase4d_comparison.csv`

| Method | Edge AUROC | Edge AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability | Recorded Runtime* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.496727 | 0.277178 | 0.333333 | 0.625000 | 0.375000 | 0.793694 | 0.126536 | 0.021375 |
| GNNExplainer | 0.870617 | 0.527731 | 0.666667 | 0.250000 | 0.750000 | 0.793694 | 0.441272 | 0.129505 |

* Runtime semantics differ by method. GNNExplainer reports explanation-generation time, while Random reports random edge-selection plus evaluation time. Runtime values are descriptive efficiency references and should not be interpreted as strictly identical computational stages.

## 8. Freeze Checklist

Dataset:

- BA-House artifact exists at `artifacts/data/ba_house.pt`
- Metadata/checksum stored in `artifacts/data/ba_house_metadata.json`
- Train/validation/test split stored in metadata

Model:

- GCN checkpoint exists at `artifacts/checkpoints/ba_house_gcn_best.pt`
- Model config fixed in `configs/model/gcn.yaml`
- Reload parity is covered by `tests/test_checkpoint_reload.py`

Targets:

- Explainer-validation list exists
- Explainer-test list exists
- Validation and test target ids have no overlap
- Target ordering and duplicate checks are covered by tests

Evaluation:

- Edge AUROC implemented and tested
- Edge AP implemented and tested
- Fidelity diagnostics and Fid+/Fid- implemented and tested
- Sparsity/budget implemented and tested
- Runtime reporting checked
- Jaccard stability implemented and tested

Baseline:

- Random Explanation Baseline exists
- Random stability exists
- Random uses the same 20% budget and candidate-edge scope

## 9. Known Limitations

- This benchmark currently uses BA-House synthetic data only.
- Results are for the frozen GCN checkpoint only.
- GNNExplainer uses the current PyG GNNExplainer configuration with `epochs=5`.
- Stability measures seed sensitivity only. It does not test robustness to graph or feature perturbation.
- Fidelity legacy is a probability delta and can be small or negative even when discrete Fid+ indicates a changed prediction.
- Random baseline is a sanity/reference baseline, not a learned explainer.
- PGExplainer, Integrated Gradients, GraphSAGE, and Elliptic experiments are not included in this Phase 4 freeze.

## 10. Exact Reproduce Commands

Reproduction should use the recorded dataset checksum, checkpoint checksum, Git commit, environment versions, and frozen Phase 4 evaluation config. The metric values in this report should be compared against the frozen outputs listed above rather than regenerated under a different commit or environment.

Environment check:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

Generate BA-House dataset if the artifact must be recreated:

```powershell
.\.venv\Scripts\python.exe scripts\generate_ba_house.py
```

Train GCN checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\train_gcn.py
```

Evaluate GCN checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_gcn.py
```

Run single-node explanation:

```powershell
.\.venv\Scripts\python.exe scripts\explain_one_node.py --node-id 465 --device auto
```

Run frozen Phase 4C GNNExplainer batch:

```powershell
.\.venv\Scripts\python.exe scripts\run_batch_gnnexplainer.py `
  --target-file results\metrics\explainer_test_targets.csv `
  --output-root results\explanations\frozen_phase4c_v2 `
  --metrics-output-dir results\metrics\frozen_phase4c_v2 `
  --raw-reuse-root results\explanations `
  --seed 42 `
  --device auto
```

Run GNNExplainer stability:

```powershell
.\.venv\Scripts\python.exe scripts\run_stability.py --device auto
```

Run Random baseline:

```powershell
.\.venv\Scripts\python.exe scripts\run_random_baseline.py --device auto
```

Compute Random Jaccard stability from existing Random selected-edge artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\compute_random_stability.py
```

Run validation checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
git diff --check
```

## 11. Handoff Gate Sang PGExplainer

Phase 4 is ready for review before PGExplainer because:

- Evaluation interface is shared and no longer depends on a GNNExplainer-only metric definition.
- Core metrics have tests.
- Target lists are frozen.
- Random sanity baseline exists.
- GNNExplainer and Random stability reports exist.
- This Phase 4 report is complete.

PGExplainer work should start only after this report and the current Phase 4 artifacts are reviewed.
