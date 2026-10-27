# Elliptic Model Parity Report

## GCN Original-Graph Parity

Baseline GCN uses `GCNConv(cached=True)`. Task26 instantiates the same architecture with `cached=False`, loads identical weights, sets eval mode, and compares original-graph logits/probabilities/predictions before any XAI run.

| Check | Value |
|---|---:|
| tolerance | `1e-06` |
| max_abs_logit_diff | `0.0` |
| mean_abs_logit_diff | `0.0` |
| prediction_mismatch_count | `0` |
| prediction_count | `203769` |
| pass | `True` |

## GraphSAGE Original-Graph Parity

| Check | Value |
|---|---:|
| max_abs_logit_diff | `0.0` |
| mean_abs_logit_diff | `0.0` |
| prediction_mismatch_count | `0` |
| prediction_count | `203769` |
| pass | `True` |

## Probability Parity

| Check | Value |
|---|---:|
| GCN max_abs_probability_diff | `0.0` |
| GraphSAGE max_abs_probability_diff | `0.0` |

## Perturbation Smoke

The smoke test removes a deterministic 5% edge subset and verifies that the explanation-compatible models receive changed edge input.

| Model | max_abs_test_logit_diff | mean_abs_test_logit_diff | prediction_change_count | pass |
|---|---:|---:|---:|---:|
| GCN cached=False | `114.30775451660156` | `0.38554415106773376` | `39` | `True` |
| GraphSAGE wrapper | `58.29523468017578` | `0.217808797955513` | `11` | `True` |

Original edge count: `468710`. Perturbed edge count: `445274`.
