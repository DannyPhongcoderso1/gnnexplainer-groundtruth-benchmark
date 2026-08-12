# Phase 4C Output Schema

## Frozen Budget

```text
budget_protocol = fixed_candidate_edge_ratio
selected_edge_ratio = 0.2
rounding_rule = ceil(candidate_edge_count * selected_edge_ratio), minimum 1
```

## Ranking Metrics

Edge AUROC is the primary synthetic ground-truth ranking metric aligned with
synthetic GNN explainer evaluation. Edge AP is reported as an additional metric
for imbalanced edge labels.

Both metrics are computed from continuous scores over the full candidate-edge
set, not only the selected 20% explanation edges.

## Fid+ / Fid-

`fid_plus`, `fid_minus`, and `one_minus_fid_minus` are model-focused
prediction-change metrics following GraphFramEx / PyTorch Geometric semantics.

```text
fid_plus = 1 if remove_prediction != original_prediction else 0
fid_minus = 1 if retain_prediction != original_prediction else 0
one_minus_fid_minus = 1 - fid_minus
```

Higher Fid+ indicates stronger necessity. Lower Fid- and higher 1-Fid- indicate
stronger sufficiency.

## Probability Diagnostics

The continuous remove/retain probability deltas are confidence-change
diagnostics, not standardized Fid+/Fid- metrics.

```text
remove_probability_delta = original_probability - remove_probability
retain_probability_delta = original_probability - retain_probability
fidelity_legacy = remove_probability_delta
```

All Phase 4C v2 per-node outputs must satisfy:

```text
abs(fidelity_legacy - remove_probability_delta) < 1e-12
```

Do not describe `fidelity_legacy` as GraphFramEx Fidelity+.

## Runtime Fields

Runtime fields are reported separately:

```text
generation_runtime_seconds = time spent producing the raw explanation mask
evaluation_runtime_seconds = time spent recomputing metrics from an existing raw mask, when measured
```

For reused raw masks, `generation_runtime_seconds` is recovered from source
metadata when available. If it cannot be recovered, it must be left null rather
than reported as `0.0`.
