# Elliptic XAI Protocol

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

The selected-edge budget remains `selected_edge_ratio = 0.2` with `ceil(candidate_edge_count * selected_edge_ratio), minimum 1`, unless a later pre-final reviewed amendment changes it before any final Elliptic explanation run.

## Frozen Targets And PG Pools

- Final common test targets: `200`
- Common PG train: `2000`
- Common PG validation: `500`
- Selection rule: known-labeled temporal split nodes with positive 2-hop candidate-edge count; deterministic seeded sample if count exceeds cap
- Sampling seed: `42`
- Sampling does not use explanation results.

Target files are recorded in `docs/elliptic_targets_freeze.json`.

## Stop Boundary

No final Elliptic explanation may run before review.
