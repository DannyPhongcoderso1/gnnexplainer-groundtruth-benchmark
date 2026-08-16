# GCN Explainer Three-Way Comparison Freeze

## Scope

This report compares Random Explanation, GNNExplainer, and PGExplainer on the
same frozen GCN backbone and the same 12 frozen BA-House final targets. It
reuses frozen outputs only and does not rerun, tune, or replace any experiment.

## Inputs And Protocol

- Backbone: frozen GCN.
- Target join key: `node_id`.
- Candidate-edge scope: computation subgraph.
- Edge AUROC/AP: continuous scores over the full candidate-edge set.
- Explanation budget: fixed candidate-edge ratio `0.2`.
- Sparsity is primarily a protocol confirmation because the selected ratio is fixed.
- PGExplainer primary quality uses seed `42`, not a best seed from stability.

## Fairness Checks

All methods match on candidate-edge count, positive/negative GT counts,
selected-edge count, original prediction, and selected ratio for `12/12`
targets. Random uses 10 frozen random seeds. GNNExplainer and PGExplainer
stability use 5 frozen seeds.

## Summary

Values in the main table are means over the frozen evaluation rows. The full
`summary.csv` also records standard deviation and median columns. The
`per_node.csv` and `paired_deltas.csv` files retain node-level values,
remove/retain diagnostics, and paired differences.

| Method | AUROC | AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability | Runtime semantics |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Random | 0.496727 | 0.277178 | 0.333333 | 0.625000 | 0.375000 | 0.793694 | 0.126536 | random edge selection plus metric evaluation runtime |
| GNNExplainer | 0.870617 | 0.527731 | 0.666667 | 0.250000 | 0.750000 | 0.793694 | 0.441272 | per-target optimization/generation runtime |
| PGExplainer | 0.747969 | 0.366563 | 0.416667 | 0.583333 | 0.416667 | 0.793694 | 0.951911 | upfront training plus trained-explainer inference per target |

## Runtime

Runtime is decomposed rather than collapsed into a single winner column.
GNNExplainer runtime is per-target optimization/generation. PGExplainer has an
upfront training cost and then cheap trained-explainer inference. Random runtime
is random edge selection plus metric evaluation.

## Interpretation

Observed metrics show trade-offs across ground-truth agreement, faithfulness,
stability, and efficiency. These are descriptive results on 12 targets, so the
comparison should not be treated as a universal superiority claim or a formal
statistical significance result.

PGExplainer final-test results have already been exposed. They must not be used
to tune epochs, learning rate, regularization, temperature, bias, target
membership, checkpoint selection, or seed choice.

## Artifacts

- `results/metrics/gcn_explainer_comparison/summary.csv`
- `results/metrics/gcn_explainer_comparison/per_node.csv`
- `results/metrics/gcn_explainer_comparison/paired_deltas.csv`
- `results/metrics/gcn_explainer_comparison/runtime.csv`
- `results/metrics/gcn_explainer_comparison/manifest.json`
- `docs/gcn_explainer_comparison_freeze.json`
