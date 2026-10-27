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
- Primary comparison unit: one node-level observation per `node_id`.

## Fairness Checks

All methods match on candidate-edge count, positive/negative GT counts,
selected-edge count, original prediction, and selected ratio for `12/12`
targets. Random uses 10 frozen random seeds. GNNExplainer and PGExplainer
stability use 5 frozen seeds.

## Summary

Values in the main table are means over the frozen evaluation rows. The full
`summary.csv` also records standard deviation and median columns. Random is
first averaged within each `node_id` across its 10 frozen random seeds, then
summarized across the same 12 node-level observations used by GNNExplainer and
PGExplainer. The original 120 Random rows remain preserved in the frozen Random
artifact and the aggregation audit is saved in
`random_aggregation_audit.csv`.

| Method | AUROC | AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability | Runtime semantics |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Random | 0.496727 | 0.277178 | 0.333333 | 0.625000 | 0.375000 | 0.793694 | 0.126536 | random edge selection plus metric evaluation runtime |
| GNNExplainer | 0.870617 | 0.527731 | 0.666667 | 0.250000 | 0.750000 | 0.793694 | 0.441272 | per-target optimization/generation runtime |
| PGExplainer | 0.747969 | 0.366563 | 0.416667 | 0.583333 | 0.416667 | 0.793694 | 0.951911 | upfront training plus trained-explainer inference per target |

## Random Aggregation Audit

Random has 120 frozen seed-level rows: 12 nodes times 10 seeds. For the main
comparison it is reduced to 12 node-level values by taking the mean over the 10
seeds within each node. Because every node has exactly 10 Random seeds, AUROC
and AP means are unchanged, while standard deviation and median reflect the
node-level sampling unit.

| Metric | Raw n | Raw mean | Raw std | Raw median | Node n | Node mean | Node std | Node median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| edge_auroc | 120 | 0.496727 | 0.102063 | 0.500951 | 12 | 0.496727 | 0.026778 | 0.491445 |
| edge_ap | 120 | 0.277178 | 0.233067 | 0.216445 | 12 | 0.277178 | 0.232547 | 0.259364 |

## Stability Protocol

All methods use the same pairwise Jaccard metric over top-20% selected
candidate-edge sets: `Jaccard(A, B) = |A intersection B| / |A union B|`. The
stochastic source follows each method's frozen protocol: Random uses repeated
random edge selections, GNNExplainer uses independent explanation optimization
runs, and PGExplainer uses independently trained checkpoints.

| Method | Seeds/checkpoints | Pairs per node | Stochastic source |
|---|---:|---:|---|
| Random | 10 | 45 | repeated random edge selections |
| GNNExplainer | 5 | 10 | independent explanation optimization runs |
| PGExplainer | 5 | 10 | independently trained PGExplainer checkpoints |

## Runtime

Runtime is decomposed rather than collapsed into a single winner column.
GNNExplainer runtime is per-target optimization/generation. PGExplainer has an
upfront training cost and then cheap trained-explainer inference. Random runtime
is random edge selection plus metric evaluation. Runtime boundaries differ by
method and therefore are interpreted as computational-behavior descriptors
rather than perfectly identical wall-clock workloads.

| Method | Upfront training | Per-target generation/inference | Other eval cost | Semantics |
|---|---:|---:|---:|---|
| Random | N/A | N/A | 0.021375 s | random edge selection plus metric evaluation runtime |
| GNNExplainer | N/A | 0.129505 s | N/A | per-target optimization/generation runtime |
| PGExplainer | 22.389041 s | 0.005550 s | 0.270408 s | upfront training plus trained-explainer inference per target |

The PGExplainer primary seed42 training runtime is kept separate from
Task16B independent stability-checkpoint training runtime. Stability-checkpoint
timings are not averaged into the primary seed42 quality runtime.

## Sparsity

Sparsity is largely determined by the fixed 20% candidate-edge budget and is
therefore a protocol confirmation rather than an independent method-quality
discriminator.

## Interpretation

On frozen GCN/BA-House, GNNExplainer achieves the strongest observed
ground-truth agreement and faithfulness, while PGExplainer exhibits the highest
observed top-20% Jaccard stability and low post-training per-target inference
cost. These descriptive results show a trade-off between explanation quality,
reproducibility, and computational behavior, not one method dominating every
evaluation dimension.

PGExplainer final-test results have already been exposed. They must not be used
to tune epochs, learning rate, regularization, temperature, bias, target
membership, checkpoint selection, or seed choice.

## Artifacts

- `results/metrics/gcn_explainer_comparison/summary.csv`
- `results/metrics/gcn_explainer_comparison/per_node.csv`
- `results/metrics/gcn_explainer_comparison/paired_deltas.csv`
- `results/metrics/gcn_explainer_comparison/runtime.csv`
- `results/metrics/gcn_explainer_comparison/random_aggregation_audit.csv`
- `results/metrics/gcn_explainer_comparison/manifest.json`
- `docs/gcn_explainer_comparison_freeze.json`
