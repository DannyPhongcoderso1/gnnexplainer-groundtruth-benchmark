# Elliptic GNNExplainer Report

## Scope

Task27 applies GNNExplainer to the frozen Elliptic GCN and GraphSAGE explanation-compatible backbones. The same Task26 common final target file is used for both backbones. No Edge AUROC/AP explanation metrics are computed because Elliptic has no ground-truth explanatory edge masks in this thesis.

## Frozen Protocol

- Target file: `configs/targets/elliptic_xai_common_test_targets.csv`
- Target SHA256: `e414941e6943b06484ecd5028c51daf80ff2bf9f1bbb06e6a5fd816de8dfb9c0`
- Target count: `200`
- Primary seed: `42`
- Stability seeds: `[0, 1, 2, 3, 4]`
- Selected edge ratio: `0.2`
- Rounding rule: `max(1, ceil(candidate_edge_count * 0.20))`
- Candidate edge scope: `2-hop computation subgraph around target node`
- Explanation type: `phenomenon`

## Main Metrics

| Backbone | Fid+ | Fid- | 1-Fid- | Remove delta | Retain probability | Retain delta | Retain ratio | Sparsity | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 0.055000 | 0.000000 | 1.000000 | 0.045679 | 0.999209 | -0.012138 | 1.017160 | 0.775332 | 0.501168 |
| GraphSAGE | 0.035000 | 0.005000 | 0.995000 | 0.044117 | 0.990821 | -0.001415 | 1.003377 | 0.775332 | 0.574761 |

## Runtime Semantics

GNNExplainer runtime is per-instance optimization/generation time. Metric evaluation time is recorded separately as evaluation runtime. These are not PGExplainer training or inference runtimes.

## Integrity

- Original wrapper predictions are checked against frozen Task26 predictions.
- GCN uses the Task26 `GCNConv(cached=False)` compatible wrapper.
- GraphSAGE uses the Task26 compatible wrapper.
- Model weights are checked for immutability during each target run.
- Remove/retain perturbations must alter the edge input.
- Failures are written explicitly and do not silently drop targets.

## Cross-Backbone Interpretation Boundary

The GCN and GraphSAGE rows are descriptive backbone-context observations on the same frozen target nodes. They must not be interpreted as causal backbone effects. Primary explainer comparison is deferred to later within-backbone tasks.

## Outputs

- `results/explanations/elliptic/gcn/gnnexplainer/`
- `results/explanations/elliptic/graphsage/gnnexplainer/`
- `results/metrics/elliptic/gcn/gnnexplainer/`
- `results/metrics/elliptic/graphsage/gnnexplainer/`
- `docs/elliptic_gnnexplainer_freeze.json`

## Stop Boundary

STOP FOR REVIEW. Task28 and later tasks are not started here.
