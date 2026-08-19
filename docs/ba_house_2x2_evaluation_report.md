# BA-House 2x2 Evaluation Report

## Scope And Freeze Status

This report freezes the BA-House evidence layer for the 2x2 matrix of backbone and explainer methods. It reuses existing frozen outputs only and does not rerun GCN, GraphSAGE, GNNExplainer, PGExplainer, Random baseline, stability, or any final-test evaluation.

BA-House only is covered here. ShapeGGen starts in a later task and is not started by this report.

Frozen target file: `configs/targets/ba_house_2x2_targets.csv`

- Target count: `12`
- Target SHA256: `2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f`
- Target IDs: `306, 315, 316, 355, 365, 375, 376, 411, 445, 465, 486, 491`
- Hash verified against Task19 manifest: `True`
- All four primary cells use the same node IDs: `True`

## Interpretation Hierarchy

Layer A is the primary comparison: compare explainers while holding the backbone fixed. The two primary comparisons are GCN with GNNExplainer vs PGExplainer, and GraphSAGE with GNNExplainer vs PGExplainer.

Layer B is a secondary pattern comparison across backbone contexts. It asks whether the relative ordering or trade-off between explainers is reproduced under GCN and GraphSAGE contexts. This report does not make a causal claim about the backbone.

Layer C is the controlled PG common-pool cross-backbone sensitivity from Task19B. It uses common 48/16 PG train/validation identities and reduces the explainer-training-pool membership confound, while the two backbones still differ in architecture, representation quality, predictive competence, and optimization history.

Layer D is GraphSAGE PG training-pool sensitivity: natural 106/35 vs common 48/16. Task19B is post-hoc controlled sensitivity and does not replace the original Task19 natural-pool primary result.

Task19A audit conclusions are preserved: no leakage or metric shortcut was accepted as the explanation for the GraphSAGE PGExplainer perfect natural-pool result, but the result must be interpreted under the frozen protocol and with explicit training-pool sensitivity context.

## A. Primary 2x2 Ground-Truth Agreement

| Backbone | Explainer | Edge AUROC mean | Edge AUROC std | Edge AUROC median | Edge AP mean | Edge AP std | Edge AP median | n_nodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | GNNExplainer | 0.870617 | 0.077797 | 0.892918 | 0.527731 | 0.236662 | 0.584178 | 12 |
| GCN | PGExplainer | 0.747969 | 0.188227 | 0.843335 | 0.366563 | 0.253247 | 0.434493 | 12 |
| GraphSAGE | GNNExplainer | 0.895612 | 0.083495 | 0.916028 | 0.594340 | 0.229296 | 0.642899 | 12 |
| GraphSAGE | PGExplainer natural106/35 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 12 |

## B. Primary Faithfulness

| Backbone | Explainer | Fid+ | Fid- | 1-Fid- | remove_delta | retain_delta | retain_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | GNNExplainer | 0.666667 | 0.250000 | 0.750000 | -0.000357 | -0.007996 | 1.029282 |
| GCN | PGExplainer | 0.416667 | 0.583333 | 0.416667 | -0.001014 | 0.000245 | 0.999108 |
| GraphSAGE | GNNExplainer | 0.916667 | 0.500000 | 0.500000 | 0.412075 | 0.213959 | 0.611798 |
| GraphSAGE | PGExplainer natural106/35 | 1.000000 | 0.500000 | 0.500000 | 0.425826 | 0.174690 | 0.688363 |

## C. Stability

Required note: same top-20% Jaccard metric, method-specific stochastic source.

| Backbone | Explainer | n_nodes | Jaccard mean | Jaccard std | Jaccard median | Jaccard min | Jaccard max | stochastic source | seeds/pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | GNNExplainer | 12 | 0.441272 | 0.279181 | 0.368264 | 0.159524 | 0.977273 | GNNExplainer independent optimization seeds | 5 seeds, 10 pairs/node |
| GCN | PGExplainer | 12 | 0.951911 | 0.047503 | 0.964542 | 0.866667 | 1.000000 | independently trained PGExplainer checkpoints | 5 seeds, 10 pairs/node |
| GraphSAGE | GNNExplainer | 12 | 0.479613 | 0.258427 | 0.421404 | 0.200000 | 0.977273 | GNNExplainer independent optimization seeds | 5 seeds, 10 pairs/node |
| GraphSAGE | PGExplainer natural106/35 | 12 | 0.805056 | 0.118526 | 0.830964 | 0.644444 | 1.000000 | independently trained PGExplainer checkpoints | 5 seeds, 10 pairs/node |

## D. Efficiency And Runtime

Runtime is decomposed and should not be collapsed into one runtime winner. GNNExplainer reports per-target optimization/generation time. PGExplainer has upfront explainer training plus per-target inference. Random reports random edge selection plus evaluation runtime.

| Backbone | Explainer | upfront training | per-target generation/inference | evaluation runtime | semantics |
| --- | --- | --- | --- | --- | --- |
| GCN | GNNExplainer | none | 0.129505 | not separately recorded in primary seed42 summary | per-target GNNExplainer optimization/generation time |
| GCN | PGExplainer | 22.389041 s explainer training, seed42 | 0.005550 | not separately recorded in primary seed42 summary | PGExplainer has upfront explainer training then per-target inference |
| GraphSAGE | GNNExplainer | none | 0.129616 | not separately recorded in primary seed42 summary | per-target GNNExplainer optimization/generation time |
| GraphSAGE | PGExplainer natural106/35 | 44.858333 s explainer training, seed42 | 0.014336 | not separately recorded in primary seed42 summary | PGExplainer natural backbone-specific train/val pool: 106/35 |

## E. Per-Node Paired Matrix

The full per-node paired matrix is stored at `results/metrics/ba_house_2x2/per_node.csv`. It contains one row per frozen target and all four primary model-explainer cells keyed by `node_id`.

Descriptive lowest average agreement nodes across the four primary cells:

| node_id | candidate edges | positive GT edges | GT prevalence | mean Edge AUROC | mean Edge AP | mean Fid+ |
| --- | --- | --- | --- | --- | --- | --- |
| 315.000000 | 866.000000 | 12.000000 | 0.013857 | 0.934231 | 0.339490 | 1.000000 |
| 465.000000 | 434.000000 | 12.000000 | 0.027650 | 0.936513 | 0.418708 | 1.000000 |
| 376.000000 | 538.000000 | 12.000000 | 0.022305 | 0.936391 | 0.432536 | 1.000000 |
| 306.000000 | 330.000000 | 12.000000 | 0.036364 | 0.921122 | 0.456443 | 0.750000 |
| 411.000000 | 176.000000 | 12.000000 | 0.068182 | 0.920605 | 0.542444 | 0.750000 |

These are descriptive diagnostics only. No target was added or dropped because of explanation quality.

## F. Controlled PG Common-Pool Cross-Backbone Sensitivity

This is Task19B post-hoc controlled sensitivity. The GCN PG common 48/16 result reuses the original frozen GCN PG checkpoint/results because the common pool equals the original GCN PG pool by set inclusion. The GraphSAGE PG common 48/16 result is the Task19B controlled run.

| Metric | PG+GCN common48/16 | PG+GraphSAGE common48/16 | delta |
| --- | --- | --- | --- |
| edge_auroc | 0.747969 | 0.999869 | 0.251900 |
| edge_ap | 0.366563 | 0.993590 | 0.627027 |
| fid_plus | 0.416667 | 1.000000 | 0.583333 |
| fid_minus | 0.583333 | 0.500000 | -0.083333 |
| one_minus_fid_minus | 0.416667 | 0.500000 | 0.083333 |
| stability | 0.951911 | 0.789956 | -0.161955 |
| sparsity | 0.793694 | 0.793694 | 0.000000 |
| inference_runtime_seconds | 0.005550 | 0.003302 | -0.002248 |

## G. GraphSAGE PG Training-Pool Sensitivity

This is the cleanest BA-House check of PG explainer-training pool membership/size sensitivity on a fixed GraphSAGE backbone. The original Task19 natural-pool result remains the primary GraphSAGE PGExplainer result.

| Metric | natural106/35 | common48/16 | delta |
| --- | --- | --- | --- |
| edge_auroc | 1.000000 | 0.999869 | -0.000131 |
| edge_ap | 1.000000 | 0.993590 | -0.006410 |
| fid_plus | 1.000000 | 1.000000 | 0.000000 |
| fid_minus | 0.500000 | 0.500000 | 0.000000 |
| one_minus_fid_minus | 0.500000 | 0.500000 | 0.000000 |
| stability | 0.805056 | 0.789956 | -0.015100 |
| sparsity | 0.793694 | 0.793694 | 0.000000 |
| inference_runtime_seconds | 0.014336 | 0.003302 | -0.011034 |

## H. Random Sanity Baseline

Random stays outside the 2x2 matrix. Raw Random output has 12 nodes x 10 seeds = 120 rows. The primary Random quality summary uses node-level sampling units: first average within each node across seeds, then summarize the 12 node-level values across nodes.

Random node-level summary file: `results/metrics/ba_house_2x2/random_node_level_summary.csv`

| metric | n | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| edge_auroc | 12 | 0.496727 | 0.026778 | 0.491445 | 0.455952 | 0.547132 |
| edge_ap | 12 | 0.277178 | 0.232547 | 0.259364 | 0.017968 | 0.589914 |
| fid_plus | 12 | 0.333333 | 0.130268 | 0.300000 | 0.100000 | 0.500000 |
| fid_minus | 12 | 0.625000 | 0.128806 | 0.650000 | 0.300000 | 0.800000 |
| one_minus_fid_minus | 12 | 0.375000 | 0.128806 | 0.350000 | 0.200000 | 0.700000 |
| remove_probability_delta | 12 | -0.000994 | 0.002396 | -0.001423 | -0.004110 | 0.003292 |
| retain_probability_delta | 12 | -0.005173 | 0.001405 | -0.005128 | -0.009043 | -0.003187 |
| retain_probability_ratio | 12 | 1.018939 | 0.005123 | 1.018748 | 1.011607 | 1.033008 |
| sparsity | 12 | 0.793694 | 0.008901 | 0.797033 | 0.769231 | 0.800000 |
| runtime_seconds | 12 | 0.021375 | 0.006899 | 0.019086 | 0.018473 | 0.042955 |
| evaluation_runtime_seconds | 12 | 0.021375 | 0.006899 | 0.019086 | 0.018473 | 0.042955 |

## Analysis Answers

1. With GCN fixed, GNNExplainer has higher Edge AUROC, Edge AP, Fid+, and 1-Fid- than PGExplainer, while PGExplainer has much higher Jaccard stability.
2. With GraphSAGE fixed, PGExplainer natural-pool has perfect Edge AUROC/AP and higher stability than GNNExplainer. Both methods have strong Fid+, while Fid- and 1-Fid- are tied at the mean level.
3. The relative explainer ranking differed across backbone contexts under the frozen BA-House protocol. This is a pattern comparison, not a causal claim about the backbone.
4. Under common PG 48/16 pools, GraphSAGE PGExplainer still shows a different PG profile from GCN PGExplainer, with higher AUROC/AP and Fid+ but lower stability.
5. On GraphSAGE, natural 106/35 vs common 48/16 changes AUROC by -0.000131, AP by -0.006410, Fid+ by 0.000000, 1-Fid- by 0.000000, and stability by -0.015100.
6. GT agreement and faithfulness are not perfectly aligned per node. The paired per-node matrix is the source of truth for these descriptive checks.
7. PGExplainer is more stable than GNNExplainer in both backbone contexts. Random has the lowest stability.
8. The observed trade-off is quality versus stability and runtime semantics: PGExplainer pays upfront training and then fast inference, while GNNExplainer optimizes per target.
9. Difficult-node diagnostics are descriptive only. Candidate size and GT prevalence are recorded per node for inspection, but no significance or causal claim is made.

## Output Files

- `results/metrics/ba_house_2x2/main.csv`
- `results/metrics/ba_house_2x2/per_node.csv`
- `results/metrics/ba_house_2x2/paired_deltas.csv`
- `results/metrics/ba_house_2x2/runtime.csv`
- `results/metrics/ba_house_2x2/stability.csv`
- `results/metrics/ba_house_2x2/random_node_level_summary.csv`
- `results/metrics/ba_house_2x2/controlled_pg_common_pool.csv`
- `results/metrics/ba_house_2x2/pg_pool_sensitivity.csv`
- `results/metrics/ba_house_2x2/manifest.json`
- `docs/ba_house_2x2_freeze.json`

## Acceptance Checklist

- 12-target hash verified: `True`
- All four primary cells use same node IDs: `True`
- No frozen experiment rerun: `True`
- Task19A audit conclusions referenced: `True`
- Task19B common-pool analysis included: `True`
- Original Task19 natural-pool result preserved: `True`
- Task19B labeled post-hoc sensitivity: `True`
- Primary within-backbone comparisons identified: `True`
- Controlled PG cross-backbone result separated: `True`
- GraphSAGE natural-vs-common PG sensitivity separated: `True`
- Random primary aggregation uses 12 node-level summaries: `True`
- Random raw seed rows preserved separately: `True`
- Stability stochastic source documented per method: `True`
- Runtime semantics decomposed: `True`
- Report says BA-House only: `True`

## Review Gate

Stop for review after this Task20 freeze package. Do not start ShapeGGen automatically.
