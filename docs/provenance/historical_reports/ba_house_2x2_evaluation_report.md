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

Primary cell completeness:

| Backbone | Explainer | success_count |
| --- | --- | --- |
| GCN | GNNExplainer | 12/12 |
| GCN | PGExplainer | 12/12 |
| GraphSAGE | GNNExplainer | 12/12 |
| GraphSAGE | PGExplainer (natural pool 106/35) | 12/12 |

No final target was silently dropped because of explanation quality.

## Backbone Predictive-Performance Context

| Backbone | Validation accuracy | Test accuracy | Motif-test correct | Source |
| --- | --- | --- | --- | --- |
| GCN | 0.650000 | 0.620000 | 16/40 | `results/metrics/model_quality_report.json` |
| GraphSAGE | 0.950000 | 0.960000 | 36/40 | `results/metrics/graphsage_ba_house_training/summary.json` |

The backbones differ substantially in predictive competence on BA-House. Therefore, absolute cross-backbone explanation-metric differences must not be interpreted as isolated causal effects of the backbone alone.

## Interpretation Hierarchy

Layer A is the primary comparison: compare explainers while holding the backbone fixed. The two primary comparisons are GCN with GNNExplainer vs PGExplainer, and GraphSAGE with GNNExplainer vs PGExplainer.

Layer B is a secondary pattern comparison across backbone contexts. It asks whether the relative ordering or trade-off between explainers is reproduced under GCN and GraphSAGE contexts. This report does not make a causal claim about the backbone.

Layer C is the controlled PG common-pool cross-backbone sensitivity from Task19B. It uses common 48/16 PG train/validation identities and reduces the explainer-training-pool membership confound, while the two backbones still differ in architecture, representation quality, predictive competence, and optimization history.

Layer D is GraphSAGE PG training-pool sensitivity: natural 106/35 vs common 48/16. Task19B is post-hoc controlled sensitivity and does not replace the original Task19 natural-pool primary result.

Task19A audit status: PASS. No leakage or metric shortcut was accepted as the explanation for the GraphSAGE PGExplainer perfect Edge AUROC/AP natural-pool result, but the result must be interpreted under the frozen protocol and with explicit training-pool sensitivity context.

Task19B status: post-hoc controlled sensitivity. It does not replace Task19, and the original natural-pool results remain the primary historical GraphSAGE PGExplainer experiment.

## A. Primary 2x2 Ground-Truth Agreement

### Interpretation of Edge AUROC and Edge AP

Edge AUROC measures how well continuous explainer edge scores rank ground-truth explanatory edges above non-ground-truth candidate edges. Higher values indicate better ranking agreement with the synthetic ground-truth edge mask. A value of `1.0` indicates perfect ranking, while a value around `0.5` is random-ranking level.

Edge AP summarizes precision-recall behavior of the same continuous edge ranking. It is especially informative when ground-truth-positive explanation edges are sparse relative to the candidate-edge set. Higher is better, `1.0` indicates perfect ranking, and random AP depends on positive-edge prevalence rather than being fixed at `0.5`.

Both Edge AUROC and Edge AP are computed from continuous edge-importance scores over the full candidate-edge set. They are not computed only on the selected top-20% edge set. The top-20% selected set is used separately for Fid+, Fid-, 1-Fid-, sparsity, and Jaccard stability.

BA-House target context:

| Quantity | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- |
| candidate_edge_count | 221.166667 | 89.000000 | 24 | 866 |
| positive_gt_edges | 12.000000 | 12.000000 | 12 | 12 |
| ground_truth_edge_prevalence | 0.241228 | 0.218137 | 0.013857 | 0.500000 |

This context helps explain why Edge AP can differ substantially from Edge AUROC. AUROC has an approximate random-ranking level of `0.5`, while random AP depends on the prevalence of ground-truth-positive candidate edges.

| Backbone | Explainer | Edge AUROC mean | Edge AUROC std | Edge AUROC median | Edge AP mean | Edge AP std | Edge AP median | n_nodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | GNNExplainer | 0.870617 | 0.077797 | 0.892918 | 0.527731 | 0.236662 | 0.584178 | 12 |
| GCN | PGExplainer | 0.747969 | 0.188227 | 0.843335 | 0.366563 | 0.253247 | 0.434493 | 12 |
| GraphSAGE | GNNExplainer | 0.895612 | 0.083495 | 0.916028 | 0.594340 | 0.229296 | 0.642899 | 12 |
| GraphSAGE | PGExplainer (natural pool 106/35) | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 12 |

Within the frozen GCN backbone, GNNExplainer has higher ground-truth agreement than PGExplainer. The Edge AUROC delta is `0.870617 - 0.747969 = +0.122648`, and the Edge AP delta is `0.527731 - 0.366563 = +0.161168`. On BA-House with the GCN backbone fixed, GNNExplainer achieved higher ground-truth edge-ranking agreement than PGExplainer according to both Edge AUROC and Edge AP.

Within the frozen GraphSAGE backbone, PGExplainer has higher ground-truth agreement than GNNExplainer. The Edge AUROC delta is `1.000000 - 0.895612 = +0.104388`, and the Edge AP delta is `1.000000 - 0.594340 = +0.405660`. On BA-House with GraphSAGE fixed, PGExplainer achieved higher ground-truth edge-ranking agreement than GNNExplainer, reaching Edge AUROC and Edge AP of `1.0` on the frozen 12-node evaluation set. The perfect values are specific to the frozen BA-House protocol and target set and should not be interpreted as universal superiority.

The relative ordering of GNNExplainer and PGExplainer changes across the two backbone contexts. In the GCN context, GNNExplainer has higher AUROC/AP than PGExplainer. In the GraphSAGE context, PGExplainer has higher AUROC/AP than GNNExplainer. This indicates that explainer performance should be evaluated together with the model being explained rather than assuming a backbone-independent ranking.

Higher values were observed for GNNExplainer in the GraphSAGE context than in the GCN context under the frozen BA-House protocol. For PGExplainer, the primary table keeps the original frozen GraphSAGE natural-pool result with train `106`, validation `35`, AUROC `1.000000`, and AP `1.000000`.

The strong GraphSAGE PGExplainer ranking result persisted when PGExplainer for both backbone contexts used the same common 48 training-node and 16 validation-node identities, while final explanation evaluation remained on the same frozen 12-node target set. Under that common-pool setting, GCN PGExplainer had AUROC `0.747969` and AP `0.366563`, while GraphSAGE PGExplainer had AUROC `0.999869` and AP `0.993590`. This indicates that the original difference was not solely attributable to the larger GraphSAGE-specific explainer training pool.

Edge AUROC and Edge AP measure agreement with the known synthetic ground-truth explanation, whereas Fid+/Fid- assess the relationship between selected explanation edges and the behavior of the trained classifier. These metric families answer different questions and therefore should not be treated as interchangeable.

## B. Primary Faithfulness

Faithfulness definitions under the frozen implementation:

- Fid+ = `1` if removing the selected explanation edges changes the original predicted class.
- Fid- = `1` if retaining only the selected explanation edges changes the original predicted class.
- 1-Fid- = `1 - Fid-`.
- Fid+ higher is better for necessity under this protocol.
- Fid- lower is better.
- 1-Fid- higher is better for sufficiency under this protocol.

Fid+, Fid-, and 1-Fid- are per-target binary outcomes. The table reports their mean over the 12 frozen targets. Counts are shown as `changed_count/12` for Fid+ and Fid-.

Probability diagnostics:

- `original_probability` is the probability of the original predicted class on the original graph.
- `remove_probability_delta = original_probability - remove_probability`.
- `retain_probability_delta = original_probability - retain_probability`.
- `retain_probability_ratio = retain_probability / original_probability`.

These probability diagnostics are auxiliary continuous diagnostics and must not be mislabeled as standardized Fid+/Fid- metrics. Binary prediction-change metrics and probability deltas need not move identically, and their signs are not forced to agree.

| Backbone | Explainer | Fid+ mean | Fid+ count | Fid- mean | Fid- count | 1-Fid- mean | remove_probability_delta | retain_probability_delta | retain_probability_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | GNNExplainer | 0.666667 | 8/12 | 0.250000 | 3/12 | 0.750000 | -0.000357 | -0.007996 | 1.029282 |
| GCN | PGExplainer | 0.416667 | 5/12 | 0.583333 | 7/12 | 0.416667 | -0.001014 | 0.000245 | 0.999108 |
| GraphSAGE | GNNExplainer | 0.916667 | 11/12 | 0.500000 | 6/12 | 0.500000 | 0.412075 | 0.213959 | 0.611798 |
| GraphSAGE | PGExplainer (natural pool 106/35) | 1.000000 | 12/12 | 0.500000 | 6/12 | 0.500000 | 0.425826 | 0.174690 | 0.688363 |

## Budget And Sparsity

All four primary cells use the same fixed budget:

- `selected_edge_ratio = 0.20`
- `selected_edge_count = max(1, ceil(candidate_edge_count * 0.20))`

Primary-cell mean sparsity:

| Backbone | Explainer | mean_sparsity |
| --- | --- | --- |
| GCN | GNNExplainer | 0.793694 |
| GCN | PGExplainer | 0.793694 |
| GraphSAGE | GNNExplainer | 0.793694 |
| GraphSAGE | PGExplainer (natural pool 106/35) | 0.793694 |

Because the 20% budget is fixed, sparsity is primarily a protocol and fairness confirmation in this report. It should not be used to rank explainers here, and equal sparsity is not an algorithmic property of the explainers.

## C. Stability

Required note: same top-20% Jaccard metric, method-specific stochastic source.

The stability summaries use per-node mean Jaccard as the summary sampling unit. For learned explainers, `n_nodes = 12`, `pairs_per_node = 10`, and `n_pairs_total = 120` because five seeds produce ten pairwise comparisons per node. The pairwise Jaccard values are repeated measurements within nodes and are not independent target observations.

| Backbone | Explainer | n_nodes | pairs_per_node | n_pairs_total | summary_sampling_unit | Jaccard mean | Jaccard std | Jaccard median | Jaccard min | Jaccard max | stochastic source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | GNNExplainer | 12 | 10 | 120 | per_node_mean_jaccard | 0.441272 | 0.279181 | 0.368264 | 0.159524 | 0.977273 | GNNExplainer independent optimization seeds |
| GCN | PGExplainer | 12 | 10 | 120 | per_node_mean_jaccard | 0.951911 | 0.047503 | 0.964542 | 0.866667 | 1.000000 | independently trained PGExplainer checkpoints |
| GraphSAGE | GNNExplainer | 12 | 10 | 120 | per_node_mean_jaccard | 0.479613 | 0.258427 | 0.421404 | 0.200000 | 0.977273 | GNNExplainer independent optimization seeds |
| GraphSAGE | PGExplainer (natural pool 106/35) | 12 | 10 | 120 | per_node_mean_jaccard | 0.805056 | 0.118526 | 0.830964 | 0.644444 | 1.000000 | independently trained PGExplainer checkpoints |

## D. Efficiency And Runtime

Runtime is decomposed and should not be collapsed into one runtime winner. PGExplainer requires an upfront training stage but subsequently generates explanations through inexpensive inference, whereas GNNExplainer performs a separate optimization procedure for each target node. Random reports random edge selection plus evaluation runtime.

| Backbone | Explainer | upfront training | per-target generation/inference | evaluation runtime | semantics |
| --- | --- | --- | --- | --- | --- |
| GCN | GNNExplainer | none | 0.129505 | not separately recorded in primary seed42 summary | per-target GNNExplainer optimization/generation time |
| GCN | PGExplainer | 22.389041 s explainer training, seed42 | 0.005550 | not separately recorded in primary seed42 summary | PGExplainer has upfront explainer training then per-target inference |
| GraphSAGE | GNNExplainer | none | 0.129616 | not separately recorded in primary seed42 summary | per-target GNNExplainer optimization/generation time |
| GraphSAGE | PGExplainer natural106/35 | 44.858333 s explainer training, seed42 | 0.014336 | not separately recorded in primary seed42 summary | PGExplainer natural backbone-specific train/val pool: 106/35 |

## E. Per-Node Paired Matrix

The full per-node paired matrix is stored at `results/metrics/ba_house_2x2/per_node.csv`. It contains one row per frozen target and all four primary model-explainer cells keyed by `node_id`.

Five nodes with the lowest mean Edge AP across the four primary cells:

| node_id | candidate edges | positive GT edges | GT prevalence | mean Edge AUROC | mean Edge AP | mean Fid+ |
| --- | --- | --- | --- | --- | --- | --- |
| 315 | 866 | 12 | 0.013857 | 0.934231 | 0.339490 | 1.000000 |
| 465 | 434 | 12 | 0.027650 | 0.936513 | 0.418708 | 1.000000 |
| 376 | 538 | 12 | 0.022305 | 0.936391 | 0.432536 | 1.000000 |
| 306 | 330 | 12 | 0.036364 | 0.921122 | 0.456443 | 0.750000 |
| 411 | 176 | 12 | 0.068182 | 0.920605 | 0.542444 | 0.750000 |

These rows were selected by sorting the four-cell per-node table by mean Edge AP ascending, with mean Edge AUROC used only as a deterministic tie-breaker. This is not an arbitrary composite metric. These are descriptive diagnostics only. No target was added or dropped because of explanation quality.

## F. Controlled PG Common-Pool Cross-Backbone Sensitivity

This is Task19B post-hoc controlled sensitivity. The GCN PG common 48/16 result reuses the original frozen GCN PG checkpoint/results because the common pool equals the original GCN PG pool by set inclusion. The GraphSAGE PG common 48/16 result is the Task19B controlled run.

Delta direction: `delta = PGExplainer GraphSAGE common pool 48/16 - PGExplainer GCN common pool 48/16`.

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

Delta direction: `delta = PGExplainer common pool 48/16 - PGExplainer natural pool 106/35`.

| Metric | PGExplainer (natural pool 106/35) | PGExplainer (common pool 48/16) | delta |
| --- | --- | --- | --- |
| edge_auroc | 1.000000 | 0.999869 | -0.000131 |
| edge_ap | 1.000000 | 0.993590 | -0.006410 |
| fid_plus | 1.000000 | 1.000000 | 0.000000 |
| fid_minus | 0.500000 | 0.500000 | 0.000000 |
| one_minus_fid_minus | 0.500000 | 0.500000 | 0.000000 |
| stability | 0.805056 | 0.789956 | -0.015100 |
| sparsity | 0.793694 | 0.793694 | 0.000000 |
| inference_runtime_seconds | 0.014336 | 0.003302 | -0.011034 |

Seed42 upfront training runtimes were recorded with comparable semantics as PGExplainer explainer-training runtime. The larger natural pool can affect upfront training cost, so training and inference should not be collapsed.

| Runtime diagnostic | PGExplainer (natural pool 106/35) | PGExplainer (common pool 48/16) | delta |
| --- | --- | --- | --- |
| upfront_training_runtime_seconds | 44.858333 | 9.819772 | -35.038561 |

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

Random stability uses independent random edge selection as its stochastic source. The frozen result uses `10` seeds, `45` pairwise comparisons per node, `12` nodes, and `540` total pairwise comparisons. The summary sampling unit is again per-node mean Jaccard, not raw pairwise rows.

| Metric | n_nodes | seeds | pairs_per_node | n_pairs_total | summary_sampling_unit | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random mean Jaccard stability | 12 | 10 | 45 | 540 | per_node_mean_jaccard | 0.126536 | 0.017376 | 0.123461 | 0.108960 | 0.169899 |

## Statistical Discipline

- The final learned-explainer comparison uses 12 frozen target nodes.
- `node_id` is the primary sampling unit for main target-level comparison.
- Per-node rows are preserved in `results/metrics/ba_house_2x2/per_node.csv`.
- Paired deltas use identical node IDs.
- Random repeated seeds are first aggregated within node for main Random quality summaries.
- Stability pairwise values are repeated measurements within nodes and are not independent target observations.
- No statistical significance claim is made by default.
- No seed-level pseudo-replication is used for primary summaries.
- No arbitrary composite score is introduced.

## Analysis Answers

1. Finding 1, GCN context: on BA-House with the frozen GCN backbone fixed, GNNExplainer has stronger ground-truth agreement and stronger Fid+/1-Fid- than PGExplainer, while PGExplainer has substantially higher mean Jaccard stability. This is a protocol-specific within-backbone observation.
2. Finding 2, GraphSAGE context: on BA-House with the frozen GraphSAGE backbone fixed, PGExplainer has stronger ground-truth agreement than GNNExplainer and both methods have strong Fid+. PGExplainer also shows higher mean Jaccard stability in this backbone context.
3. Finding 3, backbone-context dependence: the relative ground-truth agreement ranking between GNNExplainer and PGExplainer differs between GCN and GraphSAGE. This is BA-House-specific and protocol-specific, and it is not a causal claim about the backbone.
4. Finding 4, multi-dimensional explanation quality: higher ground-truth agreement does not automatically imply uniformly better faithfulness or stability. Edge AUROC/AP, Fid+/Fid-, and Jaccard stability answer related but distinct questions.
5. Finding 5, common-pool sensitivity: the near-perfect GraphSAGE PGExplainer edge-ranking metrics persisted under the controlled common `48/16` PG training/validation pool, with only small changes from the natural `106/35` condition. This supports the interpretation that the natural-pool result was not solely due to the larger GraphSAGE-specific PG pool, but it does not establish a pure or causal backbone explanation.
6. GT agreement and faithfulness are not perfectly aligned per node. The paired per-node matrix is the source of truth for these descriptive checks.
7. On BA-House, PGExplainer shows higher mean Jaccard stability than GNNExplainer under both evaluated backbone contexts. Random has the lowest stability and remains a sanity baseline outside the 2x2 matrix.
8. The methods show different ground-truth agreement, faithfulness, stability, and computational profiles across the two backbone contexts. PGExplainer requires upfront training followed by fast inference, whereas GNNExplainer optimizes an explanation separately for each target.
9. Nodes with relatively lower agreement metrics are descriptive only. Candidate size and GT prevalence are recorded per node for inspection, but no significance or causal claim is made.

## Thesis-Reuse Summary

The BA-House 2x2 evaluation uses a common frozen 12-node target set for all four primary backbone-explainer cells. In the GCN context, GNNExplainer achieved higher ground-truth edge-ranking agreement than PGExplainer according to both Edge AUROC and Edge AP. In the GraphSAGE context, PGExplainer achieved higher ground-truth edge-ranking agreement than GNNExplainer and reached Edge AUROC/AP of `1.000000` on the frozen target set. The relative ranking of the two explainers therefore changes across backbone contexts under the BA-House protocol. Edge AUROC/AP measure agreement with the synthetic ground-truth edge mask, while Fid+/Fid- measure how selected explanation edges affect the trained classifier behavior. PGExplainer showed higher mean Jaccard stability than GNNExplainer in both evaluated backbone contexts. Task19B showed that the strong GraphSAGE PGExplainer ranking result persisted when PGExplainer for both backbone contexts used the same common 48 training-node and 16 validation-node identities, while evaluation remained on the frozen 12-node target set. The GCN and GraphSAGE classifiers also differ substantially in predictive performance on BA-House, so cross-backbone explanation-metric differences are interpreted as backbone-context observations rather than isolated causal effects. These statements are BA-House-specific, protocol-specific, and non-causal.

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

## Validation And Git Provenance

- Current branch: `main`
- Current HEAD: `44a4ad2 Freeze BA-House 2x2 explainer evaluation`
- Suggested reviewed tag: `ba-house-2x2-v1`
- Tag already exists at audit time: `False`
- `pytest`: `98 passed, 3 warnings`
- `ruff check .`: `pass`
- `git diff --check`: `pass`
- Experiment rerun during this reporting audit: `False`
- Experiment code changed during this reporting audit: `False`

## Acceptance Checklist

- 12-target hash verified: `True`
- All four primary cells use same node IDs: `True`
- Four primary cells success counts recorded: `True`
- No frozen experiment rerun: `True`
- Task19A preserved: `True`
- Task19B preserved as post-hoc sensitivity: `True`
- Original Task19 natural-pool result preserved: `True`
- Common-pool semantics correctly described: `True`
- Faithfulness definitions recorded: `True`
- Probability diagnostics defined: `True`
- Stability aggregation unit documented: `True`
- Random node-level quality aggregation documented: `True`
- Random stability documented: `True`
- Fixed 20% budget documented: `True`
- Sparsity treated as protocol confirmation: `True`
- Backbone predictive-performance context included: `True`
- Per-node diagnostic criterion explicit: `True`
- No arbitrary composite metric: `True`
- Statistical-discipline note included: `True`
- Primary within-backbone comparisons identified: `True`
- Controlled PG cross-backbone result separated: `True`
- GraphSAGE natural-vs-common PG sensitivity separated: `True`
- Runtime semantics decomposed: `True`
- Delta directions defined: `True`
- No causal or universal claim: `True`
- Report says BA-House only: `True`
- Freeze manifest verified: `True`
- Tests/lint/diff validation recorded: `True`
- Git provenance recorded: `True`

## Review Gate

Stop for review after this Task20 freeze package. Do not start ShapeGGen automatically.
