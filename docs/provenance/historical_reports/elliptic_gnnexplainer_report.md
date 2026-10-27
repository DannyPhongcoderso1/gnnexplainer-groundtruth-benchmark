# Elliptic GNNExplainer Final Audit Report

## Scope And Stop Boundary

Task27 has completed GNNExplainer on the frozen Elliptic GCN and GraphSAGE backbones. This final audit only inspects existing artifacts and deterministic aggregations. It does not rerun GNNExplainer, retrain any backbone, run PGExplainer, run Random, calculate Edge AUROC/AP, change targets, change checkpoints, change seeds, change the 20% budget, or start Task28.

## Frozen Inputs And Git Provenance

- Execution source commit: `a68b933490ef6b94d92dd3b55f2bddc8bf139226`
- Initial Task27 result commit: `940ca4c51b24c845d3a94477f63280103825c1e4`
- Branch: `main`
- Dataset SHA256: `f807b83a7c739b16f69b5676226ce441b64d5fe8de95506a7727f86e8451291b`
- GCN checkpoint SHA256: `55004ffa1b712ae374fbf630d55f31ac69602d696dbeef8cb150d3f788a9cc07`
- GraphSAGE checkpoint SHA256: `f25d5cbeb10f9a27dbba1adcb2fdb21b406219956c107f8d51e8ecf5188866bf`
- Target CSV SHA256: `e414941e6943b06484ecd5028c51daf80ff2bf9f1bbb06e6a5fd816de8dfb9c0`
- Task27 config SHA256: `f6f4e6ac97116782bf4b6047f50c92c117bbb12229d311a934cefbdb4b7313b3`

## Optimization Target Semantics

- explanation_type: `phenomenon`
- target source: `prediction_vector(model, data) from frozen explanation-compatible model logits`
- code path: `scripts/run_elliptic_gnnexplainer.py::prediction_vector -> run_explainer -> Explainer(... target=target_vector, index=node_id)`
- target shape: `[num_nodes] = [203769]`
- target dtype: `torch.int64 / long class-index tensor from argmax`
- conclusion: `PASS: phenomenon target equals original frozen model prediction and is not chosen from ground-truth transaction label.`

| Backbone | Checked rows | Target=prediction | Mismatches | Target=GT | Target!=GT |
| --- | --- | --- | --- | --- | --- |
| GCN | 1200 | 1200 | 0 | 1164 | 36 |
| GraphSAGE | 1200 | 1200 | 0 | 1170 | 30 |

## Candidate Scope And Budget

- Candidate scope: `2-hop computation subgraph around target node`
- Selected edge ratio: `0.2`
- Rounding rule: `max(1, ceil(candidate_edge_count * 0.20))`
- Probability reference class: `original predicted class`

## Success And Metric Coverage

| Backbone | Primary success | Primary failure | Stability seed success | Stability seed failure | Pair rows |
| --- | --- | --- | --- | --- | --- |
| GCN | 200 | 0 | 1000 | 0 | 2000 |
| GraphSAGE | 200 | 0 | 1000 | 0 | 2000 |

## Main Metric Dispersion

| Backbone | Fid+ | Fid- | 1-Fid- | Remove delta | Retain probability | Retain delta | Retain ratio | Sparsity | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | 0.055000 | 0.000000 | 1.000000 | 0.045679 | 0.999209 | -0.012138 | 1.017160 | 0.775332 | 0.501168 |
| GraphSAGE | 0.035000 | 0.005000 | 0.995000 | 0.044117 | 0.990821 | -0.001415 | 1.003377 | 0.775332 | 0.574761 |

Fid+, Fid-, and 1-Fid- are binary prediction-change summaries. Remove-side and retain-side probability diagnostics are reported separately and should not be collapsed into a single generic fidelity number.

## Raw Continuous Edge-Score Artifact Audit

- Inventory path: `docs\elliptic_gnnexplainer_raw_artifact_inventory.json`
- Inventory SHA256: `c6d09709725f90f2c73d5e624b6a76019f4ac22cd66c1367ec3a35e9c5eb00ba`
- Candidate edge IDs are stored as deterministic SHA256 hashes per run to keep the tracked inventory compact.

| Backbone | Expected raw | Actual raw | Primary | Stability | Missing | Duplicates | Failed | Score mismatch | Selected violations | NaN/Inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | 1200 | 1200 | 200 | 1000 | 0 | 0 | 0 | 0 | 0 | 0 |
| GraphSAGE | 1200 | 1200 | 200 | 1000 | 0 | 0 | 0 | 0 | 0 | 0 |

## Candidate And Selected-Edge Distributions

| Backbone | Candidate distribution | Selected distribution | Sparsity distribution |
| --- | --- | --- | --- |
| GCN | {'n': 1200, 'mean': 126.56, 'std': 240.46794747214832, 'median': 27.0, 'min': 4.0, 'max': 1424.0, 'sum': 151872.0} | {'n': 1200, 'mean': 25.68, 'std': 48.05290662241344, 'median': 6.0, 'min': 1.0, 'max': 285.0, 'sum': 30816.0} | {'n': 1200, 'mean': 0.7753316951532506, 'std': 0.03514007838053787, 'median': 0.7948012575791601, 'min': 0.6666666666666667, 'max': 0.8, 'sum': 930.3980341838961} |
| GraphSAGE | {'n': 1200, 'mean': 126.56, 'std': 240.46794747214832, 'median': 27.0, 'min': 4.0, 'max': 1424.0, 'sum': 151872.0} | {'n': 1200, 'mean': 25.68, 'std': 48.05290662241344, 'median': 6.0, 'min': 1.0, 'max': 285.0, 'sum': 30816.0} | {'n': 1200, 'mean': 0.7753316951532506, 'std': 0.03514007838053787, 'median': 0.7948012575791601, 'min': 0.6666666666666667, 'max': 0.8, 'sum': 930.3980341838961} |

## Metric Identity Checks

| Backbone | Checked rows | Violations | Max abs error | Zero denominators | Ratio structural nulls |
| --- | --- | --- | --- | --- | --- |
| GCN | 1200 | 0 | 0.0 | 0 | 0 |
| GraphSAGE | 1200 | 0 | 0.0 | 0 | 0 |

Checked identities: `1-Fid- = 1 - Fid-`, `remove_probability_delta = original_probability - remove_probability`, `retain_probability_delta = original_probability - retain_probability`, `retain_probability_ratio = retain_probability / original_probability`, and `sparsity = 1 - selected_edge_count / candidate_edge_count`.

## Stability Audit

| Backbone | Seeds | Pairs | Expected pairs | Self-pairs | Duplicate pairs | Wrong seeds | Out of range | Wrong node pair counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | [0, 1, 2, 3, 4] | 2000 | 2000 | 0 | 0 | 0 | 0 | 0 |
| GraphSAGE | [0, 1, 2, 3, 4] | 2000 | 2000 | 0 | 0 | 0 | 0 | 0 |

Stability uses pairwise Jaccard over selected top-20% sets. The stochastic source is independent per-instance GNNExplainer optimization runs. Summary aggregation uses per-node mean pairwise Jaccard, not 2,000 pair rows as independent node observations.

## Runtime Decomposition

| Backbone | Stage | Unit | n | Mean | Std | Median | Min | Max | Sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | primary_generation | seconds | 200 | 0.524587 | 0.015002 | 0.523326 | 0.505982 | 0.662638 | 104.917322 |
| GCN | primary_evaluation | seconds | 200 | 0.113703 | 0.008332 | 0.111568 | 0.101460 | 0.136700 | 22.740642 |
| GCN | stability_generation | seconds | 1000 | 0.506056 | 0.015848 | 0.500757 | 0.481435 | 0.548550 | 506.056095 |
| GCN | stability_evaluation | seconds | 1000 | 0.112142 | 0.011949 | 0.109434 | 0.095603 | 0.144697 | 112.141990 |
| GraphSAGE | primary_generation | seconds | 200 | 0.382976 | 0.004035 | 0.383272 | 0.376373 | 0.402490 | 76.595239 |
| GraphSAGE | primary_evaluation | seconds | 200 | 0.103991 | 0.017604 | 0.100436 | 0.083668 | 0.133086 | 20.798172 |
| GraphSAGE | stability_generation | seconds | 1000 | 0.384206 | 0.004287 | 0.384463 | 0.376083 | 0.397834 | 384.205745 |
| GraphSAGE | stability_evaluation | seconds | 1000 | 0.106312 | 0.018478 | 0.102116 | 0.083957 | 0.139380 | 106.312325 |

Batch wall-clock total: `1514.6692133999968` seconds. Sum of recorded stage runtimes: `1333.7675300001647` seconds. Unattributed difference: `180.9016833998321` seconds. The difference captures serialization, I/O, aggregation, setup, and timer nesting effects that were not separately timed.

## Metrics CSV Inventory

| Backbone | File | Rows | Expected | Columns | SHA256 | Duplicates | Nulls | NaN/Inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | per_node_metrics | 200 | 200 | 28 | cffb1f685f3455d44ce6a343d2d600a2dfe0796dc2ee85154cecd8634bc9f023 | 0 | 200 | 0 |
| GCN | summary_metrics | 14 | 14 | 7 | 9eb4d1f46d4908167e272c3d214d45281b29d01a8084148cfc54d93436ae01ef | 0 | 0 | 0 |
| GCN | stability_seed_metrics | 1000 | 1000 | 28 | 140ccba47e978bcff19fa657be91a236169ff76362ddea9d58f42e559a9765fe | 0 | 1000 | 0 |
| GCN | stability_pairwise_jaccard | 2000 | 2000 | 7 | e7dea22c28fc60b10c609672e6a3d1b6466188f9aec6dfb37273dfc3cb0e2e1d | 0 | 0 | 0 |
| GCN | stability_per_node_jaccard | 200 | 200 | 10 | d67422a268f020874a3da91fdaa41fde0983c0fec367c328e566338cd77a41f1 | 0 | 200 | 0 |
| GCN | stability_summary | 1 | 1 | 7 | 1ab1048f7b5fd43dded3ecd2f78be53dc026938f59a921a9b787aa9b6a5a53b9 | 0 | 0 | 0 |
| GraphSAGE | per_node_metrics | 200 | 200 | 28 | a85cdc3574e3d2fead0e228afac7ee8a15aa439455b9da0465833c232666d142 | 0 | 200 | 0 |
| GraphSAGE | summary_metrics | 14 | 14 | 7 | 30e1df4657a7fe2dbc2a8acbdea36a146773421c2d7d819d421fc68414b13d32 | 0 | 0 | 0 |
| GraphSAGE | stability_seed_metrics | 1000 | 1000 | 28 | 9de12ec7cd206292a025c47c19e55be58c91e98bbe926f93f1c9bb99ae81512b | 0 | 1000 | 0 |
| GraphSAGE | stability_pairwise_jaccard | 2000 | 2000 | 7 | fe29e4fc3265062e1785f906406c2725613c3609401fa893f631d8a99320b3da | 0 | 0 | 0 |
| GraphSAGE | stability_per_node_jaccard | 200 | 200 | 10 | 520bc721b0b9b17b53123474c679be6909a77a81d285a00538cfc91d56960fd9 | 0 | 200 | 0 |
| GraphSAGE | stability_summary | 1 | 1 | 7 | ad1dc8a6b511ebeab1e9fba0fb6a8995736e014e977b5f02abc84a8bd65a4506 | 0 | 0 | 0 |

## Interpretation Boundary And Limitations

Elliptic explanations are model-importance explanations under the frozen protocol, not true illicit-transaction causes. Cross-backbone differences are descriptive backbone-context observations and should not be stated as causal effects. Edge AUROC/AP remain disabled because no ground-truth explanatory edge masks exist for Elliptic in this thesis.

## Acceptance Checklist

| Check | Status |
| --- | --- |
| cache_safe_perturbation_semantics_verified | True |
| exact_task26_common_final_targets_used | True |
| failures_explicit_not_silent | True |
| gcn_graphsage_weights_unchanged | True |
| no_auroc_ap_explanation_metrics | True |
| no_causal_cross_backbone_wording | True |
| primary_seed42_complete_both_backbones | True |
| runtime_semantics_recorded | True |
| stability_complete_on_prespecified_scope | True |
| stochastic_source_documented | True |

## Remaining Blockers

None.

## STOP FOR REVIEW

Task28 is not authorized. No rerun was performed in this audit.
