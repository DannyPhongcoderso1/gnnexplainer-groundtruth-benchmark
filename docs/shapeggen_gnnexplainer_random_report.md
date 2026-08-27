# ShapeGGen Task23 GNNExplainer And Random Audit Freeze

## Scope And Stop Boundary

This document audits Task23 only. It does not rerun experiments, tune final targets, replace seed42 with stability seeds, or start PGExplainer/Task24.

## Frozen Inputs

| Input | Path | SHA256 | Expected | Match |
|---|---|---|---|---:|
| dataset | `artifacts\data\shapeggen.pt` | `9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e` | `9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e` | True |
| targets | `configs\targets\shapeggen_2x2_targets.csv` | `bd128dec7810421d592debe8177f3cb46d6c7d9b01fd3f562b8b13971517b41d` | `bd128dec7810421d592debe8177f3cb46d6c7d9b01fd3f562b8b13971517b41d` | True |
| gcn_checkpoint | `artifacts\checkpoints\shapeggen_gcn_best.pt` | `15bfa8960fe4e0f0712c3a794db93ed4e90aad6fce3555a16db23a8a13b91e36` | `15bfa8960fe4e0f0712c3a794db93ed4e90aad6fce3555a16db23a8a13b91e36` | True |
| graphsage_checkpoint | `artifacts\checkpoints\shapeggen_graphsage_best.pt` | `498a74875360a82749b78cb40a69b3e4ec04ec634b85e4713e85d96f34c666a0` | `498a74875360a82749b78cb40a69b3e4ec04ec634b85e4713e85d96f34c666a0` | True |
| task23_config | `configs\evaluation\shapeggen_task23.yaml` | `7c4d0daa3b270c002995f0e0260bece849d4a0ff0bc86f74683ac6a5dd126c8b` | `recorded` | True |
| gnnexplainer_config | `configs\explainer\gnnexplainer.yaml` | `990631db53fddcce2b4d281c24c50cfe822655fa5965762fda127c1f48b20d60` | `recorded` | True |

## Frozen Protocol

- target_count: 167
- selected_edge_ratio: 0.2
- rounding_rule: ceil(candidate_edge_count * selected_edge_ratio), minimum 1
- edge_ranking_metric_scope: continuous_scores_over_full_candidate_edge_set
- candidate_edge_scope: target_specific_graphxai_candidate_edge_mask
- GNNExplainer primary seed: 42
- GNNExplainer stability seeds: [0, 1, 2, 3, 4]
- Random seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## Git And Environment Provenance

- git_commit: `70df56bdaffd7a76f98f668b69247be502a763d1`
- git_branch: `main`
- tags_at_head: `none`
- python: `3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]`
- platform: `Windows-10-10.0.26200-SP0`
- torch: `2.11.0+cu128`
- torch_cuda: `12.8`
- cuda_available: `True`
- gpu: `NVIDIA GeForce RTX 4060 Laptop GPU`

## Metric Definitions

- edge_auroc_edge_ap: candidate_edge_scores_and_labels receives continuous edge scores and candidate_edge_mask, then compute_edge_ground_truth_metrics computes AUROC/AP over the full candidate-edge set without thresholding.
- fid_plus: 1 when removing selected edges changes original prediction, else 0.
- fid_minus: 1 when retaining only selected edges changes original prediction, else 0.
- one_minus_fid_minus: 1 - Fid- for every raw row.
- probability_reference_class: All probabilities are softmax probabilities of the original predicted class.
- remove_probability_delta: original_probability - remove_probability.
- retain_probability: softmax probability of original predicted class on retain-only graph.
- retain_probability_delta: original_probability - retain_probability.
- retain_probability_ratio: retain_probability / original_probability when original_probability != 0.0, otherwise None. No epsilon or clipping is used.
- sparsity: 1 - selected_edge_count / candidate_edge_count.

## Recomputed Quality Summary

| Backbone | Method | Edge AUROC | Edge AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GCN | GNNExplainer | 0.648445 | 0.324281 | 0.562874 | 0.011976 | 0.988024 | 0.795928 | 0.367797 |
| GCN | Random | 0.502273 | 0.237006 | 0.127545 | 0.239521 | 0.760479 | 0.795928 | 0.116099 |
| GraphSAGE | GNNExplainer | 0.631585 | 0.297549 | 0.497006 | 0.257485 | 0.742515 | 0.795928 | 0.321343 |
| GraphSAGE | Random | 0.502273 | 0.237006 | 0.128743 | 0.585629 | 0.414371 | 0.795928 | 0.116099 |

## Full Dispersion Tables

### GCN GNNExplainer seed42 quality

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| edge_auroc | 167 | 0.648445 | 0.067154 | 0.647788 | 0.478571 | 0.850000 |
| edge_ap | 167 | 0.324281 | 0.098408 | 0.316146 | 0.134013 | 0.639648 |
| fid_plus | 167 | 0.562874 | 0.497523 | 1.000000 | 0.000000 | 1.000000 |
| fid_minus | 167 | 0.011976 | 0.109105 | 0.000000 | 0.000000 | 1.000000 |
| one_minus_fid_minus | 167 | 0.988024 | 0.109105 | 1.000000 | 0.000000 | 1.000000 |
| remove_probability_delta | 167 | 0.246468 | 0.320759 | 0.268808 | -0.471428 | 0.900782 |
| retain_probability | 167 | 0.943223 | 0.109289 | 0.978325 | 0.017170 | 0.999889 |
| retain_probability_delta | 167 | -0.225394 | 0.136997 | -0.223372 | -0.481501 | 0.640771 |
| retain_probability_ratio | 167 | 1.345208 | 0.253414 | 1.303052 | 0.026097 | 1.931069 |
| sparsity | 167 | 0.795928 | 0.003666 | 0.796512 | 0.777778 | 0.800000 |
| runtime_seconds | 167 | 0.048118 | 0.026834 | 0.044654 | 0.035499 | 0.363731 |
| generation_runtime_seconds | 167 | 0.048118 | 0.026834 | 0.044654 | 0.035499 | 0.363731 |
| evaluation_runtime_seconds | 167 | 0.014312 | 0.004095 | 0.013103 | 0.009433 | 0.027105 |

### GCN Random node-level quality

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| edge_auroc | 167 | 0.502273 | 0.026172 | 0.503162 | 0.387395 | 0.570378 |
| edge_ap | 167 | 0.237006 | 0.078723 | 0.231524 | 0.084175 | 0.507570 |
| fid_plus | 167 | 0.127545 | 0.171691 | 0.000000 | 0.000000 | 0.700000 |
| fid_minus | 167 | 0.239521 | 0.239467 | 0.200000 | 0.000000 | 1.000000 |
| one_minus_fid_minus | 167 | 0.760479 | 0.239467 | 0.800000 | 0.000000 | 1.000000 |
| remove_probability_delta | 167 | 0.002619 | 0.066535 | 0.001556 | -0.178481 | 0.251788 |
| retain_probability | 167 | 0.720531 | 0.195719 | 0.752150 | 0.189571 | 0.998389 |
| retain_probability_delta | 167 | -0.002701 | 0.205380 | -0.031583 | -0.448721 | 0.470912 |
| retain_probability_ratio | 167 | 1.022681 | 0.310010 | 1.039941 | 0.333204 | 1.828665 |
| sparsity | 167 | 0.795928 | 0.003666 | 0.796512 | 0.777778 | 0.800000 |
| runtime_seconds | 167 | 0.019086 | 0.002177 | 0.018787 | 0.012113 | 0.024715 |
| generation_runtime_seconds | 0 | NA | NA | NA | NA | NA |
| evaluation_runtime_seconds | 167 | 0.019086 | 0.002177 | 0.018787 | 0.012113 | 0.024715 |

### GCN GNNExplainer stability

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| mean_pairwise_jaccard | 167 | 0.367797 | 0.051099 | 0.375318 | 0.199768 | 0.504207 |

### GCN Random stability

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| mean_pairwise_jaccard | 167 | 0.116099 | 0.007710 | 0.115616 | 0.098998 | 0.150549 |

### GraphSAGE GNNExplainer seed42 quality

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| edge_auroc | 167 | 0.631585 | 0.060833 | 0.626414 | 0.461905 | 0.780208 |
| edge_ap | 167 | 0.297549 | 0.100516 | 0.289933 | 0.102859 | 0.671247 |
| fid_plus | 167 | 0.497006 | 0.501495 | 0.000000 | 0.000000 | 1.000000 |
| fid_minus | 167 | 0.257485 | 0.438564 | 0.000000 | 0.000000 | 1.000000 |
| one_minus_fid_minus | 167 | 0.742515 | 0.438564 | 1.000000 | 0.000000 | 1.000000 |
| remove_probability_delta | 167 | 0.453402 | 0.405311 | 0.465970 | -0.341256 | 0.998674 |
| retain_probability | 167 | 0.728904 | 0.359180 | 0.938372 | 0.000180 | 1.000000 |
| retain_probability_delta | 167 | 0.230430 | 0.344820 | 0.048583 | -0.355926 | 0.998374 |
| retain_probability_ratio | 167 | 0.755205 | 0.372738 | 0.951413 | 0.000197 | 1.553856 |
| sparsity | 167 | 0.795928 | 0.003666 | 0.796512 | 0.777778 | 0.800000 |
| runtime_seconds | 167 | 0.031272 | 0.007081 | 0.030317 | 0.023372 | 0.107734 |
| generation_runtime_seconds | 167 | 0.031272 | 0.007081 | 0.030317 | 0.023372 | 0.107734 |
| evaluation_runtime_seconds | 167 | 0.009278 | 0.002622 | 0.008563 | 0.005888 | 0.019402 |

### GraphSAGE Random node-level quality

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| edge_auroc | 167 | 0.502273 | 0.026172 | 0.503162 | 0.387395 | 0.570378 |
| edge_ap | 167 | 0.237006 | 0.078723 | 0.231524 | 0.084175 | 0.507570 |
| fid_plus | 167 | 0.128743 | 0.237989 | 0.000000 | 0.000000 | 1.000000 |
| fid_minus | 167 | 0.585629 | 0.461177 | 0.900000 | 0.000000 | 1.000000 |
| one_minus_fid_minus | 167 | 0.414371 | 0.461177 | 0.100000 | 0.000000 | 1.000000 |
| remove_probability_delta | 167 | 0.115199 | 0.176748 | 0.035028 | -0.166944 | 0.840878 |
| retain_probability | 167 | 0.453841 | 0.424049 | 0.275200 | 0.000001 | 1.000000 |
| retain_probability_delta | 167 | 0.505493 | 0.418558 | 0.650989 | -0.352139 | 0.998800 |
| retain_probability_ratio | 167 | 0.469430 | 0.443091 | 0.275224 | 0.000001 | 1.547963 |
| sparsity | 167 | 0.795928 | 0.003666 | 0.796512 | 0.777778 | 0.800000 |
| runtime_seconds | 167 | 0.014183 | 0.001369 | 0.014082 | 0.011446 | 0.017740 |
| generation_runtime_seconds | 0 | NA | NA | NA | NA | NA |
| evaluation_runtime_seconds | 167 | 0.014183 | 0.001369 | 0.014082 | 0.011446 | 0.017740 |

### GraphSAGE GNNExplainer stability

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| mean_pairwise_jaccard | 167 | 0.321343 | 0.094019 | 0.343239 | 0.160258 | 0.575310 |

### GraphSAGE Random stability

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| mean_pairwise_jaccard | 167 | 0.116099 | 0.007710 | 0.115616 | 0.098998 | 0.150549 |

## Differences Versus Previous Report

| Backbone | Method | Metric | Recomputed | Previous rounded | Difference |
|---|---|---|---:|---:|---:|
| GCN | GNNExplainer | edge_auroc | 0.648445 | 0.648445 | -1.90701058278e-08 |
| GCN | GNNExplainer | edge_ap | 0.324281 | 0.324281 | -2.05831782174e-08 |
| GCN | GNNExplainer | fid_plus | 0.562874 | 0.562874 | 2.51497005999e-07 |
| GCN | GNNExplainer | fid_minus | 0.011976 | 0.011976 | 4.7904191617e-08 |
| GCN | GNNExplainer | one_minus_fid_minus | 0.988024 | 0.988024 | -4.79041916135e-08 |
| GCN | GNNExplainer | sparsity | 0.795928 | 0.795928 | -3.77594015499e-07 |
| GCN | GNNExplainer | stability | 0.367797 | 0.367797 | -2.05670589004e-07 |
| GCN | Random | edge_auroc | 0.502273 | 0.502273 | 4.9805557778e-07 |
| GCN | Random | edge_ap | 0.237006 | 0.237006 | 3.20113550112e-08 |
| GCN | Random | fid_plus | 0.127545 | 0.127545 | -8.98203592614e-08 |
| GCN | Random | fid_minus | 0.239521 | 0.239521 | -4.19161676757e-08 |
| GCN | Random | one_minus_fid_minus | 0.760479 | 0.760479 | 4.19161676479e-08 |
| GCN | Random | sparsity | 0.795928 | 0.795928 | -3.77594015499e-07 |
| GCN | Random | stability | 0.116099 | 0.116099 | -9.04674057889e-08 |
| GraphSAGE | GNNExplainer | edge_auroc | 0.631585 | 0.631585 | 2.6405486242e-07 |
| GraphSAGE | GNNExplainer | edge_ap | 0.297549 | 0.297549 | 2.93373302762e-07 |
| GraphSAGE | GNNExplainer | fid_plus | 0.497006 | 0.497006 | -1.19760479311e-08 |
| GraphSAGE | GNNExplainer | fid_minus | 0.257485 | 0.257485 | 2.99401197168e-08 |
| GraphSAGE | GNNExplainer | one_minus_fid_minus | 0.742515 | 0.742515 | -2.99401198278e-08 |
| GraphSAGE | GNNExplainer | sparsity | 0.795928 | 0.795928 | -3.77594015499e-07 |
| GraphSAGE | GNNExplainer | stability | 0.321343 | 0.321343 | -4.69756167742e-07 |
| GraphSAGE | Random | edge_auroc | 0.502273 | 0.502273 | 4.9805557778e-07 |
| GraphSAGE | Random | edge_ap | 0.237006 | 0.237006 | 3.20113550112e-08 |
| GraphSAGE | Random | fid_plus | 0.128743 | 0.128743 | -4.85029940128e-07 |
| GraphSAGE | Random | fid_minus | 0.585629 | 0.585629 | -2.57485029853e-07 |
| GraphSAGE | Random | one_minus_fid_minus | 0.414371 | 0.414371 | 2.57485029964e-07 |
| GraphSAGE | Random | sparsity | 0.795928 | 0.795928 | -3.77594015499e-07 |
| GraphSAGE | Random | stability | 0.116099 | 0.116099 | -9.04674057889e-08 |

## Success And Failure Coverage

| Backbone | Run | Expected | Succeeded | Failed | Missing | Duplicated |
|---|---|---:|---:|---:|---:|---:|
| GCN | gnn_primary_seed42 | 167 | 167 | 0 | 0 | 0 |
| GCN | gnn_stability_seeds0_to_4 | 835 | 835 | 0 | 0 | 0 |
| GCN | random_seeds0_to_9 | 1670 | 1670 | 0 | 0 | 0 |
| GCN | random_node_means | 167 | 167 | 0 | 0 | 0 |
| GraphSAGE | gnn_primary_seed42 | 167 | 167 | 0 | 0 | 0 |
| GraphSAGE | gnn_stability_seeds0_to_4 | 835 | 835 | 0 | 0 | 0 |
| GraphSAGE | random_seeds0_to_9 | 1670 | 1670 | 0 | 0 | 0 |
| GraphSAGE | random_node_means | 167 | 167 | 0 | 0 | 0 |

## Candidate And Ground-Truth Audit

- target_count: 167
- unique_node_count: 167
- duplicate_target_count: 0
- sorted_by_node_id: True
- all_split_test: True
- violations: NONE
- reverse_duplicate_policy: {'directed_edge_entries': 4518, 'self_loop_count': 0, 'duplicate_directed_edge_count': 0, 'missing_reverse_edge_count': 0, 'matches_task21_policy': True}

| Metric | n | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|---:|
| candidate_edge_count | 167 | 123.185629 | 53.265941 | 118.000000 | 30.000000 | 278.000000 |
| gt_positive_edge_count | 167 | 22.491018 | 6.605828 | 26.000000 | 12.000000 | 28.000000 |
| gt_negative_edge_count | 167 | 100.694611 | 49.532722 | 94.000000 | 16.000000 | 252.000000 |
| selected_edge_count | 167 | 25.041916 | 10.655816 | 24.000000 | 6.000000 | 56.000000 |

## Random Aggregation Audit

Random raw seed rows are preserved. The primary Random quality table first averages each metric across seeds 0..9 inside each node_id, then summarizes the 167 node-level means across nodes. Edge AUROC/AP can match across backbones because they depend on the same candidate/GT/random scores. Perturbation metrics can differ because predictions after edge removal or retention are computed by each frozen backbone.

## Stability Audit

| Backbone | Method | Expected pairs | Actual pairs | Duplicate pairs | Self pairs | Out-of-range | Source |
|---|---|---:|---:|---:|---:|---:|---|
| GCN | gnnexplainer | 1670 | 1670 | 0 | 0 | 0 | independent per-instance explanation optimization runs |
| GCN | random | 7515 | 7515 | 0 | 0 | 0 | repeated random edge selections |
| GraphSAGE | gnnexplainer | 1670 | 1670 | 0 | 0 | 0 | independent per-instance explanation optimization runs |
| GraphSAGE | random | 7515 | 7515 | 0 | 0 | 0 | repeated random edge selections |

## Runtime Audit

GNNExplainer runtime is per-target explanation optimization/generation time. Random runtime is random edge selection plus metric evaluation time.

| Backbone | Run | Field | n | mean | std | median | min | max | total |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| GCN | GNNExplainer seed42 generation plus separate metric evaluation | runtime_seconds | 167 | 0.048118 | 0.026834 | 0.044654 | 0.035499 | 0.363731 | 8.035734 |
| GCN | GNNExplainer seed42 generation plus separate metric evaluation | generation_runtime_seconds | 167 | 0.048118 | 0.026834 | 0.044654 | 0.035499 | 0.363731 | 8.035734 |
| GCN | GNNExplainer seed42 generation plus separate metric evaluation | evaluation_runtime_seconds | 167 | 0.014312 | 0.004095 | 0.013103 | 0.009433 | 0.027105 | 2.390183 |
| GCN | Random per-node seed-mean quality runtime reference | runtime_seconds | 167 | 0.019086 | 0.002177 | 0.018787 | 0.012113 | 0.024715 | 3.187419 |
| GCN | Random per-node seed-mean quality runtime reference | generation_runtime_seconds | 0 | NA | NA | NA | NA | NA | NA |
| GCN | Random per-node seed-mean quality runtime reference | evaluation_runtime_seconds | 167 | 0.019086 | 0.002177 | 0.018787 | 0.012113 | 0.024715 | 3.187419 |
| GCN | Random edge selection plus metric evaluation per seed row | runtime_seconds | 1670 | 0.019086 | 0.005757 | 0.018498 | 0.009278 | 0.057958 | 31.874187 |
| GCN | Random edge selection plus metric evaluation per seed row | generation_runtime_seconds | 0 | NA | NA | NA | NA | NA | NA |
| GCN | Random edge selection plus metric evaluation per seed row | evaluation_runtime_seconds | 1670 | 0.019086 | 0.005757 | 0.018498 | 0.009278 | 0.057958 | 31.874187 |
| GraphSAGE | GNNExplainer seed42 generation plus separate metric evaluation | runtime_seconds | 167 | 0.031272 | 0.007081 | 0.030317 | 0.023372 | 0.107734 | 5.222477 |
| GraphSAGE | GNNExplainer seed42 generation plus separate metric evaluation | generation_runtime_seconds | 167 | 0.031272 | 0.007081 | 0.030317 | 0.023372 | 0.107734 | 5.222477 |
| GraphSAGE | GNNExplainer seed42 generation plus separate metric evaluation | evaluation_runtime_seconds | 167 | 0.009278 | 0.002622 | 0.008563 | 0.005888 | 0.019402 | 1.549492 |
| GraphSAGE | Random per-node seed-mean quality runtime reference | runtime_seconds | 167 | 0.014183 | 0.001369 | 0.014082 | 0.011446 | 0.017740 | 2.368619 |
| GraphSAGE | Random per-node seed-mean quality runtime reference | generation_runtime_seconds | 0 | NA | NA | NA | NA | NA | NA |
| GraphSAGE | Random per-node seed-mean quality runtime reference | evaluation_runtime_seconds | 167 | 0.014183 | 0.001369 | 0.014082 | 0.011446 | 0.017740 | 2.368619 |
| GraphSAGE | Random edge selection plus metric evaluation per seed row | runtime_seconds | 1670 | 0.014183 | 0.004039 | 0.014603 | 0.006019 | 0.038441 | 23.686193 |
| GraphSAGE | Random edge selection plus metric evaluation per seed row | generation_runtime_seconds | 0 | NA | NA | NA | NA | NA | NA |
| GraphSAGE | Random edge selection plus metric evaluation per seed row | evaluation_runtime_seconds | 1670 | 0.014183 | 0.004039 | 0.014603 | 0.006019 | 0.038441 | 23.686193 |

## Artifact Inventory

| Backbone | Family | Path | Role | Rows/Runs | Columns/File count | SHA256/Manifest | Missing/Violations |
|---|---|---|---|---:|---:|---|---|
| GCN | gnn_primary | `results\metrics\shapeggen\gcn\gnnexplainer\per_node_metrics.csv` | metric_csv | 167 | 33 | `ff6e1136daeec2c193289e7c3c988709ad37551d3c54a07d8c29b6903de4dc06` | missing=167, nan_inf=0, duplicates=0 |
| GCN | gnn_stability_seed | `results\metrics\shapeggen\gcn\gnnexplainer\stability_seed_metrics.csv` | metric_csv | 835 | 33 | `59f2ed269ef2912e1cf3164ad56a56c51020dbfc59aea07cb7a1d61658ce3c79` | missing=835, nan_inf=0, duplicates=0 |
| GCN | gnn_stability_pairwise | `results\metrics\shapeggen\gcn\gnnexplainer\stability_pairwise_jaccard.csv` | metric_csv | 1670 | 7 | `6c7887386621a699dfc29db78a7e2d53ab564249333e9730c92ca94b3114d432` | missing=0, nan_inf=0, duplicates=0 |
| GCN | gnn_stability_node | `results\metrics\shapeggen\gcn\gnnexplainer\stability_per_node_jaccard.csv` | metric_csv | 167 | 10 | `51da556a098048d6a5087fd9ed23e54c8ab3557964f2c445cd96790c297ee7bc` | missing=167, nan_inf=0, duplicates=0 |
| GCN | random_seed | `results\metrics\shapeggen\gcn\random\random_seed_level_metrics.csv` | metric_csv | 1670 | 33 | `060baf55fd9ee15c18be026ea28868349ff9109d9a0dd08d14d3e93343f98794` | missing=3340, nan_inf=0, duplicates=0 |
| GCN | random_node | `results\metrics\shapeggen\gcn\random\random_node_level_metrics.csv` | metric_csv | 167 | 33 | `0f1ad5f9f5e5884483eefc594a07b32e77e5a7e283f226c2c6ae94eb80bacfa6` | missing=334, nan_inf=0, duplicates=0 |
| GCN | random_stability_pairwise | `results\metrics\shapeggen\gcn\random\random_stability_pairwise_jaccard.csv` | metric_csv | 7515 | 7 | `a385eb15828a00e835d1f8842c04b434e2e918e7b28d33521ff577e4ef20d5dc` | missing=0, nan_inf=0, duplicates=0 |
| GCN | random_stability_node | `results\metrics\shapeggen\gcn\random\random_stability_per_node_jaccard.csv` | metric_csv | 167 | 10 | `5521d8107735f744a5cdfdbb5a2a91c188b87ec0e24d2c5bf5368a798dbc0032` | missing=167, nan_inf=0, duplicates=0 |
| GCN | gnnexplainer | `results\explanations\shapeggen\gcn\gnnexplainer` | gcn_gnnexplainer_raw_explanations | 1002/1002 | 6012 | `748c5bb57dc8d5199aa9957233bdea7262e91d30c775a84d48e91e3ba1a1c207` | missing_dirs=0, missing_files={}, score_len=0, edge_mask_size=0, selected_count=0 |
| GCN | random | `results\explanations\shapeggen\gcn\random` | gcn_random_raw_explanations | 1670/1670 | 10020 | `aefe3607dde5a99d0dbcc84c3c7c2f7def3f8be2a3c092821bcb8bbafd100c1c` | missing_dirs=0, missing_files={}, score_len=0, edge_mask_size=0, selected_count=0 |
| GraphSAGE | gnn_primary | `results\metrics\shapeggen\graphsage\gnnexplainer\per_node_metrics.csv` | metric_csv | 167 | 33 | `153c372a076241b80baabf9a684e4dec47b39e38b18bb5e93caa32bfaccd11c8` | missing=167, nan_inf=0, duplicates=0 |
| GraphSAGE | gnn_stability_seed | `results\metrics\shapeggen\graphsage\gnnexplainer\stability_seed_metrics.csv` | metric_csv | 835 | 33 | `dd42fe250fb09c3cfc82d3c93a30d8d5b868e77646d2022f39c0caf8801c3fd9` | missing=835, nan_inf=0, duplicates=0 |
| GraphSAGE | gnn_stability_pairwise | `results\metrics\shapeggen\graphsage\gnnexplainer\stability_pairwise_jaccard.csv` | metric_csv | 1670 | 7 | `ab0fa97e0b17e8620d0faed6dc04f624d693bd891eb16d8904277ae13d08127f` | missing=0, nan_inf=0, duplicates=0 |
| GraphSAGE | gnn_stability_node | `results\metrics\shapeggen\graphsage\gnnexplainer\stability_per_node_jaccard.csv` | metric_csv | 167 | 10 | `f63e61a91b9eff44ae806259b54ac19e77dbc8eb14e5ba86de60ab16d133da63` | missing=167, nan_inf=0, duplicates=0 |
| GraphSAGE | random_seed | `results\metrics\shapeggen\graphsage\random\random_seed_level_metrics.csv` | metric_csv | 1670 | 33 | `994ebd3972ac5a28406435cd984ce9bae631df4ae2f508306f5280697f6494c1` | missing=3340, nan_inf=0, duplicates=0 |
| GraphSAGE | random_node | `results\metrics\shapeggen\graphsage\random\random_node_level_metrics.csv` | metric_csv | 167 | 33 | `de8beb38b327e9af42499d419560a3433af0b0f4672d063af4b714cd1d635032` | missing=334, nan_inf=0, duplicates=0 |
| GraphSAGE | random_stability_pairwise | `results\metrics\shapeggen\graphsage\random\random_stability_pairwise_jaccard.csv` | metric_csv | 7515 | 7 | `a385eb15828a00e835d1f8842c04b434e2e918e7b28d33521ff577e4ef20d5dc` | missing=0, nan_inf=0, duplicates=0 |
| GraphSAGE | random_stability_node | `results\metrics\shapeggen\graphsage\random\random_stability_per_node_jaccard.csv` | metric_csv | 167 | 10 | `5521d8107735f744a5cdfdbb5a2a91c188b87ec0e24d2c5bf5368a798dbc0032` | missing=167, nan_inf=0, duplicates=0 |
| GraphSAGE | gnnexplainer | `results\explanations\shapeggen\graphsage\gnnexplainer` | graphsage_gnnexplainer_raw_explanations | 1002/1002 | 6012 | `c329079628a07a5106368617eed766c4cfe95d9fa2c3745a762de964e72278cb` | missing_dirs=0, missing_files={}, score_len=0, edge_mask_size=0, selected_count=0 |
| GraphSAGE | random | `results\explanations\shapeggen\graphsage\random` | graphsage_random_raw_explanations | 1670/1670 | 10020 | `7c2d372a4ce685543e7306c9b2098a4d9eafc464edb77cae2608a574245c7fac` | missing_dirs=0, missing_files={}, score_len=0, edge_mask_size=0, selected_count=0 |

## Validation

- pytest: 115 passed, 3 warnings
- Ruff: All checks passed
- git diff --check: pass

## Acceptance Checklist

- frozen_input_hashes_match: True
- target_candidate_gt_mapping_valid: True
- coverage_complete_without_failures_or_duplicates: True
- raw_artifacts_complete_and_aligned: True
- metric_formula_checks_pass: True
- stability_pair_counts_valid: True
- reported_values_recomputed: True
- fid_count_checks_match: True
- task24_pgexplainer_not_started: True

## Remaining Blockers

NONE

STOP FOR REVIEW
