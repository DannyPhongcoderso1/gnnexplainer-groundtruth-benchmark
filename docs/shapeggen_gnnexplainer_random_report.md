# ShapeGGen GNNExplainer And Random Freeze

## Scope

Task23 runs the non-learned explanation branch on the frozen ShapeGGen GCN and
GraphSAGE backbones. It uses the frozen final 2x2 target set, target-specific
GraphXAI candidate/ground-truth edge masks, the 20% candidate-edge budget, and
the shared synthetic evaluator semantics.

No PGExplainer run was started in this task.

## Frozen Inputs

- Dataset: `artifacts/data/shapeggen.pt`
- Dataset SHA256: `9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e`
- Target file: `configs/targets/shapeggen_2x2_targets.csv`
- Target SHA256: `bd128dec7810421d592debe8177f3cb46d6c7d9b01fd3f562b8b13971517b41d`
- Target count: `167`
- GCN checkpoint SHA256: `15bfa8960fe4e0f0712c3a794db93ed4e90aad6fce3555a16db23a8a13b91e36`
- GraphSAGE checkpoint SHA256: `498a74875360a82749b78cb40a69b3e4ec04ec634b85e4713e85d96f34c666a0`
- Task config SHA256: `7c4d0daa3b270c002995f0e0260bece849d4a0ff0bc86f74683ac6a5dd126c8b`
- GNNExplainer config SHA256: `990631db53fddcce2b4d281c24c50cfe822655fa5965762fda127c1f48b20d60`

## Protocol

- Edge ranking metrics: `continuous_scores_over_full_candidate_edge_set`
- Candidate-edge scope: `target_specific_graphxai_candidate_edge_mask`
- Selected edge ratio: `0.2`
- Rounding rule: `ceil(candidate_edge_count * selected_edge_ratio), minimum 1`
- GNNExplainer primary seed: `42`
- GNNExplainer stability seeds: `[0, 1, 2, 3, 4]`
- Random seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Random quality aggregation: `mean metrics across seeds within node_id, then summarize across nodes`

## Main Summary

| Backbone | Method | Edge AUROC | Edge AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GCN | GNNExplainer | 0.648445 | 0.324281 | 0.562874 | 0.011976 | 0.988024 | 0.795928 | 0.367797 |
| GCN | Random | 0.502273 | 0.237006 | 0.127545 | 0.239521 | 0.760479 | 0.795928 | 0.116099 |
| GraphSAGE | GNNExplainer | 0.631585 | 0.297549 | 0.497006 | 0.257485 | 0.742515 | 0.795928 | 0.321343 |
| GraphSAGE | Random | 0.502273 | 0.237006 | 0.128743 | 0.585629 | 0.414371 | 0.795928 | 0.116099 |

## Runtime Semantics

GNNExplainer runtime is per-target explanation optimization/generation time.
Random runtime is random edge selection plus metric evaluation time. The two
runtime fields are descriptive references and should not be collapsed into a
single strict speed ranking.

## Acceptance

- exact_frozen_common_final_targets_used: True
- both_backbone_checkpoint_files_unchanged: True
- both_backbone_weights_unchanged: True
- candidate_gt_mapping_valid_for_every_target: True
- seed42_gnn_quality_complete_on_both_backbones: True
- gnn_stability_seeds_0_to_4_complete: True
- random_10_seed_protocol_complete: True
- random_node_level_quality_summary_used: True
- raw_random_seed_rows_preserved: True
- same_20_percent_budget: True
- same_metric_semantics: True
- runtime_labels_are_explicit: True
- no_final_test_tuning: True
- failures_explicit_never_silently_dropped: True
- pgexplainer_not_started: True

## Outputs

- `results/explanations/shapeggen/gcn/gnnexplainer/`
- `results/explanations/shapeggen/gcn/random/`
- `results/explanations/shapeggen/graphsage/gnnexplainer/`
- `results/explanations/shapeggen/graphsage/random/`
- `results/metrics/shapeggen/gcn/gnnexplainer/`
- `results/metrics/shapeggen/gcn/random/`
- `results/metrics/shapeggen/graphsage/gnnexplainer/`
- `results/metrics/shapeggen/graphsage/random/`
- `results/metrics/shapeggen/task23_comparison_summary.csv`
- `docs/shapeggen_gnnexplainer_random_freeze.json`

## Stop Boundary

Task23 stops here for review. PGExplainer, Task24, Elliptic, or any later
experiment is not started automatically.
