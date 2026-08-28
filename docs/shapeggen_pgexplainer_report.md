# ShapeGGen Task24 PGExplainer Freeze

## Scope

Task24 trains and evaluates PGExplainer for ShapeGGen GCN and GraphSAGE using
the frozen common PG train and validation node identities from Task22. It does
not tune on final targets and does not start Task25.

## Frozen Inputs

- Dataset: `artifacts/data/shapeggen.pt`
- Dataset SHA256: `9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e`
- Common train: `configs/targets/shapeggen_pg_common_train.csv`
- Common train SHA256: `c05a5be38a64b629c8bb837b1fb3408d8a008726713e70f0318aee3499746b70`
- Common validation: `configs/targets/shapeggen_pg_common_val.csv`
- Common validation SHA256: `b988e40c2af234724b07802ec31d0831ef039a3ef4de3bfee1b80051b49cdc5d`
- Final targets: `configs/targets/shapeggen_2x2_targets.csv`
- Final target SHA256: `bd128dec7810421d592debe8177f3cb46d6c7d9b01fd3f562b8b13971517b41d`
- GCN checkpoint SHA256: `15bfa8960fe4e0f0712c3a794db93ed4e90aad6fce3555a16db23a8a13b91e36`
- GraphSAGE checkpoint SHA256: `498a74875360a82749b78cb40a69b3e4ec04ec634b85e4713e85d96f34c666a0`

## Protocol

- PG epochs: `30`
- Learning rate: `0.003`
- Edge size: `0.05`
- Edge entropy: `1.0`
- Temperature: `5.0 -> 2.0`
- Primary quality seed: `42`
- Stability seeds: `0,1,2,3,4`
- Selected edge ratio: `0.2`
- Edge AUROC/AP: continuous scores over the full candidate-edge set
- Stochastic source: independently trained PGExplainer checkpoints

## Fairness Audit

- Same COMMON_TRAIN identities: `True`
- Same COMMON_VAL identities: `True`
- Same final node identities: `True`
- Train/val/final disjoint: `True`, `True`, `True`

## Main Summary

| Backbone | Edge AUROC | Edge AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability | Training runtime s | Mean inference runtime s | Mean evaluation runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 0.385184 | 0.182545 | 0.041916 | 0.203593 | 0.796407 | 0.795928 | 0.827289 | 180.965073 | 0.007096 | 0.014371 |
| GraphSAGE | 0.530641 | 0.285020 | 0.622754 | 0.610778 | 0.389222 | 0.795928 | 0.793581 | 110.941023 | 0.003956 | 0.008129 |

## Runtime Semantics

PGExplainer runtime is separated into upfront explainer training,
trained-explainer inference per target, and metric evaluation per target.
These fields are descriptive runtime references and are not collapsed with
GNNExplainer or Random timing boundaries.

## Outputs

- `artifacts/explainers/shapeggen_pg_gcn_seed42.pt`
- `artifacts/explainers/shapeggen_pg_graphsage_seed42.pt`
- `artifacts/explainers/shapeggen_pg_{gcn,graphsage}_seed{0..4}.pt`
- `results/metrics/shapeggen/{gcn,graphsage}/pgexplainer/`
- `results/explanations/shapeggen/{gcn,graphsage}/pgexplainer/`
- `docs/shapeggen_pgexplainer_freeze.json`

## Acceptance

- common_train_identities_identical_across_backbones: True
- common_val_identities_identical_across_backbones: True
- common_final_targets_identical_across_backbones: True
- gcn_pg_seed42_complete: True
- graphsage_pg_seed42_complete: True
- thirty_epochs_each: True
- five_stability_checkpoints_each_backbone: True
- classifier_weights_unchanged_all_runs: True
- no_final_target_leakage: True
- checkpoint_reload_passes: True
- same_twenty_percent_budget_evaluator_semantics: True
- jaccard_stability_complete: True
- no_best_seed_selection: True
- runtime_separated: True
- no_final_test_tuning: True

## Validation

- pytest: `119 passed, 3 warnings`
- Ruff: `All checks passed`
- git diff --check: `passed`

## Stop Boundary

Task24 stops here for review. Task25 is not started automatically.
