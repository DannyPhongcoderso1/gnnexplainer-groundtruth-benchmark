# PGExplainer Common-Pool Sensitivity Report

## 1. Motivation

Task 19 found a very strong GraphSAGE PGExplainer result using the natural
GraphSAGE eligible pool. Task 19B asks whether that result persists when the
PGExplainer training and validation node identities are controlled across GCN
and GraphSAGE.

## 2. Why Task 19B Was Added

The Task 19 audit showed that the GCN PG pool is a subset of the GraphSAGE PG
pool. A symmetric common pool therefore exists and can reduce the
training-pool membership confound for a secondary analysis.

## 3. Post-Hoc Sensitivity Status

This is a post-hoc controlled sensitivity analysis conducted after primary
final-test exposure. It is not a pre-registered primary experiment and does
not replace the original Task 19 primary results.

Primary original results remain:

- GCN PGExplainer with backbone-specific pool `48/16`
- GraphSAGE PGExplainer with backbone-specific pool `106/35`

Controlled sensitivity results are:

- GCN PGExplainer common pool `48/16`, reusing the existing frozen result
- GraphSAGE PGExplainer common pool `48/16`, newly trained in Task 19B

## 4. Common-Pool Construction

The common pool was created by deterministic `node_id` intersection:

- Common train count: `48`
- Common train SHA256: `05e4574515fb7a03b846adb677295c940eb7af380216c15dfedf65bb9430bc3c`
- Common validation count: `16`
- Common validation SHA256: `57fc6135495d9a6f8c7b8c1c3babcddd6b76187ebb353ed6fd8c289771980776`
- Final target count: `12`
- Final target SHA256: `2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f`

Train, validation, and final-test pools are disjoint.

## 5. Frozen Inputs

- Dataset SHA256: `4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94`
- GCN checkpoint SHA256: `4c81814172650e20b6edee5fc97bc66cae46ac1e8ba76baec2226c59a7d1d507`
- GraphSAGE checkpoint SHA256: `09992c5ac4d2cd80c10792c12be5f945b3a20d749d8433008eba9cf3694135eb`
- GCN PG seed42 SHA256: `a075cd84c52e14a65b042f99f33e04e6b37e4919a70132ccb71e505b08c12eda`
- GraphSAGE natural PG seed42 SHA256:
  `0fc6ca4a232388ba861005f9f6d7b63a4cc6241eb7ca1a0eb44921cf0b03fb69`
- GraphSAGE common PG seed42 SHA256:
  `4b2f34c00555c8d3dd9beb6fe04dbe14d71c9d1a7636f9958bb56b03130c843e`

## 6. GCN Common-Pool Reuse Verification

The existing frozen GCN PGExplainer train identities exactly equal the common
train identities, and the existing GCN validation identities exactly equal the
common validation identities. GCN PGExplainer was not retrained.

## 7. GraphSAGE Common-Pool Training

- Seed: `42`
- Epochs completed: `30`
- Final mean loss: `1.173682`
- Training runtime seconds: `9.819772`
- GraphSAGE weights unchanged: `True`
- Optimizer excludes GraphSAGE parameters:
  `True`
- Checkpoint reload: `True`

## 8. Seed42 Quality Results

- Targets: `12`
- Success: `12`
- Failed: `0`
- Edge AUROC: `0.999869`
- Edge AP: `0.993590`
- Fid+: `1.000000`
- Fid-: `0.500000`
- 1-Fid-: `0.500000`
- Sparsity: `0.793694`

## 9. Stability Results

- Seeds: `[0, 1, 2, 3, 4]`
- Pairwise rows: `120`
- Jaccard mean: `0.789956`
- Jaccard std: `0.204639`
- Jaccard median: `0.776083`
- Jaccard min: `0.500000`
- Jaccard max: `1.000000`

## 10. Runtime

Runtime is decomposed into training per checkpoint, inference per target, and
evaluation per final-target batch. These values are descriptive efficiency
references and are not used for model selection.

## 11. GraphSAGE Natural vs Common Pool

| Metric | Natural 106/35 | Common 48/16 | Delta |
|---|---:|---:|---:|
| edge_auroc | 1.000000 | 0.999869 | -0.000131 |
| edge_ap | 1.000000 | 0.993590 | -0.006410 |
| fid_plus | 1.000000 | 1.000000 | 0.000000 |
| fid_minus | 0.500000 | 0.500000 | 0.000000 |
| one_minus_fid_minus | 0.500000 | 0.500000 | 0.000000 |
| stability | 0.805056 | 0.789956 | -0.015100 |
| sparsity | 0.793694 | 0.793694 | 0.000000 |
| inference_runtime_seconds | 0.014336 | 0.003302 | -0.011034 |

## 12. Controlled GCN vs GraphSAGE PG Comparison

| Metric | PG + GCN common | PG + GraphSAGE common | Delta |
|---|---:|---:|---:|
| edge_auroc | 0.747969 | 0.999869 | 0.251900 |
| edge_ap | 0.366563 | 0.993590 | 0.627027 |
| fid_plus | 0.416667 | 1.000000 | 0.583333 |
| fid_minus | 0.583333 | 0.500000 | -0.083333 |
| one_minus_fid_minus | 0.416667 | 0.500000 | 0.083333 |
| stability | 0.951911 | 0.789956 | -0.161955 |
| sparsity | 0.793694 | 0.793694 | 0.000000 |
| inference_runtime_seconds | 0.005550 | 0.003302 | -0.002248 |

## 13. Interpretation

Holding PGExplainer training-node and validation-node identities fixed across
backbones reduces the training-pool membership confound and provides a more
controlled cross-backbone sensitivity comparison. This does not prove a causal
backbone effect because the backbones still differ in architecture,
representations, predictive competence, and optimization history.

The strong GraphSAGE PGExplainer result persisted under the reduced common
training pool, suggesting that the original result was not solely attributable
to the larger backbone-specific PGExplainer training pool. The small decreases
in AP and stability are descriptive sensitivity signals, not failure cases and
not grounds for tuning.

## 14. Limitations

The final 12 BA-House targets had already been exposed before this task. The
analysis is therefore secondary and descriptive.

## 15. Implications for Task 20

Task 20 should keep three layers separate: primary within-backbone explainer
comparison, controlled PG cross-backbone sensitivity, and GraphSAGE PG
training-pool sensitivity.

## 16. Implications for ShapeGGen

For ShapeGGen, common cross-backbone PG train/validation pools should be
frozen before explainer training and before final explanation evaluation. This
prevents repeating the BA-House training-pool confound.

## 17. Freeze Statement

The original Task 19 artifacts remain unchanged. No GCN PG retraining,
hyperparameter tuning, best-seed selection, target modification, or Task 20
analysis was performed in this task.
