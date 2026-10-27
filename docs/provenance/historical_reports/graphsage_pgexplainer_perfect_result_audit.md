# GraphSAGE PGExplainer Perfect-Result Audit

## 1. Scope

This is a read-only methodological audit before Task 20. It inspects frozen
Task 19 artifacts only and does not retrain, rerun explainers, modify targets,
change metrics, change the 20% budget, or tune hyperparameters.

## 2. Frozen Inputs

- GraphSAGE PG seed42 per-node metrics:
  `results/metrics/graphsage_pgexplainer/seed42/per_node_metrics.csv`
- GraphSAGE PG seed42 raw explanations:
  `results/explanations/graphsage_pgexplainer/seed42`
- Freeze manifest:
  `docs/graphsage_explainers_ba_house_freeze.json`
- 2x2 target SHA256:
  `2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f`

## 3. Per-Node Perfect-Metric Audit

| node_id | candidate | positive GT | negative GT | selected | AUROC | AP | Fid+ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 306 | 330 | 12 | 318 | 66 | 1.000000 | 1.000000 | 1 |
| 315 | 866 | 12 | 854 | 174 | 1.000000 | 1.000000 | 1 |
| 316 | 34 | 12 | 22 | 7 | 1.000000 | 1.000000 | 1 |
| 355 | 30 | 12 | 18 | 6 | 1.000000 | 1.000000 | 1 |
| 365 | 26 | 12 | 14 | 6 | 1.000000 | 1.000000 | 1 |
| 375 | 28 | 12 | 16 | 6 | 1.000000 | 1.000000 | 1 |
| 376 | 538 | 12 | 526 | 108 | 1.000000 | 1.000000 | 1 |
| 411 | 176 | 12 | 164 | 36 | 1.000000 | 1.000000 | 1 |
| 445 | 24 | 12 | 12 | 5 | 1.000000 | 1.000000 | 1 |
| 465 | 434 | 12 | 422 | 87 | 1.000000 | 1.000000 | 1 |
| 486 | 144 | 12 | 132 | 29 | 1.000000 | 1.000000 | 1 |
| 491 | 24 | 12 | 12 | 5 | 1.000000 | 1.000000 | 1 |

- AUROC = 1.0: `12/12`
- AP = 1.0: `12/12`
- Fid+ = 1: `12/12`

## 4. Metric-Scope Verification

AUROC/AP are computed from continuous PGExplainer edge scores over the full
candidate-edge set. The implementation uses full-edge-aligned `edge_scores`,
`ground_truth_edge_mask`, and `candidate_edge_mask`, then indexes scores and
labels with the same candidate mask before calling AUROC/AP.

- `edge_ranking_metric_scope`:
  `continuous_scores_over_full_candidate_edge_set`
- AUROC/AP use full candidate scores:
  `True`

## 5. Candidate/GT Mapping Verification

- Candidate-edge count parity:
  `12/12`
- Positive GT count parity:
  `12/12`
- Negative GT count parity:
  `12/12`
- Edge score vector full-edge aligned:
  `True`
- Selected edges all in candidate set:
  `True`
- Off-by-one/sorting mismatch detected:
  `False`
- Reverse-edge mismatch detected:
  `False`

## 6. GT Leakage Audit

- GT used in PG training loss:
  `False`
- Score equals GT mask detected:
  `False`
- Score multiplied by GT mask detected:
  `False`
- Candidate edges restricted to GT positives detected:
  `False`

Evidence: PG training calls `algorithm.train(..., target=frozen_predictions,
index=target.node_id)`. GT masks are used after explanation generation for
evaluation/validation-health metrics, not as PG training labels.

## 7. Score Separation Audit

| node_id | min positive score | max negative score | margin |
|---:|---:|---:|---:|
| 306 | 1.000000 | 0.000000 | 1.000000 |
| 315 | 1.000000 | 0.000000 | 1.000000 |
| 316 | 1.000000 | 0.000000 | 1.000000 |
| 355 | 1.000000 | 0.000000 | 1.000000 |
| 365 | 1.000000 | 0.000000 | 1.000000 |
| 375 | 1.000000 | 0.000000 | 1.000000 |
| 376 | 1.000000 | 0.000000 | 1.000000 |
| 411 | 1.000000 | 0.000000 | 1.000000 |
| 445 | 1.000000 | 0.000000 | 1.000000 |
| 465 | 1.000000 | 0.000000 | 1.000000 |
| 486 | 0.000225 | 0.000000 | 0.000225 |
| 491 | 1.000000 | 0.000000 | 1.000000 |

- Minimum positive-vs-negative margin:
  `0.000225`
- All positive scores above all negative scores:
  `True`

## 8. Permutation Sanity Check

Stored score vectors were permuted against the same GT labels with audit seed
`20260816` and `1000` permutations per
node. This is a metric sanity check only, not a model/explainer rerun.

- Permuted AUROC mean:
  `0.499596`
- Permuted AUROC std:
  `0.074689`
- Permuted AP mean:
  `0.249568`
- Permuted AP std:
  `0.211534`

Interpretation: Permuting stored scores destroys the perfect ranking and yields near-chance AUROC with AP near prevalence-driven random levels.

## 9. Fid+ Audit

| node_id | original prediction | remove prediction | changed | remove delta |
|---:|---:|---:|---|---:|
| 306 | 1 | 2 | True | 0.540932 |
| 315 | 1 | 2 | True | 0.353189 |
| 316 | 1 | 2 | True | 0.503685 |
| 355 | 1 | 2 | True | 0.381616 |
| 365 | 1 | 2 | True | 0.381616 |
| 375 | 1 | 2 | True | 0.434535 |
| 376 | 1 | 2 | True | 0.504723 |
| 411 | 1 | 2 | True | 0.573621 |
| 445 | 1 | 2 | True | 0.101164 |
| 465 | 1 | 2 | True | 0.525185 |
| 486 | 1 | 2 | True | 0.440731 |
| 491 | 1 | 2 | True | 0.368913 |

- Fid+ rows with changed prediction:
  `12/12`

## 10. GCN vs GraphSAGE PG Training-Pool Audit

- GCN train file:
  `configs/targets/pgexplainer_train_targets.csv`
- GCN train SHA256:
  `05e4574515fb7a03b846adb677295c940eb7af380216c15dfedf65bb9430bc3c`
- GraphSAGE train file:
  `configs/targets/pgexplainer_graphsage_train_targets.csv`
- GraphSAGE train SHA256:
  `5e7efee3dfd90cc6a5bcd547d23352b27344fe313467ea48f502b13be6160284`
- GCN validation file:
  `configs/targets/pgexplainer_val_targets.csv`
- GraphSAGE validation file:
  `configs/targets/pgexplainer_graphsage_val_targets.csv`

Eligibility rule: base split membership AND node_mask/GT-explanation eligible AND backbone prediction correct AND candidate-edge mapping valid.

Train pools:

- `|GCN_train| = 48`
- `|GraphSAGE_train| = 106`
- `|intersection| = 48`
- `|GCN only| = 0`
- `|GraphSAGE only| = 58`
- `GCN_train subset GraphSAGE_train = True`

Validation pools:

- `|GCN_val| = 16`
- `|GraphSAGE_val| = 35`
- `|intersection| = 16`
- `|GCN only| = 0`
- `|GraphSAGE only| = 19`

## 11. Common-Pool Intersection

- `COMMON_train = 48`
- `COMMON_val = 16`
- Large enough if frozen before final exposure:
  `True`

## 12. Implications for Task 20

- Within-backbone GNNExplainer vs PGExplainer:
  `PRIMARY / CLEAN`
- GNNExplainer GCN vs GraphSAGE:
  `SECONDARY BUT STRONGLY CONTROLLED`
- PGExplainer GCN vs GraphSAGE:
  `SECONDARY / DESCRIPTIVE because training-pool membership differs`

Current frozen design answers how PGExplainer behaves when trained naturally
against each frozen backbone under the same eligibility rule. A common-pool
design would answer how PGExplainer changes when training-node identities are
held fixed and only the backbone changes. Neither design is universally
correct, but cross-backbone PG interpretation should be descriptive because
pool membership differs.

## 13. Freeze / No-Rerun Statement

No final explanations were regenerated, no PGExplainer was retrained, no target
files or checkpoints were modified, and no hyperparameters or metric definitions
were changed. Task 20 was not started.
