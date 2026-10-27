# GraphSAGE Explainers BA-House Freeze

## Scope

Task 19 evaluates GNNExplainer and PGExplainer on the frozen GraphSAGE
BA-House backbone. The BA-House 2x2 target intersection was frozen before any
GraphSAGE explanation results were used.

## Frozen Inputs

- Dataset SHA256: `4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94`
- GraphSAGE checkpoint SHA256: `09992c5ac4d2cd80c10792c12be5f945b3a20d749d8433008eba9cf3694135eb`
- 2x2 targets: `12` nodes
- 2x2 target SHA256: `2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f`
- Selected-edge budget: fixed candidate-edge ratio `0.2`
- Edge ranking metrics: continuous scores over the full candidate-edge set

## Target And Leakage Checks

- PG train targets: `106`
- PG validation targets: `35`
- Final targets used for PG training: `0`
- PG seed42 validation health success:
  `35/35`
- GraphSAGE weights unchanged during PG training:
  `True`

## Main Quality

| Method | AUROC | AP | Fid+ | Fid- | 1-Fid- | Remove delta | Sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|
| GNNExplainer | 0.895612 | 0.594340 | 0.916667 | 0.500000 | 0.500000 | 0.412075 | 0.793694 |
| PGExplainer | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 0.500000 | 0.425826 | 0.793694 |

## Stability

| Method | Seeds | Pairwise comparisons/node | Mean Jaccard | Std | Median |
|---|---:|---:|---:|---:|---:|
| GNNExplainer | 5 | 10 | 0.479613 | 0.258427 | 0.421404 |
| PGExplainer | 5 | 10 | 0.805056 | 0.118526 | 0.830964 |

## Runtime

Runtime is decomposed by method and stage. GNNExplainer reports per-target
optimization/generation time. PGExplainer reports upfront training per
checkpoint, trained-explainer inference per target, and evaluation batch time.

| Runtime | Mean seconds | Std | Median |
|---|---:|---:|---:|
| GNNExplainer generation/target | 0.129616 | 0.188752 | 0.077888 |
| PGExplainer seed42 training | 44.858333 | N/A | N/A |
| PGExplainer stability training/checkpoint | 56.925291 | 3.988295 | 57.442397 |
| PGExplainer inference/target | 0.014336 | 0.031808 | 0.005216 |
| PGExplainer stability evaluation batch | 0.429165 | 0.018266 | 0.420761 |

## Freeze Statement

The GraphSAGE explainer runs are frozen with no target selection, best-seed
selection, or hyperparameter tuning after final metrics. Task 20 conclusions
were not created in this task.
