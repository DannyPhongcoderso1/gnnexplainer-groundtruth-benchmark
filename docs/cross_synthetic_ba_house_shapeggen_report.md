# Cross-Synthetic BA-House And ShapeGGen Report

## Scope

This report compares conclusion patterns between BA-House and ShapeGGen. It reuses frozen BA-House Task20, ShapeGGen Task23, and ShapeGGen Task24 evidence only. It does not concatenate per-node rows across datasets or report pooled cross-dataset AUROC/AP as a main conclusion.

## Dataset Context

| Dataset   | Field                                | Value                                                                                             |
| --- | --- | --- |
| BA-House  | nodes/edges/features/classes         | 500 nodes, 3398 directed edge entries, 10 features, classes {'0': 300, '1': 80, '2': 80, '3': 40} |
| BA-House  | model split counts                   | {'test': 100, 'train': 300, 'validation': 100}                                                    |
| BA-House  | common final target count            | 12                                                                                                |
| BA-House  | COMMON PG train/val counts           | 48/16                                                                                             |
| BA-House  | candidate-edge count distribution    | n=12, mean=221.166667, median=89.000000, min=24.000000, max=866.000000                            |
| BA-House  | GT-positive prevalence distribution  | n=12, mean=0.241228, median=0.218137, min=0.013857, max=0.500000                                  |
| BA-House  | GCN/GraphSAGE predictive performance | GCN test accuracy 0.620000; GraphSAGE test accuracy 0.960000                                      |
| ShapeGGen | nodes/edges/features/classes         | 1298 nodes, 4518 directed edge entries, 10 features, classes {0: 502, 1: 796}                     |
| ShapeGGen | model split counts                   | {'test': 261, 'train': 778, 'validation': 259}                                                    |
| ShapeGGen | common final target count            | 167                                                                                               |
| ShapeGGen | COMMON PG train/val counts           | 539/182                                                                                           |
| ShapeGGen | candidate-edge count distribution    | n=167, mean=123.185629, median=118.000000, min=30.000000, max=278.000000                          |
| ShapeGGen | GT-positive prevalence distribution  | n=167, mean=0.203649, median=0.197183, min=0.067416, max=0.466667                                 |
| ShapeGGen | GCN/GraphSAGE predictive performance | GCN test accuracy 0.681992; GraphSAGE test accuracy 0.869732                                      |

The dataset context explains why raw metric magnitudes are not direct difficulty scores. AP is sensitive to ground-truth-positive prevalence, candidate subgraph sizes differ, and backbone predictive performance differs.

## Frozen Metric Snapshot

| Dataset   | Backbone  | Explainer                 | AUROC    | AP       | Fid+     | Fid-     | 1-Fid-   | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA-House  | GCN       | GNNExplainer              | 0.870617 | 0.527731 | 0.666667 | 0.250000 | 0.750000 | 0.441272  |
| BA-House  | GCN       | PGExplainer               | 0.747969 | 0.366563 | 0.416667 | 0.583333 | 0.416667 | 0.951911  |
| BA-House  | GraphSAGE | GNNExplainer              | 0.895612 | 0.594340 | 0.916667 | 0.500000 | 0.500000 | 0.479613  |
| BA-House  | GraphSAGE | PGExplainer natural106/35 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 0.500000 | 0.805056  |
| ShapeGGen | GCN       | GNNExplainer              | 0.648445 | 0.324281 | 0.562874 | 0.011976 | 0.988024 | 0.367797  |
| ShapeGGen | GCN       | PGExplainer               | 0.385184 | 0.182545 | 0.041916 | 0.203593 | 0.796407 | 0.827289  |
| ShapeGGen | GraphSAGE | GNNExplainer              | 0.631585 | 0.297549 | 0.497006 | 0.257485 | 0.742515 | 0.321343  |
| ShapeGGen | GraphSAGE | PGExplainer               | 0.530641 | 0.285020 | 0.622754 | 0.610778 | 0.389222 | 0.793581  |

## BA-House Evidence Hierarchy

BA-House original primary evidence uses natural PG pools: GCN PGExplainer uses 48/16 and GraphSAGE PGExplainer uses 106/35. Task19B is a post-hoc controlled common-pool sensitivity analysis with 48/16 for both backbones. Task19B does not replace the original frozen BA-House primary result.

| Metric                    | GCN common 48/16 | GraphSAGE common 48/16 | Delta GraphSAGE - GCN |
| --- | --- | --- | --- |
| edge_auroc                | 0.747969         | 0.999869               | 0.251900              |
| edge_ap                   | 0.366563         | 0.993590               | 0.627027              |
| fid_plus                  | 0.416667         | 1.000000               | 0.583333              |
| fid_minus                 | 0.583333         | 0.500000               | -0.083333             |
| one_minus_fid_minus       | 0.416667         | 0.500000               | 0.083333              |
| stability                 | 0.951911         | 0.789956               | -0.161955             |
| sparsity                  | 0.793694         | 0.793694               | 0.000000              |
| inference_runtime_seconds | 0.005550         | 0.003302               | -0.002248             |

## Conclusion-Consistency Matrix

| Question / metric family                            | BA-House observed direction/pattern                                                               | ShapeGGen observed direction/pattern                                                              | Consistent? | Methodological notes                                                                                         |
| --- | --- | --- | --- | --- |
| GNN vs PG with GCN fixed                            | GNNExplainer higher AUROC/AP and stronger Fid+/1-Fid-; PGExplainer more stable.                   | GNNExplainer higher AUROC/AP and stronger Fid+/1-Fid-; PGExplainer more stable.                   | yes         | Primary within-backbone comparison in both datasets.                                                         |
| GNN vs PG with GraphSAGE fixed                      | PGExplainer higher AUROC/AP; Fid+ strong for both; PGExplainer more stable.                       | GNNExplainer higher AUROC/AP; PGExplainer higher Fid+ but lower 1-Fid-; PGExplainer more stable.  | mixed       | Primary within-backbone comparison; BA-House natural-pool PG result remains frozen with sensitivity context. |
| Stability ordering                                  | PGExplainer above GNNExplainer, both above Random.                                                | PGExplainer above GNNExplainer, both above Random.                                                | yes         | Same Jaccard summary, method-specific stochastic sources.                                                    |
| GT agreement vs perturbation-based metric alignment | Higher AUROC/AP does not always imply uniformly better Fid+/Fid-.                                 | Higher AUROC/AP does not always imply uniformly better Fid+/Fid-.                                 | yes         | Ranking agreement and perturbation-based model-behavior metrics answer distinct questions.                   |
| Runtime/quality trade-off                           | PGExplainer has upfront training and fast inference; quality ranking depends on backbone context. | PGExplainer has upfront training and fast inference; quality ranking depends on backbone context. | yes         | Runtime columns must remain decomposed.                                                                      |
| Descriptive GNN cross-backbone pattern              | GraphSAGE GNNExplainer higher AUROC/AP than GCN GNNExplainer.                                     | GCN GNNExplainer higher AUROC/AP than GraphSAGE GNNExplainer.                                     | mixed       | Secondary descriptive comparison only.                                                                       |
| Controlled PG cross-backbone pattern                | GraphSAGE common-pool PGExplainer higher AUROC/AP than GCN common-pool PGExplainer.               | GraphSAGE common-pool PGExplainer higher AUROC/AP than GCN common-pool PGExplainer.               | yes         | Better controlled by common PG train/validation identities, still non-causal.                                |

## Generalization Statement

The BA-House GCN-context pattern was reproduced on ShapeGGen: GNNExplainer had higher ground-truth agreement than PGExplainer while PGExplainer was more stable. The BA-House GraphSAGE-context ground-truth-agreement pattern was not reproduced: BA-House favored PGExplainer, whereas ShapeGGen favored GNNExplainer on AUROC/AP. Stability ordering was reproduced across the two synthetic contexts. Overall, relative explainer performance is consistent for some metric families and mixed for others under the frozen protocols.

## Methodological Guardrails

- Random remains outside the 2x2 matrices.
- Random aggregation uses node-level primary sampling after averaging raw seed rows within node.
- Stability uses the same Jaccard metric but different stochastic sources by method.
- Task19B common-pool result remains labeled as post-hoc sensitivity.
- ShapeGGen PG comparison uses prospectively frozen common PG train/validation identities.
- Cross-backbone observed differences are descriptive and non-causal.
- No cross-dataset row pooling is used as a primary analysis.

## Limitations

- BA-House has only 12 common final targets.
- ShapeGGen has 167 common final targets.
- Candidate-edge counts differ across datasets.
- Ground-truth-positive prevalence differs across datasets.
- Edge AP depends on prevalence.
- Topology, motif/config semantics, and class structure differ across datasets.
- Backbone predictive competence differs across backbone contexts.
- BA-House original PG pools are not balanced between backbones.
- Task19B is post-hoc sensitivity, not replacement evidence.
- ShapeGGen common PG pools were frozen prospectively.
- Synthetic ground-truth agreement does not prove real-world causal correctness.
- There is no universal explainer ranking.
- Node rows are not pooled between datasets.

## Stop Boundary

STOP FOR REVIEW. Suggested reviewed tag after approval: `synthetic-core-v1`.
