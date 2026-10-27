# ShapeGGen 2x2 Cross-Synthetic Freeze Report

## Scope

This report freezes the ShapeGGen 2x2 explainer comparison on the frozen 167-node final target set. It reuses Task23 and Task24 frozen outputs only. Random is kept outside the 2x2 matrix and is used as a sanity baseline.

## Protocol

- Edge AUROC and Edge AP use continuous scores over the full candidate-edge set.
- Fid+, Fid-, 1-Fid-, sparsity, and Jaccard stability use the frozen top-20% candidate-edge budget.
- Budget rule: `ceil(candidate_edge_count * 0.2), minimum 1`.
- Within-backbone explainer comparisons are primary.
- Cross-backbone observed differences are secondary/descriptive and non-causal.

## Ground-Truth Agreement And Perturbation-Based Metrics

| Backbone  | Explainer    | AUROC    | AP       | Fid+     | Fid-     | 1-Fid-   | Sparsity | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN       | GNNExplainer | 0.648445 | 0.324281 | 0.562874 | 0.011976 | 0.988024 | 0.795928 | 0.367797  |
| GCN       | PGExplainer  | 0.385184 | 0.182545 | 0.041916 | 0.203593 | 0.796407 | 0.795928 | 0.827289  |
| GraphSAGE | GNNExplainer | 0.631585 | 0.297549 | 0.497006 | 0.257485 | 0.742515 | 0.795928 | 0.321343  |
| GraphSAGE | PGExplainer  | 0.530641 | 0.285020 | 0.622754 | 0.610778 | 0.389222 | 0.795928 | 0.793581  |

## Full Metric Summary

This table reports `n`, mean, std, median, min, and max where applicable. Fid+ is prediction change after removing selected edges. Fid- is prediction change when retaining only selected edges. `1-Fid-` is `1 - Fid-`. `remove_probability_delta` is `original_probability - remove_probability`. `retain_probability_delta` is `original_probability - retain_probability`. `retain_probability_ratio` is `retain_probability / original_probability`. All probability diagnostics refer to the original predicted class.

| Backbone  | Explainer    | Metric                     | n   | mean      | std      | median    | min       | max      |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN       | GNNExplainer | edge_auroc                 | 167 | 0.648445  | 0.067154 | 0.647788  | 0.478571  | 0.850000 |
| GCN       | GNNExplainer | edge_ap                    | 167 | 0.324281  | 0.098408 | 0.316146  | 0.134013  | 0.639648 |
| GCN       | GNNExplainer | fid_plus                   | 167 | 0.562874  | 0.497523 | 1.000000  | 0.000000  | 1.000000 |
| GCN       | GNNExplainer | fid_minus                  | 167 | 0.011976  | 0.109105 | 0.000000  | 0.000000  | 1.000000 |
| GCN       | GNNExplainer | one_minus_fid_minus        | 167 | 0.988024  | 0.109105 | 1.000000  | 0.000000  | 1.000000 |
| GCN       | GNNExplainer | remove_probability_delta   | 167 | 0.246468  | 0.320759 | 0.268808  | -0.471428 | 0.900782 |
| GCN       | GNNExplainer | retain_probability         | 167 | 0.943223  | 0.109289 | 0.978325  | 0.017170  | 0.999889 |
| GCN       | GNNExplainer | retain_probability_delta   | 167 | -0.225394 | 0.136997 | -0.223372 | -0.481501 | 0.640771 |
| GCN       | GNNExplainer | retain_probability_ratio   | 167 | 1.345208  | 0.253414 | 1.303052  | 0.026097  | 1.931069 |
| GCN       | GNNExplainer | sparsity                   | 167 | 0.795928  | 0.003666 | 0.796512  | 0.777778  | 0.800000 |
| GCN       | GNNExplainer | generation_runtime_seconds | 167 | 0.048118  | 0.026834 | 0.044654  | 0.035499  | 0.363731 |
| GCN       | GNNExplainer | evaluation_runtime_seconds | 167 | 0.014312  | 0.004095 | 0.013103  | 0.009433  | 0.027105 |
| GCN       | PGExplainer  | edge_auroc                 | 167 | 0.385184  | 0.122787 | 0.385264  | 0.122449  | 0.693910 |
| GCN       | PGExplainer  | edge_ap                    | 167 | 0.182545  | 0.081853 | 0.165002  | 0.060266  | 0.582533 |
| GCN       | PGExplainer  | fid_plus                   | 167 | 0.041916  | 0.201000 | 0.000000  | 0.000000  | 1.000000 |
| GCN       | PGExplainer  | fid_minus                  | 167 | 0.203593  | 0.403881 | 0.000000  | 0.000000  | 1.000000 |
| GCN       | PGExplainer  | one_minus_fid_minus        | 167 | 0.796407  | 0.403881 | 1.000000  | 0.000000  | 1.000000 |
| GCN       | PGExplainer  | remove_probability_delta   | 167 | -0.034709 | 0.110989 | -0.021058 | -0.446880 | 0.267613 |
| GCN       | PGExplainer  | retain_probability         | 167 | 0.744318  | 0.300440 | 0.914573  | 0.006327  | 0.999999 |
| GCN       | PGExplainer  | retain_probability_delta   | 167 | -0.026488 | 0.310017 | -0.116261 | -0.498844 | 0.806566 |
| GCN       | PGExplainer  | retain_probability_ratio   | 167 | 1.058667  | 0.449181 | 1.162528  | 0.008202  | 1.997134 |
| GCN       | PGExplainer  | sparsity                   | 167 | 0.795928  | 0.003666 | 0.796512  | 0.777778  | 0.800000 |
| GCN       | PGExplainer  | generation_runtime_seconds | 0   | NA        | NA       | NA        | NA        | NA       |
| GCN       | PGExplainer  | inference_runtime_seconds  | 167 | 0.007096  | 0.002735 | 0.006601  | 0.003587  | 0.017802 |
| GCN       | PGExplainer  | evaluation_runtime_seconds | 167 | 0.014371  | 0.004535 | 0.013478  | 0.008982  | 0.033655 |
| GraphSAGE | GNNExplainer | edge_auroc                 | 167 | 0.631585  | 0.060833 | 0.626414  | 0.461905  | 0.780208 |
| GraphSAGE | GNNExplainer | edge_ap                    | 167 | 0.297549  | 0.100516 | 0.289933  | 0.102859  | 0.671247 |
| GraphSAGE | GNNExplainer | fid_plus                   | 167 | 0.497006  | 0.501495 | 0.000000  | 0.000000  | 1.000000 |
| GraphSAGE | GNNExplainer | fid_minus                  | 167 | 0.257485  | 0.438564 | 0.000000  | 0.000000  | 1.000000 |
| GraphSAGE | GNNExplainer | one_minus_fid_minus        | 167 | 0.742515  | 0.438564 | 1.000000  | 0.000000  | 1.000000 |
| GraphSAGE | GNNExplainer | remove_probability_delta   | 167 | 0.453402  | 0.405311 | 0.465970  | -0.341256 | 0.998674 |
| GraphSAGE | GNNExplainer | retain_probability         | 167 | 0.728904  | 0.359180 | 0.938372  | 0.000180  | 1.000000 |
| GraphSAGE | GNNExplainer | retain_probability_delta   | 167 | 0.230430  | 0.344820 | 0.048583  | -0.355926 | 0.998374 |
| GraphSAGE | GNNExplainer | retain_probability_ratio   | 167 | 0.755205  | 0.372738 | 0.951413  | 0.000197  | 1.553856 |
| GraphSAGE | GNNExplainer | sparsity                   | 167 | 0.795928  | 0.003666 | 0.796512  | 0.777778  | 0.800000 |
| GraphSAGE | GNNExplainer | generation_runtime_seconds | 167 | 0.031272  | 0.007081 | 0.030317  | 0.023372  | 0.107734 |
| GraphSAGE | GNNExplainer | evaluation_runtime_seconds | 167 | 0.009278  | 0.002622 | 0.008563  | 0.005888  | 0.019402 |
| GraphSAGE | PGExplainer  | edge_auroc                 | 167 | 0.530641  | 0.101259 | 0.517290  | 0.293478  | 0.817227 |
| GraphSAGE | PGExplainer  | edge_ap                    | 167 | 0.285020  | 0.084921 | 0.284956  | 0.089552  | 0.615687 |
| GraphSAGE | PGExplainer  | fid_plus                   | 167 | 0.622754  | 0.486155 | 1.000000  | 0.000000  | 1.000000 |
| GraphSAGE | PGExplainer  | fid_minus                  | 167 | 0.610778  | 0.489040 | 1.000000  | 0.000000  | 1.000000 |
| GraphSAGE | PGExplainer  | one_minus_fid_minus        | 167 | 0.389222  | 0.489040 | 0.000000  | 0.000000  | 1.000000 |
| GraphSAGE | PGExplainer  | remove_probability_delta   | 167 | 0.536464  | 0.438656 | 0.680050  | -0.354829 | 0.999463 |
| GraphSAGE | PGExplainer  | retain_probability         | 167 | 0.404359  | 0.461939 | 0.070283  | 0.000000  | 1.000000 |
| GraphSAGE | PGExplainer  | retain_probability_delta   | 167 | 0.554974  | 0.457843 | 0.798713  | -0.356488 | 1.000000 |
| GraphSAGE | PGExplainer  | retain_probability_ratio   | 167 | 0.418035  | 0.482181 | 0.075224  | 0.000000  | 1.554731 |
| GraphSAGE | PGExplainer  | sparsity                   | 167 | 0.795928  | 0.003666 | 0.796512  | 0.777778  | 0.800000 |
| GraphSAGE | PGExplainer  | generation_runtime_seconds | 0   | NA        | NA       | NA        | NA        | NA       |
| GraphSAGE | PGExplainer  | inference_runtime_seconds  | 167 | 0.003956  | 0.001494 | 0.003507  | 0.002336  | 0.009421 |
| GraphSAGE | PGExplainer  | evaluation_runtime_seconds | 167 | 0.008129  | 0.002458 | 0.007327  | 0.005722  | 0.018820 |
| GCN       | GNNExplainer | jaccard_stability          | 167 | 0.367797  | NA       | NA        | NA        | NA       |
| GCN       | PGExplainer  | jaccard_stability          | 167 | 0.827289  | NA       | NA        | NA        | NA       |
| GraphSAGE | GNNExplainer | jaccard_stability          | 167 | 0.321343  | NA       | NA        | NA        | NA       |
| GraphSAGE | PGExplainer  | jaccard_stability          | 167 | 0.793581  | NA       | NA        | NA        | NA       |

## Efficiency

Runtime semantics are decomposed. GNNExplainer reports per-target explanation-generation optimization time. PGExplainer reports upfront explainer training, then per-target inference and evaluation runtime. Random reports random edge-selection plus evaluation runtime and is not part of this 2x2 table.

| Backbone  | Explainer    | Training   | Generation | Inference | Evaluation | Semantics                                                                                      |
| --- | --- | --- | --- | --- | --- | --- |
| GCN       | GNNExplainer | NA         | 0.048118   | NA        | 0.014312   | per-target GNNExplainer optimization/generation plus metric evaluation                         |
| GCN       | PGExplainer  | 180.965073 | NA         | 0.007096  | 0.014371   | upfront PGExplainer training plus per-target trained-explainer inference and metric evaluation |
| GraphSAGE | GNNExplainer | NA         | 0.031272   | NA        | 0.009278   | per-target GNNExplainer optimization/generation plus metric evaluation                         |
| GraphSAGE | PGExplainer  | 110.941023 | NA         | 0.003956  | 0.008129   | upfront PGExplainer training plus per-target trained-explainer inference and metric evaluation |

## Stability

The same Jaccard metric is used, but stochastic sources differ: GNNExplainer uses independent optimization seeds, PGExplainer uses independently trained checkpoints, and Random uses repeated random selections.

## Random Sanity Baseline

Random remains outside the ShapeGGen 2x2 matrix. Its main quality aggregation is node-level: raw seed rows are averaged within each node, then summarized across nodes.

| Backbone  | Metric                   | n   | mean      | std      | median    | min       | max      |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GCN       | edge_auroc               | 167 | 0.502273  | 0.026172 | 0.503162  | 0.387395  | 0.570378 |
| GCN       | edge_ap                  | 167 | 0.237006  | 0.078723 | 0.231524  | 0.084175  | 0.507570 |
| GCN       | fid_plus                 | 167 | 0.127545  | 0.171691 | 0.000000  | 0.000000  | 0.700000 |
| GCN       | fid_minus                | 167 | 0.239521  | 0.239467 | 0.200000  | 0.000000  | 1.000000 |
| GCN       | one_minus_fid_minus      | 167 | 0.760479  | 0.239467 | 0.800000  | 0.000000  | 1.000000 |
| GCN       | remove_probability_delta | 167 | 0.002619  | 0.066535 | 0.001556  | -0.178481 | 0.251788 |
| GCN       | retain_probability       | 167 | 0.720531  | 0.195719 | 0.752150  | 0.189571  | 0.998389 |
| GCN       | retain_probability_delta | 167 | -0.002701 | 0.205380 | -0.031583 | -0.448721 | 0.470912 |
| GCN       | sparsity                 | 167 | 0.795928  | 0.003666 | 0.796512  | 0.777778  | 0.800000 |
| GraphSAGE | edge_auroc               | 167 | 0.502273  | 0.026172 | 0.503162  | 0.387395  | 0.570378 |
| GraphSAGE | edge_ap                  | 167 | 0.237006  | 0.078723 | 0.231524  | 0.084175  | 0.507570 |
| GraphSAGE | fid_plus                 | 167 | 0.128743  | 0.237989 | 0.000000  | 0.000000  | 1.000000 |
| GraphSAGE | fid_minus                | 167 | 0.585629  | 0.461177 | 0.900000  | 0.000000  | 1.000000 |
| GraphSAGE | one_minus_fid_minus      | 167 | 0.414371  | 0.461177 | 0.100000  | 0.000000  | 1.000000 |
| GraphSAGE | remove_probability_delta | 167 | 0.115199  | 0.176748 | 0.035028  | -0.166944 | 0.840878 |
| GraphSAGE | retain_probability       | 167 | 0.453841  | 0.424049 | 0.275200  | 0.000001  | 1.000000 |
| GraphSAGE | retain_probability_delta | 167 | 0.505493  | 0.418558 | 0.650989  | -0.352139 | 0.998800 |
| GraphSAGE | sparsity                 | 167 | 0.795928  | 0.003666 | 0.796512  | 0.777778  | 0.800000 |

## PG Cross-Backbone Common-Pool Table

Task22 and Task24 prospectively froze common PG train/validation identities for ShapeGGen, so this table is better controlled than a natural-pool comparison. It is still a cross-backbone observed difference, not a causal claim.

| Backbone  | Common train | Common validation | Final targets | AUROC    | AP       | Fid+     | Fid-     | 1-Fid-   | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN       | 539          | 182               | 167           | 0.385184 | 0.182545 | 0.041916 | 0.203593 | 0.796407 | 0.827289  |
| GraphSAGE | 539          | 182               | 167           | 0.530641 | 0.285020 | 0.622754 | 0.610778 | 0.389222 | 0.793581  |

## Per-Node Paired Matrix

The node-keyed paired matrix is stored at `results\metrics\shapeggen_2x2\paired_matrix.csv`. It contains all four ShapeGGen 2x2 cells keyed by `node_id`.

## Main Finding

On ShapeGGen, GNNExplainer has higher Edge AUROC/AP than PGExplainer with the GCN backbone fixed. With GraphSAGE fixed, GNNExplainer also has higher Edge AUROC/AP than PGExplainer, while PGExplainer has stronger Fid+ but weaker Fid-/1-Fid-. PGExplainer is more stable than GNNExplainer for both backbones under the frozen protocols.

## Limitations

Absolute metric magnitudes should be read with dataset context. AP depends on ground-truth prevalence, and classifier competence differs across backbone contexts. This report does not pool node rows across BA-House and ShapeGGen.

## Outputs

- `results\metrics\shapeggen_2x2\main.csv`
- `results\metrics\shapeggen_2x2\ground_truth_agreement.csv`
- `results\metrics\shapeggen_2x2\perturbation_metrics.csv`
- `results\metrics\shapeggen_2x2\metric_summary.csv`
- `results\metrics\shapeggen_2x2\stability.csv`
- `results\metrics\shapeggen_2x2\runtime.csv`
- `results\metrics\shapeggen_2x2\paired_matrix.csv`
- `results\metrics\shapeggen_2x2\random_node_level_summary.csv`
- `results\metrics\shapeggen_2x2\pg_common_pool.csv`
- `results\metrics\cross_synthetic\ba_house_shapeggen_metric_snapshot.csv`
- `results\metrics\cross_synthetic\dataset_context.csv`
- `results\metrics\cross_synthetic\conclusion_consistency_matrix.csv`
- `results\metrics\cross_synthetic\generated_csv_inventory.csv`
- `results\metrics\cross_synthetic\stability_pair_count_audit.csv`

## Generated CSV Audit

Generated CSV inventory is stored at `results\metrics\cross_synthetic\generated_csv_inventory.csv`. Stability pair-count audit is stored at `results\metrics\cross_synthetic\stability_pair_count_audit.csv`. All generated CSV files are deterministic from frozen Task20/23/24 sources and remain ignored by the repository results policy.

## Stop Boundary

STOP FOR REVIEW. Do not start Task26 automatically.
