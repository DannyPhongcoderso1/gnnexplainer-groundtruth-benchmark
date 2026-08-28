# Elliptic Task26 Final Audit Report

## Scope And Boundary

Task26 freezes Elliptic handoff evidence for later real-world XAI. This audit is documentation and deterministic validation only. It did not retrain GCN, retrain GraphSAGE, train GNNExplainer, train PGExplainer, generate final Elliptic explanations, calculate explanation Edge AUROC/AP, change targets, or start Task27.

## Cross-Repository Provenance

| Role | Path | Remote | Branch | Commit | Tag | Start status |
| --- | --- | --- | --- | --- | --- | --- |
| benchmark | D:\CS\KLTN\gnnexplainer-groundtruth-benchmark | origin	https://github.com/DannyPhongcoderso1/gnnexplainer-groundtruth-benchmark.git (fetch)<br>origin	https://github.com/DannyPhongcoderso1/gnnexplainer-groundtruth-benchmark.git (push) | main | 7d1d1980f5411b4807ff868424360a74774c44c2 | none | clean |
| baseline | D:\CS\KLTN\elliptic-gnn-xai | origin	https://github.com/DannyPhongcoderso1/elliptic-graphfraud-xai.git (fetch)<br>origin	https://github.com/DannyPhongcoderso1/elliptic-graphfraud-xai.git (push) | migration/validated-baselines | 95710a9852019104f90499cdba628c37f4483a2a | baseline-v1.0 | M README.md |

The baseline repository has a pre-existing tracked `README.md` modification. It is unrelated to Task26 because model execution and target selection use `datasets/elliptic/elliptic_data.pt`, checkpoint files, YAML configs, baseline model source files, and the benchmark wrapper script. The baseline README is not loaded by the audit or parity code.

During report generation the benchmark worktree naturally contains the in-progress audit files. The durable start state remains the clean committed Task26 handoff at `7d1d1980f5411b4807ff868424360a74774c44c2`.

## Frozen Source Hashes

| Artifact | Path | SHA256 |
| --- | --- | --- |
| processed_elliptic_data | D:\CS\KLTN\elliptic-gnn-xai\datasets\elliptic\elliptic_data.pt | f807b83a7c739b16f69b5676226ce441b64d5fe8de95506a7727f86e8451291b |
| gcn_checkpoint | D:\CS\KLTN\elliptic-gnn-xai\artifacts\checkpoints\gcn_baseline.pt | 55004ffa1b712ae374fbf630d55f31ac69602d696dbeef8cb150d3f788a9cc07 |
| graphsage_checkpoint | D:\CS\KLTN\elliptic-gnn-xai\artifacts\checkpoints\graphsage_baseline.pt | f25d5cbeb10f9a27dbba1adcb2fdb21b406219956c107f8d51e8ecf5188866bf |
| gcn_training_config | D:\CS\KLTN\elliptic-gnn-xai\config\training_baseline_paper_gcn.yaml | db271b1485f7f92d6e4a630394fe709af2f77f9e6390b06d8a1bcc65b675a368 |
| graphsage_training_config | D:\CS\KLTN\elliptic-gnn-xai\config\training_baseline_paper_sage.yaml | 2118d7836632468ce20cfd1b34a7dc6507199db1356320d71f0f7f0d53fbaf2a |
| gcn_model_config | D:\CS\KLTN\elliptic-gnn-xai\config\model_gcn.yaml | 50d104a2120b1c23cf46e2df86778bf5fdbb344ec569cacdf854d10f14b0435e |
| graphsage_model_config | D:\CS\KLTN\elliptic-gnn-xai\config\model_sage.yaml | 00d5f066789632f9a7725f29dc972dd64c6cd2299f0f9cc9c3d06981d3d7018a |
| baseline_dataset_loader | D:\CS\KLTN\elliptic-gnn-xai\data\dataset.py | 4e3ea9c4169e9c452fdf8a2284756d751602c845e68ff52d87cda9d559b04481 |
| task26_wrapper_script | D:\CS\KLTN\gnnexplainer-groundtruth-benchmark\scripts\prepare_elliptic_task26.py | b4971537181e14d0ef91aa0e5e0425aabf74e24a5b3e12bb924fc1c88fe2f1ac |
| task26_test | D:\CS\KLTN\gnnexplainer-groundtruth-benchmark\tests\test_elliptic_task26_protocol.py | 8bab2c44303f607e261adc4f8d07bc02e8eac6dcea21d988c22487194f52fcda |

## Dataset And Temporal Semantics

| Statistic | Value |
| --- | --- |
| total nodes | 203769 |
| total directed edge entries | 468710 |
| feature count | 165 |
| known labels | 46564 |
| unknown labels | 157205 |
| label mapping | {'0': 'licit', '1': 'illicit', '2': 'unknown'} |
| positive semantic class | ILLICIT transaction |
| positive numeric label index | 1 |

| Split | Rule | Known rows | Timestep min | Timestep max | Label distribution |
| --- | --- | --- | --- | --- | --- |
| train | time_step < 30 and known label | 26381 | 1 | 29 | {'0': 23510, '1': 2871} |
| validation | 30 <= time_step < 40 and known label | 8999 | 30 | 39 | {'0': 7961, '1': 1038} |
| test | time_step >= 40 and known label | 11184 | 40 | 49 | {'0': 10548, '1': 636} |

Unknown-label nodes remain in the graph and may appear as neighbors in computation subgraphs, but unknown-label nodes are excluded from supervised/evaluation/final target files.

## Model Parity Audit

| Backbone | atol | rtol | logit shape | prob shape | max logit diff | max prob diff | mismatches | parity rate | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | 1e-06 | 0.0 | [203769, 2] | [203769, 2] | 0.0 | 0.0 | 0 | 1.0 | True |
| GraphSAGE | 1e-06 | 0.0 | [203769, 2] | [203769, 2] | 0.0 | 0.0 | 0 | 1.0 | True |

GCN baseline uses `GCNConv(cached=True)`. The explanation-compatible GCN uses the same weights with `GCNConv(cached=False)`, eval mode, and the same original graph inputs. GraphSAGE uses the same SAGEConv architecture with explicit `edge_index` input. No retraining was performed.

## Perturbation Smoke Audit

Smoke node: `157422`, true label `0`, candidate edges `12`, selected edges `3` at ratio `0.2`.

| Backbone | Pred original | Pred remove | Pred retain | Original prob | Remove prob | Retain prob | Max logit change | Max prob change | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | 0 | 0 | 0 | 0.9997366070747375 | 0.99997878074646 | 1.0 | 4.762311935424805 | 0.00026339292526245117 | True |
| GraphSAGE | 0 | 0 | 0 | 0.9999839067459106 | 0.9999940395355225 | 0.9999356269836426 | 0.8449053764343262 | 4.8279762268066406e-05 | True |

Edge input hashes prove the perturbation changed the tensor supplied to message passing:

| Edge input | Count | SHA256 |
| --- | --- | --- |
| original | 468710 | 5a97ef566847d206cd72a4a2bc0c914e744440056524f3e8ff5a09a0759c93bd |
| remove | 468707 | 35ce5e916e38b70038112aa2c78381d5b6027a13d18713f6d4470e4b49c1c6ea |
| retain | 3 | cf4210fdfc71efbcbc52ac5956a4929e46e5495e0dbceff54a0ceabdbf981297 |

## Eligibility Audit

| Split | GCN eligible | GraphSAGE eligible | Common intersection |
| --- | --- | --- | --- |
| train | 26381 | 26381 | 26381 |
| validation | 8999 | 8999 | 8999 |
| test | 11184 | 11184 | 11184 |

Eligibility uses known label, temporal split membership, valid candidate mapping, and computational usability. Correct classification is not required for Elliptic PG training eligibility.

## Final Target Freeze

The final common test target set is sampled from the common known-labeled test eligible population by deterministic seed `42`. The cap is `200`, chosen as a pre-final compute-budget sample size. Selection does not use explanation quality, stability, visualization, or final Task27 results.

| Field | Value |
| --- | --- |
| CSV | configs\targets\elliptic_xai_common_test_targets.csv |
| SHA256 | e414941e6943b06484ecd5028c51daf80ff2bf9f1bbb06e6a5fd816de8dfb9c0 |
| rows | 200 |
| unique nodes | 200 |
| duplicates | 0 |
| label distribution | {'0': 190, '1': 10} |
| timestep min/max | 40/49 |
| candidate-edge distribution | {'n': 200, 'mean': 126.56, 'std': 240.36773161137916, 'median': 27.0, 'min': 4, 'max': 1424} |
| GCN confusion | {'tp': 4, 'fp': 0, 'tn': 190, 'fn': 6, 'correct': 194, 'incorrect': 6} |
| GraphSAGE confusion | {'tp': 5, 'fp': 0, 'tn': 190, 'fn': 5, 'correct': 195, 'incorrect': 5} |

## Common PG Pool Freeze

COMMON_PG_TRAIN and COMMON_PG_VAL are deterministic samples from the common train/validation eligibility intersections. The caps are `2000` and `500`, recorded as pre-final compute-budget limits for PGExplainer training and validation. The final test targets and explanation quality are not used.

| Pool | CSV | Rows | Unique | Duplicates | SHA256 | Label distribution | Timestep min/max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | configs\targets\elliptic_pg_common_train.csv | 2000 | 2000 | 0 | 36c2f9b3a81336d304556702b02b6524411f308ee75f2152d79101586f9c1a22 | {'0': 1773, '1': 227} | 1/29 |
| validation | configs\targets\elliptic_pg_common_val.csv | 500 | 500 | 0 | 1a747030b9e903acd13d1ad22462cf7e3daf22c4f60fb475e421d2ecd87c8da6 | {'0': 449, '1': 51} | 30/39 |

Set intersections: `{'common_train_common_val': 0, 'common_train_final_test': 0, 'common_val_final_test': 0}`. PG train IDs and PG val IDs are identical for both backbones by using the common CSV files as the canonical pools.

## Metric Protocol Audit

Probability reference class is the original predicted class.

- Fid+ = `1` when removing selected edges changes the original prediction, else `0`.
- Fid- = `1` when retaining only selected edges changes the original prediction, else `0`.
- 1-Fid- = `1 - Fid-`.
- remove_probability_delta = `original_probability - remove_probability`.
- retain_probability = probability of the original predicted class on the retain-only graph.
- retain_probability_delta = `original_probability - retain_probability`.
- retain_probability_ratio = `retain_probability / original_probability`; if `original_probability == 0`, the ratio is recorded as null and the row is flagged because no epsilon/clipping is frozen.
- selected_edge_count = `max(1, ceil(candidate_edge_count * 0.20))`.
- sparsity = `1 - selected_edge_count / candidate_edge_count`.

Edge AUROC and Edge AP are explicitly disabled for Elliptic explanation evaluation because this thesis has class labels but no ground-truth explanatory edge masks for Elliptic. No explanatory edge ground truth may be manufactured from licit/illicit transaction labels.

Runtime remains decomposed into GNNExplainer per-instance generation, PGExplainer upfront training, PGExplainer per-target inference, metric evaluation, and Random selection/evaluation. Stability uses pairwise Jaccard over selected top-20% sets with method-specific stochastic sources.

## Artifact Inventory

| Path | Role | Rows | Size bytes | SHA256 | Duplicates | Nulls | NaN/Inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| docs/elliptic_xai_handoff_provenance.md | cross-repository provenance |  | 1890 | 451aff2423caaf397c345fa3e0fcb5017e7a850bad3be8dec7e9af53c3422048 |  |  |  |
| docs/elliptic_xai_protocol.md | Elliptic XAI metric and target protocol |  | 1905 | 9d9274b26fd5dcbd5a0e305729f6c2531959a54d93214b93e1a1e2678c61190c |  |  |  |
| docs/elliptic_model_parity_report.md | model parity and perturbation smoke report |  | 1401 | 2f46f7b3c3f1f7678b622bf36c3681d2a77f92b548083c88511832cf2be8d859 |  |  |  |
| docs/elliptic_targets_freeze.json | machine-readable freeze manifest |  | 51980 | cea6c0de2ed4662f16d967cecab65557fa2d7193f2b52bd35faac8bd3af38375 |  |  |  |
| scripts/prepare_elliptic_task26.py | Task26 target/provenance preparation script |  | 29653 | b4971537181e14d0ef91aa0e5e0425aabf74e24a5b3e12bb924fc1c88fe2f1ac |  |  |  |
| scripts/audit_elliptic_task26.py | Task26 final deterministic audit script |  | 44924 | a20e65fbcb3e4718d0f519cd5a74427cd0688075e6ffcba0b4c416a1b96686b4 |  |  |  |
| tests/test_elliptic_task26_protocol.py | Task26 protocol regression tests |  | 5331 | 8bab2c44303f607e261adc4f8d07bc02e8eac6dcea21d988c22487194f52fcda |  |  |  |
| configs\targets\elliptic_xai_common_test_targets.csv | Task26 final_test target file | 200 | 15546 | e414941e6943b06484ecd5028c51daf80ff2bf9f1bbb06e6a5fd816de8dfb9c0 | 0 | 0 | 0 |
| configs\targets\elliptic_pg_gcn_train_eligible.csv | Task26 gcn_train_eligible target file | 26381 | 1927240 | bfc68def29d51673f898ff1e67fac65cce615d495e23b8c00ec6a339e3173095 | 0 | 0 | 0 |
| configs\targets\elliptic_pg_gcn_val_eligible.csv | Task26 gcn_val_eligible target file | 8999 | 743495 | 460f1624d5889c2261e2dfc44ccd1331c1049ddd9f2c7bad0ee662c51d1f164f | 0 | 0 | 0 |
| configs\targets\elliptic_pg_graphsage_train_eligible.csv | Task26 graphsage_train_eligible target file | 26381 | 1927240 | bfc68def29d51673f898ff1e67fac65cce615d495e23b8c00ec6a339e3173095 | 0 | 0 | 0 |
| configs\targets\elliptic_pg_graphsage_val_eligible.csv | Task26 graphsage_val_eligible target file | 8999 | 743495 | 460f1624d5889c2261e2dfc44ccd1331c1049ddd9f2c7bad0ee662c51d1f164f | 0 | 0 | 0 |
| configs\targets\elliptic_pg_common_train.csv | Task26 common_train target file | 2000 | 146243 | 36c2f9b3a81336d304556702b02b6524411f308ee75f2152d79101586f9c1a22 | 0 | 0 | 0 |
| configs\targets\elliptic_pg_common_val.csv | Task26 common_val target file | 500 | 41642 | 1a747030b9e903acd13d1ad22462cf7e3daf22c4f60fb475e421d2ecd87c8da6 | 0 | 0 | 0 |

## Validation Results

| Check | Command | Status |
| --- | --- | --- |
| pytest | .\.venv\Scripts\python.exe -m pytest -q | pass in final validation gate |
| ruff | .\.venv\Scripts\python.exe -m ruff check . | pass in final validation gate |
| git_diff_check | git diff --check | pass in final validation gate |
| json_validation | .\.venv\Scripts\python.exe -m json.tool docs\elliptic_targets_freeze.json > $null | pass in final validation gate |

## Limitations And Interpretation Guardrails

Elliptic explanations in later tasks will be model-importance explanations under the frozen protocol, not true transaction causes. Cross-backbone comparisons are descriptive and not causal. Because Elliptic has no explanatory edge ground truth, AUROC/AP are unavailable for explanation correctness.

## Acceptance Checklist

| Check | Status |
| --- | --- |
| baseline/cross-repo provenance recorded | True |
| GCN cached=False wrapper created without retraining | True |
| GCN original-graph parity passes | True |
| GraphSAGE original-graph parity passes | True |
| perturbation smoke confirms edge changes reach message passing | True |
| illicit positive-class semantics frozen | True |
| Edge AUROC/AP disabled for Elliptic | True |
| common final test targets frozen before XAI | True |
| per-backbone PG eligibility audited | True |
| COMMON PG train/val derived prospectively | True |
| identical PG train/val node IDs for both backbones | True |
| train/val/final disjoint | True |
| unknown labels excluded as targets | True |
| target/config hashes recorded | True |
| no explanation-result-driven sampling | True |
| pytest/ruff/git diff/json validation pass | True |

## Remaining Blockers

None for Task26 audit and freeze completion.

## Stop Boundary

Task27 is not authorized in this turn. No final Elliptic explanation may run before review. STOP FOR REVIEW.
