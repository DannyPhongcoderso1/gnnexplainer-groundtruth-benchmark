# PGExplainer GCN Seed-42 Final Evaluation Freeze

## 1. Evaluation Inputs

- Dataset: `artifacts/data/ba_house.pt`
- Dataset SHA256: `4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94`
- Frozen GCN checkpoint: `artifacts/checkpoints/ba_house_gcn_best.pt`
- Frozen GCN checkpoint SHA256: `4c81814172650e20b6edee5fc97bc66cae46ac1e8ba76baec2226c59a7d1d507`
- PGExplainer checkpoint: `artifacts/explainers/pgexplainer_gcn_ba_house_seed42.pt`
- PGExplainer checkpoint SHA256: `a075cd84c52e14a65b042f99f33e04e6b37e4919a70132ccb71e505b08c12eda`
- Final target file: `results/metrics/explainer_test_targets.csv`
- Final target SHA256: `2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f`
- Final target count: `12`

## 2. Frozen Protocol

Task 16 uses the same frozen Phase-4 evaluation protocol: computation-subgraph candidate edges, continuous edge scores over the full candidate-edge set for Edge AUROC/AP, and a fixed candidate-edge budget of `selected_edge_ratio = 0.2` with `ceil(candidate_edge_count * selected_edge_ratio), minimum 1` rounding.

Fid+, Fid-, 1-Fid-, remove/retain probability diagnostics, and sparsity are computed with the same top-20% selected edge set used by the frozen Phase-4 implementation.

## 3. Parity Checks

- Targets: `12`
- Success: `12`
- Failed: `0`
- Candidate count parity with Phase 4: `12/12`
- Positive GT edge parity with Phase 4: `12/12`
- Negative GT edge parity with Phase 4: `12/12`
- Selected-count parity with Phase 4: `12/12`
- Original-prediction parity with Phase 4: `12/12`
- GCN weights unchanged: `true`

## 4. Main Metrics

| Metric | n | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Edge AUROC | 12 | 0.747969 | 0.188227 | 0.843335 | 0.333333 | 0.977273 |
| Edge AP | 12 | 0.366563 | 0.253247 | 0.434493 | 0.045876 | 0.675609 |
| Fid+ | 12 | 0.416667 | 0.514929 | 0.000000 | 0.000000 | 1.000000 |
| Fid- | 12 | 0.583333 | 0.514929 | 1.000000 | 0.000000 | 1.000000 |
| 1-Fid- | 12 | 0.416667 | 0.514929 | 0.000000 | 0.000000 | 1.000000 |
| Sparsity | 12 | 0.793694 | 0.008901 | 0.797033 | 0.769231 | 0.800000 |

## 5. Probability Diagnostics

| Metric | n | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| remove_probability_delta | 12 | -0.001014 | 0.004187 | -0.000095 | -0.009157 | 0.003353 |
| retain_probability | 12 | 0.272888 | 0.002125 | 0.271254 | 0.271254 | 0.276987 |
| retain_probability_delta | 12 | 0.000245 | 0.001885 | 0.000922 | -0.003488 | 0.003290 |
| retain_probability_ratio | 12 | 0.999108 | 0.006905 | 0.996611 | 0.988016 | 1.012852 |

## 6. Runtime Semantics

PGExplainer upfront training runtime is `22.389040600013686` seconds.

Task-16 final-test `inference_runtime_seconds` means the trained PGExplainer generating explanation scores for one target. It is reported as `0.005550 +/- 0.001033` seconds per target. This runtime is a PGExplainer inference measurement and should not be treated as the same computational stage as GNNExplainer optimization/generation runtime.

## 7. Result Artifact Checksums

- Per-node metrics: `results/metrics/pgexplainer_gcn_seed42/per_node_metrics.csv`
- Per-node metrics SHA256: `b88f35d20f6cb8e3f7ceb9dd8ab0d48ef7d71cb1a091ec98be7d0a7f3ed1f626`
- Summary metrics: `results/metrics/pgexplainer_gcn_seed42/summary_metrics.csv`
- Summary metrics SHA256: `6c16b5eef25ce923b7a570e41bd9f9fd385a1d7326929ca462090770a05cea3d`
- Result manifest: `results/metrics/pgexplainer_gcn_seed42/manifest.json`
- Result manifest SHA256: `37458430f9fc295b4c9e2ff2fb31f6fdea904c62cb1ee894a90605394d47ff79`
- Raw explanation root: `results/explanations/pgexplainer_gcn_seed42/`
- Raw explanation file count: `72`

Detailed raw explanation file checksums are recorded in `docs/pgexplainer_gcn_seed42_evaluation_freeze.json`.

## 8. Code Provenance

- Evaluation base commit: `698e848`
- Evaluation script SHA256: `5f3ccc9ca5042c477696c3d7ea360ab66816985cc0fba6c511f0fc06402fcdf1`
- Evaluation module SHA256: `423082d620a177b911397c8c4d059e78709e032a351ce59dc99a65f7ea9e8f16`
- Task-15 commit: `698e848`
- Training code base commit: `085a504a6c6262007bf7df1d47aa7fa0be8074f5`

The Task-16 final evaluation was executed before the Task-16 source files were committed. The source hashes above freeze the implementation used for this evaluation handoff without claiming that the run came from a future commit.

## 9. Interpretation Limits

Task 16 establishes the frozen PGExplainer seed-42 results on the same 12-node evaluation set. Cross-method interpretation is deferred to Task 17.

The 12-node final test set has now been exposed to PGExplainer. These results must not be used to tune PGExplainer hyperparameters.

No post-test tuning is allowed for epochs, lr, edge_size, edge_ent, temperature, bias, training target membership, or checkpoint-selection strategy. Any future rerun may only be reproduction or bug verification and cannot be used to choose a better configuration.

## 10. Freeze Statement

The PGExplainer GCN seed-42 final evaluation is frozen with `12/12` successful targets, Phase-4 protocol parity, verified result checksums, and no post-test tuning.
