# ShapeGGen Task24 PGExplainer Audit Freeze

## Scope

This audit is read-only with respect to experimental evidence. It does not
retrain PGExplainer, rerun explanation generation, change checkpoints, change
targets, change seeds, change hyperparameters, tune on final targets, or start
Task25.

## Initial Git State

- Branch: `main`
- HEAD: `957f490a1fa1611fe6fd100dadefac5a6d0a1c19`
- Status before audit: `M docs/shapeggen_pgexplainer_freeze.json
 M docs/shapeggen_pgexplainer_report.md
?? docs/shapeggen_pgexplainer_audit.json
?? scripts/audit_shapeggen_task24.py`
- Upstream parity before audit: `0	0`

## Dataset And Pool Audit

| Pool | Expected | Observed | Unique | Duplicate | Missing node id | Invalid node id | Sorted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Common train | 539 | 539 | 539 | 0 | 0 | 0 | True |
| Common validation | 182 | 182 | 182 | 0 | 0 | 0 | True |
| Final targets | 167 | 167 | 167 | 0 | 0 | 0 | True |

- train_val_intersection_count: `0`
- train_final_intersection_count: `0`
- val_final_intersection_count: `0`

## Checkpoint Audit

| Backbone | Seed | Exists | Readable | Epochs | Reload | Weights unchanged | SHA256 |
|---|---:|---:|---:|---:|---:|---:|---|
| GCN | 42 | True | True | 30 | True | True | `95c7b5070099c04968b3ebf5b4fa7664ec468a5cc9ff4a17752c198641def978` |
| GCN | 0 | True | True | 30 | True | True | `53e4dbc903f7a8fa4906ea595897f0db069ead6a42cf9851fbda0cc2b9b576ac` |
| GCN | 1 | True | True | 30 | True | True | `c9a11e09b152fd268496f72fb6f9b14ec0492cedec989ec90274bcceed1f924a` |
| GCN | 2 | True | True | 30 | True | True | `50154f5c71caa2117ab95e8e0cc224ca6d0a096ee746bcc21c5833c53d0e1b48` |
| GCN | 3 | True | True | 30 | True | True | `0cceda0622da6b409bb96c8d20da8033c3b452e5fbb500779df41faa65534b1f` |
| GCN | 4 | True | True | 30 | True | True | `0615477814038a02d95ba70c6c42180eddb7ab7d5c1138fbda2656fbfdf539f0` |
| GraphSAGE | 42 | True | True | 30 | True | True | `a7d77988b9716c84c0b0436b2ef5c340e4af8ce9afeb14906cb615d2eee0e913` |
| GraphSAGE | 0 | True | True | 30 | True | True | `a62b4cdcb27926bd4911fbb1591d697a6c6ad2781b7aac0b19cd155d6daa2ac7` |
| GraphSAGE | 1 | True | True | 30 | True | True | `a2bb660a9a4ef3475413b985ed97d189652791de77971c6e91bde258bdaba81f` |
| GraphSAGE | 2 | True | True | 30 | True | True | `68afac6f030041a1cb47a64faee1f5da6ea67f3db4892224fd71e64aaac35db0` |
| GraphSAGE | 3 | True | True | 30 | True | True | `a15abde1a663bc2ed98e287c1e42b98307d7e35f0f48c09dafe8526902152f70` |
| GraphSAGE | 4 | True | True | 30 | True | True | `f6bc41090abd715cfad915e398576622fd4ecfc30648ca0329e95aa2f79ba674` |

## Row Completeness Audit

| Backbone | Seed | Expected targets | Observed unique targets | Missing | Extra | Duplicates | Success | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 42 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GCN | 0 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GCN | 1 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GCN | 2 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GCN | 3 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GCN | 4 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GraphSAGE | 42 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GraphSAGE | 0 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GraphSAGE | 1 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GraphSAGE | 2 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GraphSAGE | 3 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| GraphSAGE | 4 | 167 | 167 | 0 | 0 | 0 | 167 | 0 |

## Null And Missing Audit

| File | Column/field | Total rows | Null count | Classification | Reason |
|---|---|---:|---:|---|---|
| `results\metrics\shapeggen\gcn\pgexplainer\per_node_metrics.csv` | generation_runtime_seconds | 167 | 167 | EXPECTED_STRUCTURAL_NULL | expected structural null: Task24 PGExplainer separates upfront training, per-target inference, and per-target evaluation; no per-target generation field is applicable. |
| `results\metrics\shapeggen\gcn\pgexplainer\per_node_metrics.csv` | error | 167 | 167 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\gcn\pgexplainer\stability_seed_metrics.csv` | generation_runtime_seconds | 835 | 835 | EXPECTED_STRUCTURAL_NULL | expected structural null: Task24 PGExplainer separates upfront training, per-target inference, and per-target evaluation; no per-target generation field is applicable. |
| `results\metrics\shapeggen\gcn\pgexplainer\stability_seed_metrics.csv` | error | 835 | 835 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\gcn\pgexplainer\stability_per_node_jaccard.csv` | error | 167 | 167 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\gcn\pgexplainer\summary_metrics.csv` | mean | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\gcn\pgexplainer\summary_metrics.csv` | std | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\gcn\pgexplainer\summary_metrics.csv` | median | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\gcn\pgexplainer\summary_metrics.csv` | min | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\gcn\pgexplainer\summary_metrics.csv` | max | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\gcn\pgexplainer\validation_health_seed0.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\gcn\pgexplainer\validation_health_seed1.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\gcn\pgexplainer\validation_health_seed2.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\gcn\pgexplainer\validation_health_seed3.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\gcn\pgexplainer\validation_health_seed4.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\gcn\pgexplainer\validation_health_seed42.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\per_node_metrics.csv` | generation_runtime_seconds | 167 | 167 | EXPECTED_STRUCTURAL_NULL | expected structural null: Task24 PGExplainer separates upfront training, per-target inference, and per-target evaluation; no per-target generation field is applicable. |
| `results\metrics\shapeggen\graphsage\pgexplainer\per_node_metrics.csv` | error | 167 | 167 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\stability_seed_metrics.csv` | generation_runtime_seconds | 835 | 835 | EXPECTED_STRUCTURAL_NULL | expected structural null: Task24 PGExplainer separates upfront training, per-target inference, and per-target evaluation; no per-target generation field is applicable. |
| `results\metrics\shapeggen\graphsage\pgexplainer\stability_seed_metrics.csv` | error | 835 | 835 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\stability_per_node_jaccard.csv` | error | 167 | 167 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\summary_metrics.csv` | mean | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\graphsage\pgexplainer\summary_metrics.csv` | std | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\graphsage\pgexplainer\summary_metrics.csv` | median | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\graphsage\pgexplainer\summary_metrics.csv` | min | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\graphsage\pgexplainer\summary_metrics.csv` | max | 13 | 1 | EXPECTED_STRUCTURAL_NULL | expected structural null: per-target generation_runtime_seconds has n=0 for Task24 PGExplainer because runtime is separated into upfront training, per-target inference, and per-target evaluation. |
| `results\metrics\shapeggen\graphsage\pgexplainer\validation_health_seed0.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\validation_health_seed1.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\validation_health_seed2.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\validation_health_seed3.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\validation_health_seed4.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |
| `results\metrics\shapeggen\graphsage\pgexplainer\validation_health_seed42.csv` | error | 182 | 182 | EXPECTED_STRUCTURAL_NULL | expected structural null: success rows have no error message. |

## Metric Identity Audit

| Backbone | Identity | Checked rows | Max abs error | Violations | Pass |
|---|---|---:|---:|---:|---:|
| GCN | one_minus_fid_minus | 1002 | 0 | 0 | True |
| GCN | remove_probability_delta | 1002 | 0 | 0 | True |
| GCN | retain_probability_delta | 1002 | 0 | 0 | True |
| GCN | retain_probability_ratio | 1002 | 0 | 0 | True |
| GCN | sparsity | 1002 | 0 | 0 | True |
| GraphSAGE | one_minus_fid_minus | 1002 | 0 | 0 | True |
| GraphSAGE | remove_probability_delta | 1002 | 0 | 0 | True |
| GraphSAGE | retain_probability_delta | 1002 | 0 | 0 | True |
| GraphSAGE | retain_probability_ratio | 1002 | 0 | 0 | True |
| GraphSAGE | sparsity | 1002 | 0 | 0 | True |

Fid+ is implemented as a prediction-change indicator after removing the
selected top-20% edges. Fid- is implemented as a prediction-change indicator
after retaining only the selected top-20% edges. Both use the frozen backbone
original predicted class as the probability reference for diagnostics.

## Ground-Truth Metric Audit

Edge AUROC and Edge AP use continuous scores over the full candidate-edge set,
using PGExplainer edge scores without top-20% binarization. Mapping errors:
GCN `0`, GraphSAGE `0`. Single-class cases: GCN `0`, GraphSAGE `0`. AUROC/AP
computed targets: GCN `167`, GraphSAGE `167`.

## Budget And Stability Audit

Budget uses `selected_edge_ratio=0.2` with rule
`ceil(candidate_edge_count * selected_edge_ratio), minimum 1`. Sparsity is
computed from selected and candidate counts, not forced to `0.8`.

| Backbone | Checkpoints | Targets | Expected pairs | Observed pairs | Null Jaccard | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 5 | 167 | 1670 | 1670 | 0 | 0.827289 | 0.333333 | 1.000000 |
| GraphSAGE | 5 | 167 | 1670 | 1670 | 0 | 0.793581 | 0.090909 | 1.000000 |

## Runtime Audit

PGExplainer runtime is separated into upfront explainer training, per-target
trained-explainer inference, and per-target metric evaluation.

| Backbone | Training runtime s | Mean inference runtime s | Mean evaluation runtime s |
|---|---:|---:|---:|
| GCN | 180.965073 | 0.007096 | 0.014371 |
| GraphSAGE | 110.941023 | 0.003956 | 0.008129 |

## Summary Recomputaton

| Backbone | Metric | Stored value | Recomputed value | Absolute difference | Pass |
|---|---|---:|---:|---:|---:|
| GCN | edge_auroc | 0.385184 | 0.385184 | 0 | True |
| GCN | edge_ap | 0.182545 | 0.182545 | 0 | True |
| GCN | fid_plus | 0.041916 | 0.041916 | 0 | True |
| GCN | fid_minus | 0.203593 | 0.203593 | 0 | True |
| GCN | one_minus_fid_minus | 0.796407 | 0.796407 | 0 | True |
| GCN | sparsity | 0.795928 | 0.795928 | 0 | True |
| GCN | stability | 0.827289 | 0.827289 | 0 | True |
| GCN | training_runtime_seconds | 180.965073 | 180.965073 | 0 | True |
| GCN | inference_runtime_seconds_mean | 0.007096 | 0.007096 | 0 | True |
| GCN | evaluation_runtime_seconds_mean | 0.014371 | 0.014371 | 0 | True |
| GraphSAGE | edge_auroc | 0.530641 | 0.530641 | 0 | True |
| GraphSAGE | edge_ap | 0.285020 | 0.285020 | 0 | True |
| GraphSAGE | fid_plus | 0.622754 | 0.622754 | 0 | True |
| GraphSAGE | fid_minus | 0.610778 | 0.610778 | 0 | True |
| GraphSAGE | one_minus_fid_minus | 0.389222 | 0.389222 | 0 | True |
| GraphSAGE | sparsity | 0.795928 | 0.795928 | 0 | True |
| GraphSAGE | stability | 0.793581 | 0.793581 | 0 | True |
| GraphSAGE | training_runtime_seconds | 110.941023 | 110.941023 | 0 | True |
| GraphSAGE | inference_runtime_seconds_mean | 0.003956 | 0.003956 | 0 | True |
| GraphSAGE | evaluation_runtime_seconds_mean | 0.008129 | 0.008129 | 0 | True |

Maximum aggregate difference:
`0`.

## Integrity Audit

- Old report hash before audit update: `c29e6a00a0b7f9f9e73d88eb1396d1115617ba76f3dd602cc4849f23f314da15`
- Old freeze hash before audit update: `99104b3692b44a1f2b6c4879b72515a46f26f4a4c3c04509c3853b4860febd08`
- Previous report hash from prior response: `c79773e390879a337bb6580464cbb2ff3538d99d022f618fe747b985cd5c4f9f`
- Previous freeze hash from prior response: `099b2c9c705d93a48533265b15d3dd7665788824cff0938aee77b4ad42bfd369`
- Experimental artifacts changed by this audit: `False`
- Raw explanation manifests are recorded in `docs/shapeggen_pgexplainer_audit.json`.

## Acceptance

- common_train_539: True
- common_val_182: True
- final_targets_167: True
- no_duplicate_missing_invalid_pool_nodes: True
- train_val_final_disjoint: True
- pg_checkpoints_complete_and_readable: True
- epochs_reload_and_weights_ok: True
- backbone_hashes_match: True
- row_coverage_complete: True
- unexpected_required_nulls_zero: True
- domain_violations_zero: True
- identity_violations_zero: True
- edge_protocol_complete: True
- budget_and_sparsity_ok: True
- stability_complete: True
- summary_recomputed: True
- task25_not_started: True

## Validation

- pytest: `119 passed, 3 warnings`
- Ruff: `All checks passed`
- git diff --check: `passed`
- freeze JSON parse: `passed`

## Verdict

Task24 audit verdict: `PASS`.
Task25 is authorized only after this audit commit/push is reviewed. Task25 was
not started in this task. Task25 is not started automatically.

STOP FOR REVIEW
