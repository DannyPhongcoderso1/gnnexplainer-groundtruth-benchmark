# Elliptic PGExplainer Step03 Evidence Expansion

## Scope

This evidence-expansion rerun uses the exact old Step03 pilot identities, seed `42`, two warm-up steps, and ten measured train steps. It does not start Step04, does not run full Task28 training, and does not use pilot checkpoints as canonical Task28 checkpoints.

## Reviewed Tolerance Amendment

- Original pre-review output-mask tolerance: `atol=1e-08`, `rtol=1e-08`
- Original pre-review verdict: `BLOCKED`
- Original observed GCN output-mask max difference: `4.470348358154297e-08`
- Original observed GraphSAGE output-mask max difference: `2.9802322387695312e-08`
- Reviewed amended output-mask tolerance: `atol=1e-07`, `rtol=1e-07`
- Reviewed CUDA resume continuation tolerance: `atol=1e-07`, `rtol=1e-07`
- Amendment timing: after review of the first evidence-expansion run
- Justification: bounded float32 CUDA numerical comparison
- Parameter and optimizer reload parity remain exact: `atol=0.0`, `rtol=0.0`

## Scope Identity

- Backbones: `GCN`, `GraphSAGE`
- Seed: `42`
- Warm-up steps: `2`
- Measured steps: `10`
- Old/new identity mismatches: `0`
- Final target model outputs accessed: `False`

## Timing And Memory

Timing uses `time.perf_counter`. CUDA is synchronized immediately before each measured step and immediately after the operation. Warm-up steps are excluded from statistics.

Formula:

```text
estimated_training_seconds
= mean_measured_train_step_seconds
  * 2000 train nodes
  * 30 epochs
```

Excluded overhead: validation, checkpoint serialization, reload checks, data/setup overhead, thermal throttling, and operator interruption.

| Backbone | Original mean step | Evidence mean step | Difference | Evidence std | Hours/checkpoint | Hours/all 12 | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 0.133849 | 0.202914 | 0.069065 | 0.015391 | 3.382 | 40.583 | 2933682176 | 3722444800 |
| GraphSAGE | 0.113923 | 0.180413 | 0.066491 | 0.013584 | 3.007 | 36.083 | 2549965312 | 4305453056 |

## Target Semantics

The target code path is:

```text
scripts/prepare_elliptic_task26.py::logits(model, data)
-> frozen_predictions = logits(...).argmax(dim=1).detach()
-> algorithm.train(..., target=frozen_predictions, index=node_id)
```

Ground-truth labels are audited only as a relationship check and are not used as the PGExplainer target source.

## Parameter And Optimizer Audit

| Backbone | PG tensors | PG changed | PG max diff | Backbone tensors | Backbone changed | Backbone max diff | Backbone grad tensors | Optimizer/PG shared | Optimizer/backbone shared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 4 | 4 | 0.024500453845 | 6 | 0 | 0.000000000000 | 0 | 4 | 0 |
| GraphSAGE | 4 | 4 | 0.026271604002 | 8 | 0 | 0.000000000000 | 0 | 4 | 0 |

## Reload And Resume Parity

State/optimizer tolerance is exact: `atol=0.0`, `rtol=0.0`. Output-mask tolerance is the reviewed CUDA float32 amendment: `atol=1e-07`, `rtol=1e-07`.

| Backbone | Reload tensors | Reload mismatches | Reload max diff | Masks compared | Mask max diff | Resume parameter max diff | Resume loss max diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCN | 4 | 0 | 0.000000000000 | 3 | 0.000000059605 | 0.000000007451 | 0.000000000000 |
| GraphSAGE | 4 | 0 | 0.000000000000 | 3 | 0.000000059605 | 0.000000040978 | 0.000000000000 |

CPU exact resume fixture:

- Pass: `True`
- Final PG parameter max diff: `0.0`
- Final optimizer max diff: `0.0`
- Final loss max diff: `0.0`

## Pool And Leakage Audit

- Train pool: `configs/targets/elliptic_pg_common_train.csv`, rows `2000`, SHA256 `36c2f9b3a81336d304556702b02b6524411f308ee75f2152d79101586f9c1a22`
- Validation pool: `configs/targets/elliptic_pg_common_val.csv`, rows `500`, SHA256 `1a747030b9e903acd13d1ad22462cf7e3daf22c4f60fb475e421d2ecd87c8da6`
- Final target metadata hash audited: `True`
- Final target model outputs accessed: `False`
- Final target explanations accessed: `False`
- Final target metrics accessed: `False`
- Pilot/final-target intersection: `0`

## Validation

- Pytest: `160 passed, 3 warnings in 64.39s`
- Ruff: `pass: python -m ruff check .`
- git diff --check: `pass`
- JSON validation: `pass`
- Baseline repository status: `M README.md`

## Verdict

`PASS`. Step04 remains unauthorized until external review.

STOP FOR REVIEW.
