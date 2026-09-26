# Task28 Runner Integrity Patch

## 1. Original blockers

Source: `docs/audit/task28_readiness_pre_step04_audit.md`.

- B1: resume identity did not bind full task-config bytes, implementation sources and environment.
- B2: a matching COMPLETE identity could bypass canonical artifact validation. The old reload helper did not compare the complete saved PG/optimizer state against pre-save state.
- Starting branch: `main`. Starting HEAD: `e2fefa9e978a9471602a4310b1ed1094620215a9`. Initially no tracked changes in this repository. Pre-existing untracked audit/thesis files were left untouched.

## 2. B1 fix

In `scripts/run_elliptic_pgexplainer_training.py`, `config_identity` now uses identity version 2 and includes the full task configuration and its file SHA256, PG configuration SHA256, existing frozen input identities and `hard_compatibility`.

Source SHA256 values are computed from file bytes at runtime. No hash is embedded back into its own source. Sources cover the runner, Elliptic model construction/forward/data helper, PG training/materialization/wrapper helper, model hash helper, candidate mapping helper, seeding helper and installed PyG PGExplainer implementation. The two baseline training YAML files read by `build_models` are also bound.

The latest checkpoint must match the current identity. A mismatch or corrupt latest checkpoint raises instead of falling back to an older checkpoint. Existing checkpoints with the old identity are intentionally incompatible. A subsequent commit changes the HARD Git identity too; this patch does not migrate or weaken old identities.

## 3. HARD vs provenance-only identity

| HARD compatibility | Provenance only |
|---|---|
| Dataset, train/validation pool and backbone checkpoint hashes | GPU model name |
| Full task-config bytes/values and PG-config bytes/values | Platform description |
| Runner/helper/PyG source and model-construction config hashes | Physical CUDA device ordinal |
| Git HEAD when available | Wall-clock timings and memory measurements |
| Python, Torch, PyG, NumPy and Torch CUDA runtime versions | Diagnostic log timestamps |
| Execution device type (CPU versus CUDA) | |
| Backbone, seed, effective explainer configuration and debug/run scope | |

`environment` retains descriptive provenance. Changing the GPU name alone does not alter `hard_compatibility`. Changing CPU to CUDA does. Git records HEAD, while runtime source hashes also capture uncommitted implementation edits.

## 4. B2 fix

`verify_completed_run` parses COMPLETE and manifest and checks status, identity, backbone, seed, epoch, frozen input hashes, canonical path and consistent artifact digest. `verify_canonical_artifact` then verifies existence, SHA256, successful deserialization, exact expected metadata (including configs and ordered train/validation IDs), epoch 30 and algorithm epoch 29.

PG state keys, shapes, dtypes and finite values are checked against a newly materialized PG MLP schema. Its input dimension comes from the frozen backbone checkpoint's output weight. This does not run a backbone forward or evaluate any target. CPU RNG is preserved with `torch.random.fork_rng`.

Adam parameter groups, parameter IDs, state keys, tensor shapes, finite values and positive step counters are validated before strict loading. Immediately after the final save, recursive exact equality checks the entire reloaded payload against the pre-save payload, including PG and optimizer tensors. The file digest is checked again after verification.

Any failed completed-run check raises before the reset/dry-run/training branches. It neither overwrites the artifact nor silently retrains. Structural/hash checks establish completion integrity, not a substitute for frozen Step03 numerical evidence or proof against coordinated malicious rewriting of every artifact and digest.

## 5. Completion ordering

Epoch 30 completes, then existing validation/backbone checks run. The runner atomically saves the canonical artifact, reloads and verifies exact state, verifies its digest, atomically writes the manifest, then atomically writes COMPLETE last. `publish_completion` rechecks the artifact before publishing either completion file.

Existing temp-file plus `os.replace` helpers remain in use for epoch checkpoints, final artifact, manifest and COMPLETE. A failed reload or manifest write does not publish COMPLETE. The epoch checkpoint remains available for a later compatible resume. A diagnostic completion log entry follows the commit marker and is not a prerequisite for completion integrity.

## 6. Tests added

Tiny CPU fixtures cover unchanged resume identity, changed HARD source/config/input/environment identities, full config values, GPU-name provenance, rejection of an incompatible latest checkpoint without fallback, valid completed-run acceptance, missing/hash-mismatched/corrupt artifacts, wrong state keys/shapes, NaN/Inf, wrong backbone/seed/config/checkpoint/epoch metadata, malformed marker/manifest, invalid Adam state, exact saved/reloaded state, RNG preservation, unchanged fixture backbone, marker-last ordering and failed manifest-write behavior.

The finalization fixture uses a tiny PG MLP and one Adam update in a temporary directory. Its epoch-30 metadata is synthetic test input, not an Elliptic training run. Existing CPU interrupted/resumed regression and CLI dry-run tests remain included.

An existing seed-isolation test's dummy hashes were extended with `task_config` and `pgexplainer_config` because these are now mandatory identity inputs. Its scientific assertion is unchanged.

Validation:

- Task28 runner pytest: 55 passed (13 existing plus 42 integrity cases). Two installed Torch JIT deprecation warnings.
- Ruff on the two changed Python files: PASS.
- Repository-wide Ruff: NOT CLEAN due to three pre-existing errors in untracked `scripts/audit_target_selection_flow.py`: I001 and unused `random`/`asdict` imports (F401). That unrelated file was not changed.
- `git diff --check`: PASS.
- Task28/PG YAML parsing and frozen input SHA256 verification: PASS.
- Existing Step03 evidence-expansion manifest JSON parsing: PASS. New temporary completion JSON is parsed and checked by the tests. No scientific JSON was edited.

## 7. Regression safety

AST comparison against starting HEAD confirms identical `read_pool`, `restore_rng`, `rng_state`, `latest_epoch_state`, `validate_pg_mask_health` and the complete per-target training loop. Diff inspection confirms unchanged model construction calls, seed calls, target generation, epoch count, loss calls, optimizer updates and validation semantics. Only integrity/provenance and final serialization verification changed.

Frozen dataset, train/validation pools, both backbone checkpoint hashes and PG-config hash still match the unchanged training YAML. No wrapper, candidate helper, scientific config or checkpoint was edited. Baseline status remains its pre-existing `M README.md`; that README SHA256 remains `802e0c76f69d0778d4161810f3e7509c702d12de64df528c73eaa240ce13e282`.

No Elliptic training, explainer optimization on Elliptic, final-target evaluation, Step03 rerun, Task26/27 artifact update or scientific result change occurred. No canonical checkpoint was created.

## 8. Step03 impact

Training semantics changed: NO. Step03 rerun required: NO.

Existing Step03 evidence remains applicable to the unchanged numerical training math. It does not claim to validate the new source identity or completion guard; the focused CPU integrity/resume/finalization tests validate those changes. The next permitted action is a fresh readiness re-audit, not Step04 or full training.

## 9. Changed files

- `scripts/run_elliptic_pgexplainer_training.py`
- `tests/test_elliptic_task28_training_runner.py`
- `docs/audit/task28_runner_integrity_patch_report.md`

No commit or push performed. Pre-existing untracked files and baseline changes remain untouched.

## 10. Final verdict

PASS — B1/B2 FIXED, READY FOR READINESS RE-AUDIT

This is a scoped integrity-patch verdict, not a repository-wide lint-clean claim or Step04 authorization. Readiness re-audit is authorized. Full training and final-target evaluation are not authorized and were not started.
