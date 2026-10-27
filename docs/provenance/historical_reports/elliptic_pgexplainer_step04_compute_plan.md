# Elliptic PGExplainer Step04 Compute Plan

## Canonical Source Hash Amendment

2026-09-27: HARD `source_sha256` now uses explicit UTF-8 text hashing with CRLF and lone CR converted to LF before SHA256. No other bytes, whitespace, BOM, encoding, or source syntax are normalized. The HARD record includes `source_hash_policy=utf8-crlf-cr-to-lf-v1`; Git HEAD, Python/Torch/PyG/NumPy/CUDA/device remain HARD. Old raw-source resume identities are not silently migrated or accepted.

`file_sha256` remains a raw-byte transport/input table, including binary artifacts and task/config/pool bytes. Only its runner entry changes because this patch changes runner source. Candidate-mapping source content is untouched. Baseline configuration transfer hashes are stored separately as `baseline_config_raw_sha256` in the Kaggle plan, never inferred from canonical HARD hashes. B2 artifact hashes remain raw and exact.

The reviewed source patch is published as `79e89912b1e730208623dcbb26b5bcf6c32b2f14`. This separate, immediate-child plan commit re-freezes execution using the existing deterministic resolution rule. Published `61c554bef5f462610ff8c2765ac8488d72b9e48d` is the historical pre-patch reference only. The clean-tree/HEAD checks remain in force. No Step05 authorization is granted by this amendment. Earlier local authorization descriptions below are historical, not approval to execute the amended tree or Kaggle training. The external Kaggle plan must pin the literal published execution SHA and retain all training/evaluation authorizations as false.

## Scope And Freeze

Step03 is PASS and the B1/B2 readiness re-audit is READY FOR STEP04. This plan creates operator commands only. No training, pilot rerun, final-target inference or evaluation is performed here. Objective, data, checkpoints, seeds, pools, epochs and hyperparameters are unchanged.

Machine-readable source: `docs/elliptic_pgexplainer_step04_compute_plan.json`. Operator functions: `docs/elliptic_pgexplainer_step04_runbook.ps1`. Repository: `D:/CS/KLTN/gnnexplainer-groundtruth-benchmark`, branch `main`, Python `.venv/Scripts/python.exe`.

Implementation commit: `79e89912b1e730208623dcbb26b5bcf6c32b2f14`. Execution commit is the first first-parent child of that commit updating this plan JSON. The runbook resolves that exact commit from Git history and requires both its parent to equal the implementation commit and current HEAD to equal the execution commit. This avoids embedding a commit hash inside its own commit. It does NOT accept arbitrary newer HEADs. After committing, print the literal frozen execution SHA with `git rev-parse HEAD` and retain it with operator logs. Source hashes in the JSON bind the implementation independently of commit metadata.

Git HEAD is HARD resume identity: do not commit, checkout, pull, upgrade dependencies or edit configuration between training and resume. Any later documentation/authorization commit requires a reviewed execution-identity plan before continuing. Do not bypass the guard to make an old checkpoint resume.

## Frozen Configuration

- GCN and GraphSAGE, primary seed 42, future independent stability seeds 0,1,2,3,4.
- Each checkpoint: 30 epochs, 2000 sorted training IDs, 500 validation IDs, fixed final epoch selection.
- Final 200 IDs are hashed for provenance only, not supplied to the training runner.
- Frozen target: original backbone predicted class, not ground-truth label.
- Shared PG configuration remains unchanged: Adam lr0.003, installed PyG objective/regularizers and temperature schedule. No hyperparameter or pool changes after freeze.
- Downstream scope remains 2-hop, selected-edge ratio 0.2 with minimum-one ceiling. No final evaluation occurs in this step.
- Python3.11.9, Torch2.11.0+cu128, PyG2.8.0.post1, NumPy2.4.6, CUDA runtime12.8. GPU observed: NVIDIA GeForce RTX4060 Laptop GPU. GPU name is provenance, while CUDA device type and recorded dependency/source identities are HARD.

The manifest includes complete SHA256 values for runner, task/PG configs, dataset, checkpoints, train/validation/final CSVs, and helper/model-construction sources. The runbook verifies hashes and environment before every process, including resume and completion verification. It explicitly sets the process-local ELLIPTIC_BASELINE_REPO to the frozen baseline path, without changing system settings. Never run another GPU training process alongside it. A named mutex prevents overlap between instances of this runbook, not arbitrary third-party GPU jobs.

## Operator Commands

Use a normal PowerShell window. Dot-sourcing defines functions only:

```powershell
Set-Location -LiteralPath 'D:\CS\KLTN\gnnexplainer-groundtruth-benchmark'
. .\docs\elliptic_pgexplainer_step04_runbook.ps1
Assert-Task28Freeze
```

### Primary Seed42

Run sequentially, and stop for review on any failure. These functions call the exact Python argv recorded in the manifest, with `--device cuda`, unbuffered output, no debug limits and no reset flag.

```powershell
Invoke-FrozenRun -Backbone gcn -Seed 42
Invoke-FrozenRun -Backbone graphsage -Seed 42
```

For example the first command invokes:

```powershell
.\.venv\Scripts\python.exe -u scripts/run_elliptic_pgexplainer_training.py --config configs/evaluation/elliptic_pgexplainer_training.yaml --backbone gcn --seed 42 --device cuda
```

Use the function, not the bare example, to retain preflight, locking and exit logging. Console output is appended with Tee-Object to `results/metrics/elliptic/task28_operator/elliptic_pg_<backbone>_seed<seed>.console.log`. Exit code and start/end UTC are appended to the corresponding `.exits.jsonl`. Nonzero exit throws and must stop the operator sequence. An abrupt shell/power termination may prevent recording an exit entry; its absence is not success.

### Stability Seeds: Not Authorized

These exact commands are generated for future review only. They are not executed and the current function deliberately rejects training seeds other than 42. Do not remove this guard without an approved follow-up freeze/authorization.

```powershell
Invoke-FrozenRun -Backbone gcn -Seed 0
Invoke-FrozenRun -Backbone graphsage -Seed 0
Invoke-FrozenRun -Backbone gcn -Seed 1
Invoke-FrozenRun -Backbone graphsage -Seed 1
Invoke-FrozenRun -Backbone gcn -Seed 2
Invoke-FrozenRun -Backbone graphsage -Seed 2
Invoke-FrozenRun -Backbone gcn -Seed 3
Invoke-FrozenRun -Backbone graphsage -Seed 3
Invoke-FrozenRun -Backbone gcn -Seed 4
Invoke-FrozenRun -Backbone graphsage -Seed 4
```

## Outputs And Completion

All 12 explicit run paths and argv arrays are in the JSON. Pattern: `artifacts/explainers/elliptic_pg_<backbone>_seed<seed>_run/checkpoints/epoch_001.pt` through `epoch_030.pt`; canonical artifact is the sibling `elliptic_pg_<backbone>_seed<seed>.pt`, with sibling `_manifest.json`. COMPLETE is inside the `_run` directory. All 30 epoch states are retained. No automatic deletion is authorized.

Metrics logs: `results/metrics/elliptic/<backbone>/pgexplainer/training_seed<seed>.csv`, `.jsonl`, `.txt`. CSV is the cumulative epoch table. JSONL/text and operator console logs append across attempts. Epoch checkpoints, not console lines, determine resume position.

Success requires finite training/validation, unchanged backbone, epoch30, valid PG/Adam tensors, exact final-save reload, matching identity/hashes and a consistent manifest. COMPLETE is written atomically LAST. File presence alone is insufficient. Validate a completed checkpoint safely:

```powershell
Invoke-FrozenRun -Backbone gcn -Seed 42 -ValidateOnly
Invoke-FrozenRun -Backbone graphsage -Seed 42 -ValidateOnly
```

ValidateOnly refuses missing COMPLETE before invoking `--dry-run`; with COMPLETE present the runner takes its strict already_complete verification path. It cannot fall through to training. A corrupt artifact or identity mismatch stops with an error and must not be reset/retrained silently.

Inspect progress from a separate shell (substitute graphsage as needed):

```powershell
Get-Content results/metrics/elliptic/gcn/pgexplainer/training_seed42.txt -Tail 10 -Wait
Get-ChildItem artifacts/explainers/elliptic_pg_gcn_seed42_run/checkpoints/epoch_*.pt
Import-Csv results/metrics/elliptic/gcn/pgexplainer/training_seed42.csv | Select-Object -Last 3
nvidia-smi --query-gpu=name,temperature.gpu,memory.used,utilization.gpu --format=csv
```

The runner writes epoch progress to files and prints its final JSON to stdout. A quiet console during an epoch is expected, not evidence of a hang.

## Runtime And Disk Estimate

Source: frozen Step03 evidence-expansion timings, seed42, warmup2, measured10 synchronized CUDA training steps. Formula: mean step seconds * 2000 * 30. These are estimates, not full-run observations.

| Backbone | Step seconds | Training hours/checkpoint | Six checkpoints |
|---|---:|---:|---:|
| GCN | 0.20291414 | 3.382 | 20.291 |
| GraphSAGE | 0.18041338 | 3.007 | 18.041 |

Primary pair: 6.389 training-only hours. All twelve: 38.333 hours, correctly six of each backbone, not twelve of each. A discretionary 50% planning margin gives about 57.5 hours for all twelve, not a guaranteed bound. Excludes setup, validation500, serialization, integrity checks, cooling and interruption replay. Newly strengthened integrity checks add unmeasured overhead. Internal serialization ledger excludes the second state rewrite, so do not equate it with complete wall time.

Pilot PG artifacts measured 225989/256085 bytes. Conservative allowance: 2MiB per checkpoint file *32 (30 epochs + canonical + one atomic temporary) plus 10MiB logs/metadata per run =74MiB/run, 888MiB/all12. Reserve at least 2GiB free disk. Observed free space at plan preparation: 850019545088 bytes, a snapshot not a promise. Canonical epoch sizes have not been measured; inspect actual sizes after primary runs. Dataset/backbone tensors are not duplicated in every PG state. Repeated operator logs can exceed estimates.

## Interruption And Failure Recovery

- Keep AC power, ventilation and the existing environment. Monitor NVIDIA telemetry without changing system settings. Stop for cooling or investigation on persistent thermal throttling, OOM or hardware errors; no unvalidated hardware-specific temperature threshold is invented here.
- A controlled stop after a completed epoch can use the runner's existing `--stop-after-epoch N` only in a separately reviewed operator invocation; ordinary commands above run to completion. Ctrl+C/power loss during an epoch loses that partial epoch. Resume repeats it from the last atomic epoch checkpoint, not from an arbitrary node.
- Rerun the identical Invoke-FrozenRun command under the same execution HEAD/environment. It resumes automatically. If COMPLETE exists, it verifies and returns already_complete without optimization.
- A corrupt latest checkpoint or incompatible identity fails closed. Preserve all artifacts/logs and request review. Do not delete checkpoints, fall back to an older epoch, use `--reset-incomplete`, alter hashes or change seeds/config to force acceptance.
- Crash after artifact/manifest but before COMPLETE leaves an incomplete run. A compatible rerun resumes from epoch30 and repeats final validation/publication. Do not manually create COMPLETE.
- No final-target metrics or hyperparameter tuning are authorized. Stability authorization must be reviewed separately after primary evidence.

## Validation And Publication

Plan tests check schema fields, all twelve unique identities/paths/commands, frozen hashes/config and six-plus-six estimates. PowerShell parser validation checks the runbook without executing training commands. Task28 runner/integrity tests and scoped Ruff are required. JSON parsing and git diff checks are required. Pre-existing unrelated repo-wide Ruff findings are outside this scope.

Validation at freeze preparation: 70 tests passed (55 runner/integrity and 15 plan tests), two installed Torch JIT deprecation warnings, 50.44 seconds. Scoped Ruff PASS, JSON parse PASS, PowerShell parse PASS, git diff check PASS. NVIDIA visibility confirmed RTX4060 Laptop GPU, 8188MiB total memory; telemetry is a preparation-time snapshot, not a training measurement. No training command in this runbook was executed.

Commit only this plan JSON, report, PowerShell runbook and focused plan tests, then push the current branch without force. Unrelated untracked audit/thesis files are not included. The patch commit is already the parent and is pushed as part of branch history. No tag or model artifacts are added.

STEP04 VERDICT: PASS

PRIMARY TRAINING AUTHORIZED: YES

STABILITY TRAINING AUTHORIZED: NO

FINAL EVALUATION AUTHORIZED: NO

STOP FOR REVIEW
