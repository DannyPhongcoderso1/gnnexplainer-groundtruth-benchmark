# Task28 Unified Live Progress Logging

## Scope And Safety

Classification: `logging-only observability instrumentation`.
`scientific_semantics_changed=false`. No real smoke, training, stability or
final evaluation is run by this patch. The active Kaggle session is untouched.
This deployment is for a new session only. Do not update files in the active
session, resume its checkpoints under a new HARD identity, or bypass B1/B2.

## Contract And Implemented Phases

Every progress event is one flushed stdout line:
`[TASK28][<BACKBONE>][seed<SEED>][<PHASE>] key=value ...`.
Values are JSON-encoded, including quoted paths and escaped newlines.

- TRAIN: one event after each completed epoch checkpoint and existing CSV write.
  Existing mean loss and training duration, epoch i/30 and checkpoint path.
- VALIDATION: START with the actual limited pool size, then one event per completed
  node in existing order. Existing mask count, candidate count, finite flag,
  mask standard deviation, status/error, i/N and cumulative monotonic elapsed time.
  These are mask-health checks, not Fidelity, sparsity or final evaluation.
- PUBLICATION: START before writing the canonical artifact, DONE only after
  verified publication and the existing completion log. COMPLETE remains last
  among publication artifacts. No log changes checkpoint cadence or contents.
- COMPLETE: CLI success (including a verified completed-run fast path), with
  artifact path. Dry-run/stopped-after-epoch do not emit a false COMPLETE.
- FAILED: CLI exception is logged, then re-raised with its existing failure behavior.

No additional tensor operations or RNG calls. Formatting uses existing Python
values only. Console OSError cannot redirect scientific control flow. Log output
can add I/O wall time, particularly within validation's existing wall-clock
interval. Runtime values are not claimed to be byte-identical across executions.

## Final Evaluation And Stability: Integration Pending

Inspection found no Task28 Elliptic PGExplainer final-evaluation or Jaccard runner.
The requirements exist in `D:/CS/KLTN/docs/super_plan_29082026/`
`08_MANUAL_PRIMARY_FINAL_EVALUATION.md` and `09_MANUAL_STABILITY_FINAL_EVALUATION.md`.
Existing synthetic PG stability scripts and Elliptic GNNExplainer evaluation
are different workflows and were not changed or relabeled as Task28.

`progress_result` supplies a unit-tested rendering contract for future Step08/09
integration, not a new metric pipeline. It forwards only keys already present in
the caller's completed row. FINAL_EVAL fields: `fid_plus`, `fid_minus`,
`one_minus_fid_minus`, `remove_delta`, `retain_delta`,
`retain_probability_ratio`, `sparsity`. STABILITY fields: `jaccard`, `jaccard_mean`.
No missing values are computed or fabricated. A future pipeline must pass its
actual explainer seed/pair identity, node ID, i/N and monotonic elapsed time.
Phase START and completion aggregates must call `progress` using actual pipeline
counts/summaries. No aggregate or per-node final/stability log is claimed live yet.
Those integration tests cannot be claimed until the corresponding runner exists.

## Notebook Streaming

Launcher relays stdout/stderr with separate `tee -a` processes. Workers still run
concurrently. Relays hold no GPU/run lock. Both relays are drained before each
worker status is returned. Worker nonzero exit is preserved, relay failure also
fails closed. Both independent workers are joined even when one fails.

Existing primary paths remain under `/kaggle/working/task28-execution/operator/`:
`concurrent_{gcn,graphsage}.stdout.log`, `concurrent_{gcn,graphsage}.stderr.log`,
`pids.tsv`, `exits.tsv`. Training CSV/JSONL/text and artifact paths are unchanged.
No second notebook monitoring cell is necessary. The sole operator entrypoint is:

```bash
bash /kaggle/working/task28-execution-bootstrap/task28_kaggle_setup.sh --unattended
```

## Source Identity History

- Previous execution: `2e61f0408e725c98df04aad8dbc640ea5be2f355`.
- Previous implementation: `79e89912b1e730208623dcbb26b5bcf6c32b2f14`.
- Changed scientific file: `scripts/run_elliptic_pgexplainer_training.py` only.
- OLD SHA256: `6bf2c80265554ffc5187ee16efdf23cd4a8aee8435426dba0fc3e187e024b6f8`.
- NEW LF/canonical SHA256: `7a6cc924b374a59ce33bf35d50d481c5b0fa7d473f6e4b52d8ca5d3603f8cd58`.
- B1/B2, loss, optimizer, pools, candidate mapping, model code and tolerances unchanged.
- A new source commit and immediate-child compute-plan freeze are necessary.
  Their exact identities are recorded in the rebuilt external Kaggle plan.
  The old execution commit does not contain this instrumentation.

## Validation Strategy

Only CPU/stub tests. AST equivalence strips the specifically named progress calls
and their optional validation context arguments, then compares all numerical,
checkpoint, pool and integrity functions with the historical implementation.
The CLI wrapper and pure logging helpers have separate tests.
Tests check emitted phases, real validation-loop order/counts, absent invented
metrics, fixture result equality, byte-identical fixture checkpoint serialization,
unchanged RNG, concurrent stub workers, notebook output before worker completion,
one event per completed node, log preservation and failure propagation.

## Deployment Boundary

Keep the active Kaggle run unchanged. Before any future deployment, the new
execution commit must exist on the private remote. Upload only the matching rebuilt
ten-file ZIP and sidecar, use the matching pinned bootstrap in a fresh session,
then the sole unattended entrypoint. Fresh preflight/smoke/runtime authorization
remain mandatory. Stability and final evaluation remain unauthorized.
