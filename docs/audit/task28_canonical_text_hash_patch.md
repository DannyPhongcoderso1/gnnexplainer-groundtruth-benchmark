# Task28 Canonical Text Hash Patch

## Root Cause

Read-only verification of `src/gnnexplainer_benchmark/explain/subgraph_mapping.py` reproduced both supplied hashes:

- CRLF/raw Windows: `ab5d7809fe235c1d5a5c1e083e4c77d5a19acfadd0d70eb38d2ba396972c7bc5`.
- LF/canonical: `8a39548748d7c97d3b29288683837687267dd7d2f03ed02dc4a48876e44b1391`.

Converting only LF to CRLF reproduces the former hash exactly. Candidate mapping file was not edited.

## Implementation And Identity

`scripts/run_elliptic_pgexplainer_training.py:canonical_text_sha256` reads bytes, validates UTF-8 and designated text suffix, rejects NUL, replaces CRLF then lone CR with LF, and hashes the resulting bytes. It does not rewrite the file, trim whitespace, strip BOM, parse configs, or normalize arbitrary source differences. `hard_compatibility` calls it only for its explicit Python/YAML source list, including PyG implementation source and baseline model configs.

The existing raw `sha256`, input verification, task/PG-config/pool hashes, artifact hash checks, completion checks and serialization logic remain unchanged. Binary `.pt` is rejected by the text helper. HARD retains Git HEAD, Python, Torch, PyG, NumPy, CUDA runtime and device type. Policy marker `utf8-crlf-cr-to-lf-v1` enters the HARD identity, so legacy resume states fail identity matching rather than being silently upgraded. B2 remains byte/state exact.

Compute-plan and Kaggle-plan HARD values are synchronized. All source hash values except candidate_mapping and the modified runner remain numerically identical. New runner canonical/raw hash: `6bf2c80265554ffc5187ee16efdf23cd4a8aee8435426dba0fc3e187e024b6f8`. The raw file table remains raw; baseline YAML transfer checks now explicitly use `baseline_config_raw_sha256` rather than canonical HARD values.

## Validation

- 81 tests pass: canonical hashing tests, all existing Task28 runner/B1/B2 regression tests, and compute-plan tests. Two existing Torch JIT deprecation warnings.
- 44 adapter/auth/ZIP/runtime-gate tests pass, including separate raw-config transport hashing. Total: 125 passing tests.
- LF/CRLF/CR equivalence, content/whitespace/BOM sensitivity and unchanged raw binary hashing pass.
- Local `hard_compatibility(torch.device('cuda'))` equals the updated plan after separately checking actual Git HEAD. This constructs an identity only, without executing CUDA work.
- AST comparison against frozen commit `61c554bef5f462610ff8c2765ac8488d72b9e48d` proves every runner function/class other than the hash helper/HARD collector is unchanged, including training, validation, input checks, optimizer/resume logic, B1 config identity and B2 verification.
- Scoped Ruff, JSON/adapter/bootstrap hash consistency and `git diff --check` pass. No experimental smoke, training job, or final-target evaluation was executed. Regression fixtures are tests, not experiments.

## Deployment Boundary At Initial Review

Local patch PASS. Kaggle deployment/rebuild BLOCKED pending reviewed source publication and exact execution re-freeze. The currently pinned published commit contains the old raw-hash implementation and old compute plan. A ten-file data/execution bundle does not include the runner, so rebuilding it alone cannot deploy this fix. No dirty-source overlay, monkeypatch, Git bypass, force update, or guessed new commit is used. Existing bundle/ZIP/checksum/manifest are left unchanged and are not presented as containing this patch.

Next action requires review: commit/publish the scoped source and plan amendment, resolve and pin its exact execution commit (including the local runbook freeze rule), then rebuild the bundle once. Changing Git HEAD remains a HARD identity change requiring a new freeze; it is not weakened or made dynamic here. Step05 remains unauthorized.

## Reviewed Source Publication

The subsequent final-deployment request authorizes this scoped source commit followed by a separate immediate-child compute-plan freeze commit and normal push. The paragraphs above preserve the initial blocked state, not the final deployment status. Exact published execution identity and packaging results are recorded in the external Kaggle plan/runbook after publication to avoid self-referential commit hashes.

Pre-commit revalidation: 128 focused tests PASS (including three headless-environment tests), two existing Torch JIT warnings. Scoped Ruff, pip check, JSON parsing, Bash syntax and git diff check PASS. Dataset/checkpoint/pool raw hashes and all numerical training/integrity functions remain unchanged. The external setup/notebook/preflight/smoke/launcher adapters explicitly set MPLBACKEND=Agg before runtime imports or child launch. This is headless runtime hygiene, not a scientific parameter or a new B1 HARD field. No smoke, full training, stability, final evaluation or Step05 authorization is performed.
