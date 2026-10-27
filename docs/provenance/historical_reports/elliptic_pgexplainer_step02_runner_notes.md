# Elliptic PGExplainer Step02 Runner Notes

Step02 implements train-only infrastructure for Task28. It does not train the full Elliptic PGExplainer protocol and does not access final explanation targets.

## Canonical CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_elliptic_pgexplainer_training.py --help
.\.venv\Scripts\python.exe scripts\run_elliptic_pgexplainer_training.py --backbone gcn --seed 42 --dry-run
```

The runner accepts exactly one backbone and one seed per invocation. Allowed backbones are `gcn` and `graphsage`. Allowed seeds are `42`, `0`, `1`, `2`, `3`, and `4`.

## Resume Semantics

Each run writes epoch checkpoints under `artifacts/explainers/elliptic_pg_<backbone>_seed<seed>_run/checkpoints/`. Checkpoints are written through temporary files and atomic replacement. Resume is the default path and loads the latest matching epoch checkpoint.

Resume refuses checkpoints when the config identity differs. The identity covers backbone, seed, dataset hash, common train hash, common validation hash, backbone checkpoint hash, PGExplainer config hash, selected-edge ratio, target semantics, and checkpoint-selection policy.

## Stop Boundary

This step only prepares infrastructure and tests. Full Task28 training commands are not printed until the later manual training gate.
