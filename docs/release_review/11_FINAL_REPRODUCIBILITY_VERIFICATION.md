# Final reproducibility verification — 2026-10-27

Status: **PARTIAL scientific reproducibility; validated independent publication groups**.

- All 76 minimum identities matched authoritative inventory and original frozen source hash receipts; zero missing local and zero mismatches. All 29 previously published minimum artifacts matched unchanged Git content.
- Repository-only discovery resolves61/76, reporting15 EXTERNAL_ACCESS_REQUIRED. Owner-source discovery resolves76/76. A clean restore of 15 private ZIP members plus repository bytes resolves76/76; second restore is idempotent and never overwrites mismatching files.
- All5268 GNN edge-cache file SHA256 identities match; zero new raw mask is staged. Two representative omitted caches restored by exact ID/path/hash. Random has no learned model state.
- Existing61 regressions plus 5 new restoration/discovery checks:66 PASS, two existing Torch deprecation warnings,54.82s. Tests use only previously hash-verified project fixtures where model loading is involved; discovery/restore checks use no torch import or pickle execution. New-source Ruff/static/import checks passed.
- The release evidence verifier is required again after manifest updates. It checks 217 retained cells at displayed precision, saved final200 arithmetic,400 FRESH5 records/4000 Jaccards, BA seed42 identities, ShapeGGen diagnostic counts and counterfactual aggregates. It is saved-evidence verification, not full re-execution of learned inference.
- Checkpoint/data discovery and private package restoration use standard-library raw streams, exact hashes, allowlisted paths and exclusive creation. Tests reject path traversal, hash substitution, wrong archive identity and overwriting user files.
- Source data/checkpoints/results remain frozen. The thesis PDF SHA256 is120cb35c8b7b23af4c8eb067e8844cd2658b3ff505fdf189aec62afc633fe5bc (90 physical pages). Its20 tables/two figures determine scope. All original scientific payload hashes and217 trace records must remain unchanged at publication; only trace dependency metadata is extended.

Commands run for regression:

```text
python -m pytest -q -p no:cacheprovider tests/test_random_backbone_dispatch.py tests/test_task29_tie_v1.py tests/test_task29_final_stability.py tests/test_budget.py tests/test_stability.py tests/test_sparsity.py tests/test_fidelity.py tests/test_edge_ground_truth_metrics.py tests/test_artifact_resolution.py
python -m ruff check scripts/verify_thesis_artifacts.py tests/test_artifact_resolution.py
```

Published data can check saved numerical claims across Tables4.1–4.20, except the nine explicitly withheld BA500 GraphSAGE PG cells; figure numerical labels are traced, original figure builders remain missing. Saved metrics do not independently prove a new raw-ranking or model-forward run. BA synthetic frozen states are now directly available; optional existing BA500 inference CLI is documented but not run as a new thesis experiment. Elliptic forward evaluation requires licensed external assets, baseline access and original historical guards where applicable. Omitted GNN raw-score recomputation requires cache restoration. A fresh-machine dependency install and full forward replay of every reported cell have not been validated.

Final exact staging, secret/blob-size checks, clean committed-payload checks and live remote equality are recorded in report12. No training, explanation optimization, new scientific measurement, inference-algorithm change or main merge is part of this publication.

Updated 221-file release payload manifest and release evidence verifier: PASS. The 217 retained trace records are unchanged; nine withheld cells remain withheld. All original selected numerical/record hashes and minimum previously published model/data bytes are unchanged.

One 15,159-byte frozen BA500 Random score reference is also included because the existing dispatch regression compares against this exact tensor. It is a cached numerical output, not a learned model and not part of the76 minimum model/graph artifacts. No new GNN raw mask is added.

Final staged gates and clean-archive verification PASS for55 exact paths; payload pushed as `2ff3c2ebc02cff848053c0c61769c7ae09488b20` and remote equality plus221 manifest blobs/61 Git artifact hashes verified. Zero external archives uploaded. See report12.
