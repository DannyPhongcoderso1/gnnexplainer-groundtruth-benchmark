# Release verification — 2026-10-26

Status: **PARTIAL scientific reproducibility; independent selected payload gates PASS**. No model training, new explanation run, gradient computation or frozen-result replacement was performed.

- Targeted pytest: **61 passed**, two existing Torch deprecation warnings, 38.98 seconds. Tests used the existing research Python environment and temporary ignored BA500 fixtures, which are excluded from Git.
- 45 selected Python files compiled in memory without executing historical producers or writing bytecode.
- Four CLI help checks PASS: Random baseline, ShapeGGen PGExplainer, Elliptic GNNExplainer, TASK29 tie evaluator. Generation/training entry points were not invoked.
- `python scripts/verify_release_evidence.py`: PASS. Release payload hashes, 217 retained thesis cells at displayed precision, 400 historical Elliptic records, 400 saved FRESH5 records/4000 pairwise Jaccards, 240 BA extension seed42 records, ShapeGGen counterfactual aggregates and diagnostic counts checked. Float32 probe deltas permit absolute serialization/subtraction error up to 1e-7; other numerical comparisons use 1e-10 absolute / 1e-9 relative tolerance.
- Compact projections re-read independently from original sources: all retained values and original SHA256 hashes match after the declared path normalization. All 400 original FRESH5 record hashes checked. Archived producer strings independently compared.
- 50294 original benchmark/baseline files with prior audit SHA256 rehashed unchanged; original HEADs and tracked user modifications preserved.
- Independent backup: 121 files, CRC and every member SHA256 readback PASS. Backup lives outside repositories and is not staged.
- New payload secret/personal-home-path scan PASS; no new dataset/checkpoint/ZIP/PDF, every new blob below 25 MB. Original project-root and Kaggle runtime paths are scoped historical exceptions, listed in the gate receipt/runbook.
- Documentation reference paths and all retained result paths checked. Two original figure builders remain missing. Historical integrity receipts describe their original runs; the release manifest covers current released bytes.

Exact regression command (repository root; PYTHONPATH points to release `src` and root; PYTHONDONTWRITEBYTECODE=1):

```text
python -m pytest -q -p no:cacheprovider tests/test_random_backbone_dispatch.py tests/test_task29_tie_v1.py tests/test_task29_final_stability.py tests/test_budget.py tests/test_stability.py tests/test_sparsity.py tests/test_fidelity.py tests/test_edge_ground_truth_metrics.py
```

`git diff --check`, exact index allowlist/blob comparison, final staged review and live remote equality are required again before each publication. Their final receipts are summarized in report06. These gates do not establish fresh-machine installation, full experiment reproduction or dataset redistribution rights.

Final staged allowlist/blob/diff checks PASS; six payload commits pushed and remote SHA equality verified. See report06.

Post-push portability correction: 14 inherited result paths initially used Windows-checkout SHA256 in the new manifest. Their existing Git blobs and numerical values were unchanged. The manifest now hashes the exact inherited Git blobs, and `.gitattributes` preserves those bytes on checkout. A clean `git archive` extraction of all 150 scientific payload files passed the standard-library verifier without local fixtures, datasets or checkpoints. Only attributes/manifests/review documentation changed in this correction.
