# Elliptic model SHA256 and regression verification — 2026-10-27

- Original inventory:14/14 exact model bytes matched SHA256 and size, including ten original ZIP members. New model-load checks strictly load two backbones and twelve PG states. Backbone/seed metadata,30 completed PG epochs, dataset hash and exact train/validation/final target identities are checked.
- Dataset/secret scan:14/14 restricted weights-only loads PASS;206 model/optimizer tensor entries,0 graph-like tensors,0 arbitrary object types,0 credential/private-home-path string hits. Full model/shape/hash details are in the publication manifest. No original model bytes were edited.
- Regression: existing 66 tests plus five new publication tests PASS:71 total, two existing Torch deprecation warnings,7.89s. New tests cover14 strict loads, seed-family separation, MIT-architecture parity, target-order/config identity, hash substitution rejection and raw-preprocessing temporal split on a synthetic fixture. No training call or optimizer creation is part of the new adapter.
- Owner raw-data preprocessing validation ran into a temporary directory outside Git:203769 nodes,468710 directed edge entries,165 model features; all seven tensor fingerprints and the exact original processed-file hash f807b83a7c739b16f69b5676226ce441b64d5fe8de95506a7727f86e8451291b matched. The generated graph and raw inputs are not staged/uploaded. No full real-graph200-target model replay or fresh-machine dependency installation was performed.
- New-source AST/import/Ruff checks PASS. Final release evidence verification PASS:248 payload hashes,217 retained trace cells,400 historical Elliptic node records,400 FRESH5 records/4000 Jaccard comparisons and240 BA seed42 records. All114 previously protected scientific/result artifacts remain byte-identical. Staged allowlist:38 files,14 model binaries/3,480,122 bytes,0 dataset files,0 secret hits; index bytes equal the reviewed files. A clean extraction of the Git tree passed evidence/hash verification and strict CPU loading of all14 models without any dataset or baseline checkout. This is an extracted-tree smoke check using the owner dependency environment, not a fresh-machine install. Source repository HEAD/index and existing user-modified files are unchanged. Actual push verification is recorded in report15.

Regression command:

```text
python -m pytest -q -p no:cacheprovider tests/test_random_backbone_dispatch.py tests/test_task29_tie_v1.py tests/test_task29_final_stability.py tests/test_budget.py tests/test_stability.py tests/test_sparsity.py tests/test_fidelity.py tests/test_edge_ground_truth_metrics.py tests/test_artifact_resolution.py tests/test_elliptic_model_publication.py
```

A first adapter test caught the overloaded PyG PGExplainer.train API; evaluation now freezes/evaluates only the deterministic MLP without calling the optimization API. Static line-length findings were fixed before the passing suite. No training was performed by the failing check.
