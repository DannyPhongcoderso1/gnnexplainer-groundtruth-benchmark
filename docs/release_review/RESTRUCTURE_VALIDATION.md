# Restructure validation

No model training, explanation generation, full evaluation or scientific result recomputation was performed. The pre-existing regression tests retain their original forward checks; the checkpoint gate itself makes zero forward/training calls.

| Gate | Verified result |
| --- | --- |
| Frozen model identities |71/71 exact SHA256/size; all71 restricted weights-only CPU loads and strict state-dict compatibility PASS |
| Graph identity |4/4 published synthetic graph SHA256 values unchanged; Elliptic data remains external |
| Model source |Four original class ASTs equal after relocation; synthetic and Elliptic families remain separate; old import aliases are identical |
| Dataset/configs |Complete Elliptic dataset-module AST unchanged;43 accepted config/target files preserve values/IDs, allowing only current path/class references |
| Relocated frozen content |239 model/graph/result/metadata/historical-report files byte-identical to the immutable source release |
| Regression |Original71 tests plus five path/import tests:76 passed, two existing Torch deprecation warnings,6.44s |
| CLI |33 current entry points and28 historical aliases pass --help; source imports resolve from this repository |
| Static checks |207 Python modules compile; Ruff passes43 current/canonical modules and targeted tests |
| Current docs |79 relative links in current README/guides/result/provenance navigation resolve |
| Saved evidence |413 payload hashes,217 trace cells,400 historical Elliptic records,400 FRESH5 records/4000 pairwise comparisons and240 BA seed42 records PASS |

Exact staging PASS:713 explicitly reviewed source/destination paths;337 added and285 removed Git nodes (including relocations),33,740 pre-existing scientific Git blobs unchanged,0 dataset files,0 secret hits. Complete path classification covers34,273 original paths and19 new support entries:34,292 CSV rows. Git-tree extraction and final remote receipts are recorded in RESTRUCTURE_GIT_REPORT.md. All original results, source receipts, target records and protected hashes remain auditable through the current catalog, unchanged execution records and immutable source release.

```text
python -m pytest -q -p no:cacheprovider tests/test_random_backbone_dispatch.py tests/test_task29_tie_v1.py tests/test_task29_final_stability.py tests/test_budget.py tests/test_stability.py tests/test_sparsity.py tests/test_fidelity.py tests/test_edge_ground_truth_metrics.py tests/test_artifact_resolution.py tests/test_elliptic_model_publication.py tests/test_repository_organization.py
python scripts/verify/checkpoints.py --check-load
python scripts/verify/verify_release_evidence.py
python scripts/verify/verify_thesis_artifacts.py
```

The one CLI issue found by structural checks was an unqualified legacy Elliptic helper import; it was changed to the explicit scripts.prepare_elliptic_task26 import. No scientific behavior was changed to make tests pass. Existing71 assertions were retained, with path-only fixture references updated where files moved.

Unresolved scientific limitations are unchanged: nine BA500 GraphSAGE PG cohort-conflict cells remain withheld; independent raw GNN replay needs separately accessible caches; Elliptic dataset is intentionally not distributed; original figure builders remain unlocated. Historical guarded training producers retain their original execution-path/dependency constraints. These do not block current model/hash/saved-evidence verification. No fresh-machine installation or full result reproduction is claimed.
