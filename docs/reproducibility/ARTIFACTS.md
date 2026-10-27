# Frozen artifact verification and restoration

The minimum catalog has 71 model artifacts and five graphs. There are75 directly in Git, including all 71 model artifacts and four synthetic graphs. The single Elliptic graph is intentionally not redistributed; acquire it from the provider and preprocess locally. Do not equate a local package with public access.

## Clone and install

```powershell
git clone --branch release/thesis-final-20261026 --single-branch https://github.com/DannyPhongcoderso1/gnnexplainer-groundtruth-benchmark.git
cd gnnexplainer-groundtruth-benchmark
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements/runtime.txt -r requirements/dev.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
```

Use the existing Windows lock or the separate TASK29 Linux lock as the historical environment reference. Fresh-machine installation and full model evaluation were not rerun for this publication.

## Verify without loading tensors

```powershell
python scripts/verify_thesis_artifacts.py
python scripts/verify_release_evidence.py
```

The first command uses only the standard library: no torch import, pickle deserialization, inference or download. A repository-only checkout resolves75/76 and explicitly reports the single dataset as `EXTERNAL_ACCESS_REQUIRED`. `--require-complete` returns a nonzero status until the original byte-identical local graph is available; the Elliptic-specific loader also verifies canonical tensors for local preprocessing outputs. Original frozen source hashes and exact artifact identities are recorded in `docs/provenance/frozen_artifact_sources.json` and `docs/provenance/artifact_sha256.json`.

## Elliptic data is acquired, not redistributed

All 14 frozen Elliptic model files are now directly published. Only the graph remains outside Git by explicit user policy. Follow [the Elliptic guide](ELLIPTIC_MODELS.md) to acquire the original CSVs, run the unchanged preprocessing/temporal split, verify model hashes and use the frozen model-only loader. The original private bundle remains an owner-local backup, not a public dataset distribution channel. Historical COMPLETE guard archives remain optional dependencies of the old guard drivers; the new model loader needs only published weights and configs.

## GNNExplainer caches and Random

GNNExplainer has no shared model checkpoint. Existing final per-node/per-seed/pairwise numerical records suffice to check saved aggregates. Independently recomputing score-ranking metrics or perturbation outputs needs frozen raw edge scores, graph, target mapping and backbone. `docs/provenance/gnn_cached_artifacts.json` lists all 5268 edge-evaluation files with exact SHA256 and paths; 2076 were already published, 3192 require access. No new raw mask was uploaded.

An owner with approved cache access can verify all paths without copying:

```powershell
python scripts/verify_thesis_artifacts.py --verify-masks --cache-root D:/approved-access/benchmark-cache
```

To restore an explicitly chosen dataset cache into a new directory:

```powershell
python scripts/verify_thesis_artifacts.py --restore-masks --cache-root D:/approved-access/benchmark-cache --destination-root D:/thesis-cache --dataset BA1000
```

Every omitted raw file uses its original relative path and exact hash; there is no claimed public cache URL. Node masks are optional for the edge-only reported metrics. Random has no learned checkpoint; retain the published evaluator, target identities, seed protocol and numerical outputs.

## Frozen evaluation scope

Saved-evidence verification above checks 217 retained thesis cells, final200 arithmetic, FRESH5 Jaccards and other retained measurements without optimization. BA500 GCN classifier inference can use `python scripts/evaluate_gcn.py` after artifact verification. Its frozen PG evaluator can run without training:

```powershell
python scripts/evaluate_pgexplainer_gcn.py --device cpu --output-root outputs/frozen_pg_gcn --metrics-output-dir outputs/frozen_pg_gcn_metrics
```

Run model evaluators only on these hash-verified trusted artifacts. Use new output directories; do not replace frozen results. These model CLI examples were inspected, not executed as a new thesis experiment here. BA extension generation/training drivers and ShapeGGen multi-stage drivers include training; do not invoke them merely to restore weights. Other historical inference functions remain in the existing producer source, with their original guards; a unified fresh-clone full-forward evaluation across every cell was not validated. Exact asset resolution and saved-record verification are tested; full scientific reproducibility remains PARTIAL.

Nine BA500 GraphSAGE PG cells remain withheld because natural106/35 and common48/16 are mixed in the thesis. Both checkpoint cohorts retain their true identity. BA extensions use primary seed42 and independent Stability seeds0–4. ShapeGGen graph/split seed912 differs from model seed42 and PG/explanation seeds. Optional diagnostic extensions are not official V1 results.
