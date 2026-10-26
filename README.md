# GNN XAI benchmark

Ground-truth and perturbation evaluation of GNNExplainer, PGExplainer and a Random baseline on GCN and GraphSAGE. The thesis studies BA-House (500, 1000 and 1500 nodes), frozen ShapeGGen V1, and Elliptic.

This release is **PARTIAL for scientific reproducibility**: saved evidence is available, while some original inputs/checkpoints require access. BA500 GraphSAGE PGExplainer has a natural/common training-pool conflict in the thesis; nine affected cells are withheld from the release trace. No model training or fresh experiment was performed for this release.

## Install

Use Python 3.11 in a new environment. Install a PyTorch build suitable for the machine before PyG:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements/runtime.txt -r requirements/dev.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
```

[requirements/lock.txt](requirements/lock.txt) records the historical Windows environment; [the TASK29 lock](docs/kaggle/task29/historical-linux-lock.txt) records the separate Linux run. Installation on a fresh machine was not tested in this release.

## Results and verification

Run the standard-library verifier without training or downloading datasets:

```powershell
python scripts/verify_release_evidence.py
```

[The thesis trace](results/final/thesis_traceability.json) maps 217 retained numerical cells across 20 tables and two figures to original evidence. [The release manifest](docs/provenance/release_manifest.json) provides hashes and derivation provenance. Original frozen result paths remain in place. [Release review](docs/release_review/04_VERIFICATION_REPORT.md) records tests and limitations.

The verifier checks release hashes, displayed numerical precision, saved Elliptic final200 perturbation arithmetic, all 400 FRESH5 records and 4000 Jaccard comparisons, BA extension seed42 identities, and ShapeGGen diagnostic counts. A successful check verifies saved evidence, not full model reproduction. Original figure builders remain unlocated.

## Running experiments

[The runbook](docs/reproducibility/RUNBOOK.md) explains entry points, required assets and historical execution constraints. Example help commands:

```powershell
python scripts/run_random_baseline.py --help
python scripts/run_shapeggen_pgexplainer.py --help
python scripts/run_elliptic_gnnexplainer.py --help
```

Acquire restricted assets before executing experiment commands. Existing frozen output directories must be preserved; historical scripts can refuse to overwrite them.

## Dependencies and interpretation

Elliptic backbones remain a separate [repository dependency](https://github.com/DannyPhongcoderso1/elliptic-gnn-xai), pinned at `95710a9852019104f90499cdba628c37f4483a2a` in its migration lineage; the exact verified SHA is recorded in the manifest and runbook. GraphXAI provenance is recorded separately. See [third-party notices](THIRD_PARTY_NOTICE.md) and [external assets](docs/provenance/external_assets.json). Repository access does not establish permission to redistribute datasets, weights or original project code under a new license.

ShapeGGen V1 diagnostics and failed size/warmup interventions do not replace official results or prove a universal causal mechanism. Elliptic historical RAW ranking and FRESH5 seeds 0–4 are distinct from later diagnostic variants and the historical seed42 checkpoint. FRESH5 exact-tie handling was an evaluation-only addendum after observation; interpret its stability with that limitation.
