# GNN XAI benchmark

Ground-truth and perturbation evaluation of GNNExplainer, PGExplainer and a Random baseline on GCN and GraphSAGE. The thesis studies BA-House (500, 1000 and 1500 nodes), frozen ShapeGGen V1 (graph/split seed912), and Elliptic.

This release is **PARTIAL for scientific reproducibility**: 75 of the 76 minimum frozen model/graph artifacts are now in Git: all 71 model artifacts and four synthetic graphs. The Elliptic dataset is intentionally excluded and must be acquired from its provider. BA500 GraphSAGE PGExplainer has a natural/common training-pool conflict in the thesis; nine affected cells are withheld from the release trace. No model training or fresh experiment was performed for this release.

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
python scripts/verify_thesis_artifacts.py
python scripts/verify_release_evidence.py
```

[The thesis trace](results/final/thesis_traceability.json) maps 217 retained numerical cells across 20 tables and two figures to original evidence. [The release manifest](docs/provenance/release_manifest.json) provides hashes and derivation provenance. Original frozen result paths remain in place. [Release review](docs/release_review/04_VERIFICATION_REPORT.md) records tests and limitations.

The verifier checks release hashes, displayed numerical precision, saved Elliptic final200 perturbation arithmetic, all 400 FRESH5 records and 4000 Jaccard comparisons, BA extension seed42 identities, and ShapeGGen diagnostic counts. A successful check verifies saved evidence, not full model reproduction. Original figure builders remain unlocated.

## Elliptic Dataset and Pretrained Models

Dataset source: [Elliptic / Weber et al. (2019)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). Baseline source: [Dang et al., RMIT-BDSL](https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection). **The original and processed Elliptic dataset are not redistributed.** Acquire the CSVs from Kaggle and run the included original preprocessing/temporal-split pipeline locally.

Frozen backbones: `artifacts/elliptic/backbones/`. Historical PG seed42: `artifacts/elliptic/pgexplainer/historical_seed42/`. FRESH5 seeds0â€“4: `artifacts/elliptic/pgexplainer/fresh5/`. All 14 files preserve original checkpoint bytes and SHA256. No full COMPLETE package is uploaded.

```powershell
python scripts/elliptic_frozen_models.py --check-load
```

See [dataset acquisition, SHA checks and frozen inference](docs/reproducibility/ELLIPTIC_MODELS.md) for exact architectures, config/target references and no-training replay commands. Full real-graph replay still needs an independently acquired dataset; GNN raw-score replay also needs omitted cached masks. Hash/model-load/preprocessing tests passed; a fresh-machine dependency install and full 200-target forward replay were not tested.

## Frozen artifact access

Clone with `git clone --branch release/thesis-final-20261026 --single-branch https://github.com/DannyPhongcoderso1/gnnexplainer-groundtruth-benchmark.git`. [The artifact guide](docs/reproducibility/ARTIFACTS.md) explains current 75/76 resolution, the single intentional dataset exclusion and cache access.

## Running experiments

[The runbook](docs/reproducibility/RUNBOOK.md) explains entry points, required assets and historical execution constraints. Example help commands:

```powershell
python scripts/run_random_baseline.py --help
python scripts/run_shapeggen_pgexplainer.py --help
python scripts/run_elliptic_gnnexplainer.py --help
```

Acquire restricted assets before executing experiment commands. Existing frozen output directories must be preserved; historical scripts can refuse to overwrite them.

## Dependencies and interpretation

Elliptic backbone training provenance comes from the separate [baseline repository](https://github.com/DannyPhongcoderso1/elliptic-gnn-xai), pinned at `95710a9852019104f90499cdba628c37f4483a2a` in its migration lineage; the exact verified SHA is recorded in the manifest and runbook. GraphXAI provenance is recorded separately. See [third-party notices](THIRD_PARTY_NOTICE.md) and [external assets](docs/provenance/external_assets.json). Repository access does not establish permission to redistribute datasets, weights or original project code under a new license.

ShapeGGen V1 diagnostics and failed size/warmup interventions do not replace official results or prove a universal causal mechanism. Elliptic historical RAW ranking and FRESH5 seeds 0â€“4 are distinct from later diagnostic variants and the historical seed42 checkpoint. FRESH5 exact-tie handling was an evaluation-only addendum after observation; interpret its stability with that limitation.
