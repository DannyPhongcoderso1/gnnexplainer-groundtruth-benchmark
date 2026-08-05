# GNNExplainer Ground-Truth Benchmark

This repository is an initialization scaffold for benchmarking PyTorch Geometric
GNNExplainer on synthetic datasets with ground-truth explanations.

## Current Scope

This phase only sets up:

- an isolated virtual environment;
- stable PyTorch with CUDA support on this machine;
- PyTorch Geometric;
- project structure, configs, scripts, and tests;
- smoke tests for `GCNConv`, `Explainer`, and `GNNExplainer`.

## Not Implemented Yet

- BA-House dataset generation;
- benchmark GCN training;
- ground-truth explanation metrics;
- stability experiments;
- GraphSAGE benchmark;
- Elliptic integration.

## Repository Structure

```text
configs/          Pilot dataset, model, and explainer configs
src/              Package source
scripts/          Environment and PyG XAI smoke scripts
tests/            Import and smoke tests
artifacts/        Ignored local data/checkpoint artifacts
results/          Ignored generated explanation/metric/figure outputs
docs/             Environment report and project plan
requirements/     Runtime, dev, and lock files
```

## Environment Setup

PyTorch is installed first using the official PyTorch wheel index for the selected
compute target. On this machine the selected target is CUDA.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements\runtime.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## Run Checks

Environment report:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --output docs\environment_report.json --compute-target cuda
```

PyG GNNExplainer smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_pyg_xai.py
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Ruff:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

## Planned Next Phases

1. Add PyG `ExplainerDataset` BA-House generation.
2. Validate ground-truth masks.
3. Train a small benchmark GCN.
4. Run PyG GNNExplainer and save raw masks.
5. Evaluate edge ROC-AUC, AP, Precision@k, Recall@k, Jaccard@k, faithfulness, sparsity, and runtime.
6. Run stability experiments across explainer and model seeds.
7. Add GraphSAGE.
8. Add attributed synthetic benchmark with feature ground truth.
9. Transfer validated generic XAI modules to an Elliptic feature branch.

Synthetic benchmarks do not prove that GNNExplainer is universally reliable. They
provide controlled checks for ground-truth overlap, faithfulness, sparsity, and
stability before transfer to harder real-world data.

## Reproducibility

The locked environment is stored in:

```text
requirements/lock.txt
```

The sanitized environment report is stored in:

```text
docs/environment_report.json
```

## License

TODO: choose and add a project license before public release.
