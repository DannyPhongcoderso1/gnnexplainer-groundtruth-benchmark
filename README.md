# GNN Explainability Benchmark

**A comparative study of GNNExplainer and PGExplainer across GCN and GraphSAGE backbones, with a Random reference baseline.**

This repository contains the source code, frozen model checkpoints, experiment configurations, saved evaluation results, and provenance records associated with a graduation thesis on graph neural network (GNN) explainability.

## Overview

| Dataset | Backbone models | Explainers | Evaluation focus |
| --- | --- | --- | --- |
| BA-House (500, 1,000, and 1,500 nodes) | GCN, GraphSAGE | GNNExplainer, PGExplainer, Random | Edge-ground-truth recovery and explanation behavior |
| ShapeGGen V1 | GCN, GraphSAGE | GNNExplainer, PGExplainer, Random | Edge-ground-truth recovery and explanation behavior |
| Elliptic | GCN, GraphSAGE | GNNExplainer, PGExplainer | Fidelity and stability; no edge-level explanation ground truth |

Evaluation records include **Edge AUROC**, **Average Precision (AP)**, **Fidelity+/Fidelity−**, **Stability**, **Sparsity**, and **Runtime** where applicable. Random is a reference baseline rather than a learned explanation method.

## Repository structure

This is an abbreviated view of the current Git tree. Historical producers, supporting caches, and other provenance files are retained but not all shown.

```text
gnnexplainer-groundtruth-benchmark/
├── src/gnnexplainer_benchmark/
│   ├── models/
│   │   ├── synthetic/            # GCN and GraphSAGE for BA-House and ShapeGGen
│   │   └── elliptic/             # Elliptic-specific GCN and GraphSAGE
│   ├── explainers/               # Explanation implementations and Random baseline
│   ├── data/                     # Dataset loading, preparation, splitting, targets
│   ├── evaluation/               # Recovery, fidelity, stability, sparsity, runtime
│   ├── explain/                  # Candidate-edge mapping and explanation I/O
│   ├── training/                 # Reusable trainers and checkpoint loading
│   └── utils/                    # Shared utilities
├── src/graphxai/                 # Vendored GraphXAI dependency for ShapeGGen
├── scripts/
│   ├── data/                     # Generation and preprocessing entry points
│   ├── train/                    # Backbone and explainer training entry points
│   ├── evaluate/                 # Explanation/evaluation entry points
│   └── verify/                   # Frozen-artifact and evidence checks
├── configs/
│   ├── ba_house/                 # BA500, BA1000, BA1500
│   ├── shapeggen/                # Frozen V1
│   ├── elliptic/                 # Frozen Elliptic protocols
│   └── shared/                   # Shared configuration
├── data/
│   ├── ba_house/                 # Published synthetic graphs
│   ├── shapeggen/                # Published frozen V1 graph
│   └── elliptic/                 # Acquisition instructions; dataset not included
├── artifacts/checkpoints/
│   ├── ba_house/                 # By graph size, backbone, explainer, seed
│   ├── shapeggen/                # V1 backbone and PGExplainer checkpoints
│   └── elliptic/                 # Backbones, historical seed42, FRESH5
├── results/
│   ├── ba_house/
│   ├── shapeggen/
│   ├── elliptic/                 # Frozen numerical evidence
│   └── final/                    # Unchanged thesis-to-result trace
├── docs/
│   ├── methodology/              # Research protocols and source map
│   ├── reproducibility/          # Artifact access and command guides
│   ├── provenance/               # Artifact hashes, path mapping, source licenses
│   └── release_review/           # Source audits and publication receipts
├── requirements/                # Runtime/dev dependencies and environment locks
├── tests/                       # Existing regression checks
├── pyproject.toml
└── README.md
```

**Where to look first:**

- [Research protocols and source map](docs/methodology/PROTOCOLS.md)
- [Commands and experiment entry points](docs/reproducibility/RUNBOOK.md)
- [Artifact access and restoration](docs/reproducibility/ARTIFACTS.md)
- [Frozen results index](results/README.md)
- [Historical-to-current path mapping](docs/provenance/repository_paths.json)

The `models/synthetic/` and `models/elliptic/` directories intentionally use separate backbone implementations. Historical source paths are preserved as provenance in the path catalog rather than treated as current entry points.

## Installation

### Prerequisites

- Python **3.11**
- Git
- A compatible PyTorch installation (CPU or suitable CUDA build)

The commands below target **Windows PowerShell**. Run them from the repository root.

### 1. Clone the repository

```powershell
git clone --branch refactor/repository-organization https://github.com/DannyPhongcoderso1/gnnexplainer-groundtruth-benchmark.git
cd gnnexplainer-groundtruth-benchmark
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 3. Install PyTorch

Choose a build compatible with your system using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). For the CUDA 12.8 environment used in the project's Windows setup:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If CUDA 12.8 is not appropriate for your system, use the corresponding installation command from PyTorch instead.

### 4. Install project dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/runtime.txt -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The project requires Python **>=3.11,<3.12**, as declared in `pyproject.toml`. The commands above install unpinned package versions from the requirements files; they do not recreate the historical environment exactly. The [Windows dependency lock](requirements/lock.txt) and the [historical Linux lock](docs/kaggle/task29/historical-linux-lock.txt) record the separate environments used for the frozen work. A fresh-machine installation has **not** yet been verified for this reorganized branch.

## Verify the published artifacts

These checks use saved files; they do **not** train models or regenerate explanations.

```powershell
.\.venv\Scripts\python.exe scripts/verify/verify_thesis_artifacts.py
.\.venv\Scripts\python.exe scripts/verify/verify_release_evidence.py
.\.venv\Scripts\python.exe scripts/verify/checkpoints.py --check-load
```

The release contains **71 final model artifacts** and **four synthetic graphs**, all published with recorded SHA256 identities. The processed Elliptic graph is intentionally not distributed. See the [artifact catalog](docs/provenance/external_assets.json) and [hash inventory](docs/provenance/artifact_sha256.json).

The checkpoint checker supports restricted CPU loading with `weights_only=True` without running inference or training. Use checkpoints only from trusted sources.

## Data preparation

### BA-House and ShapeGGen V1

Synthetic graph artifacts are included. Their generators and preprocessing entry points are organized under `scripts/data/`, while frozen generation settings live under `configs/`.

To inspect available options without generating data:

```powershell
.\.venv\Scripts\python.exe scripts/data/generate_ba_house.py --help
.\.venv\Scripts\python.exe scripts/data/generate_shapeggen.py --help
```

For the BA500/1000/1500 extension workflows, consult the [runbook](docs/reproducibility/RUNBOOK.md) and dataset-specific [BA-House configurations](configs/ba_house/).

### Elliptic

**Elliptic data are not included in this repository.** Obtain the dataset from the [official Elliptic dataset page](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and observe the provider's terms.

The Elliptic backbone implementation is based on the [Dang et al. / RMIT-BDSL baseline repository](https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection), with its applicable [license notice](docs/provenance/licenses/RMIT_BLOCKCHAIN_ANOMALY_DETECTION_MIT.txt) preserved.

After obtaining the original dataset, the documented preprocessing entry point is:

```powershell
.\.venv\Scripts\python.exe scripts/data/preprocess_elliptic.py --raw-dir D:/user-acquired-elliptic --out data/elliptic/elliptic_data.pt
```

Replace `D:/user-acquired-elliptic` with the path to your legitimately obtained source data. Do not commit the generated graph, raw CSVs, or dataset archives. Details of temporal splitting, expected file identity, and checkpoint usage are in the [Elliptic model guide](docs/reproducibility/ELLIPTIC_MODELS.md).

## Training and evaluation entry points

The repository preserves the original frozen results; **retraining is not needed to inspect them**. Scripts are grouped by purpose:

| Purpose | Location | Examples |
| --- | --- | --- |
| Generate / preprocess data | `scripts/data/` | BA-House, ShapeGGen, Elliptic |
| Train models and explainers | `scripts/train/` | GCN, PGExplainer |
| Evaluate explanations | `scripts/evaluate/` | GNNExplainer, Random, Elliptic frozen models |
| Verify frozen evidence | `scripts/verify/` | Hashes, saved metrics, checkpoint discovery |

Inspect CLI options without starting training or evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/train/train_gcn.py --help
.\.venv\Scripts\python.exe scripts/train/run_shapeggen_pgexplainer.py --help
.\.venv\Scripts\python.exe scripts/evaluate/run_batch_gnnexplainer.py --help
.\.venv\Scripts\python.exe scripts/evaluate/run_random_baseline.py --help
.\.venv\Scripts\python.exe scripts/evaluate/elliptic_frozen_models.py --help
```

Use the [runbook](docs/reproducibility/RUNBOOK.md) for dataset-specific arguments, original execution guards, and the distinction between historical and current entry points. Use new output directories to avoid overwriting frozen artifacts.

## Checkpoints and results

Checkpoints are grouped by dataset under [`artifacts/checkpoints/`](artifacts/checkpoints/):

- **BA-House:** BA500, BA1000, BA1500; backbone and PGExplainer states by seed.
- **ShapeGGen V1:** frozen GCN/GraphSAGE backbone and PGExplainer states.
- **Elliptic:** two backbone checkpoints, historical PGExplainer `seed42`, and **FRESH5** PGExplainer checkpoints (`seeds 0–4` for each backbone).

**FRESH5** consists of five independent PGExplainer training seeds per frozen Elliptic backbone, used for Stability analysis. It does not represent ten new backbone models. GNNExplainer has no shared trained explainer checkpoint; its per-target explanation caches are managed separately.

Results and their provenance can be found at:

- [Result index](results/README.md)
- [Thesis-to-result traceability](results/final/thesis_traceability.json)
- [Artifact SHA256 inventory](docs/provenance/artifact_sha256.json)

## Reproducibility scope and limitations

- Saved numerical evidence is checked against **217 retained thesis cells**. This verification is not equivalent to independently repeating all forward passes and explanation optimizations.
- **Nine BA500–GraphSAGE–PGExplainer cells** remain withheld from the retained thesis trace because natural-pool `106/35` and common-pool `48/16` experimental cohorts have different provenance. These records must not be relabeled as one another.
- The original plotting scripts for two thesis figures have not been recovered.
- Full raw-score replay may require separately cataloged GNNExplainer cache files, and Elliptic evaluation requires provider-acquired data and its exact preprocessing.
- The latest source cleanup was audited **statically only**; tests and model execution were not rerun after that cleanup. Earlier tests apply to earlier revisions.
- A complete fresh-machine installation and full end-to-end experiment reproduction are **not claimed**.

For background, see the [protocol map](docs/methodology/PROTOCOLS.md), [artifact access guide](docs/reproducibility/ARTIFACTS.md), [source cleanup report](docs/release_review/FINAL_SOURCE_CLEANUP_REPORT.md), and [third-party notices](THIRD_PARTY_NOTICE.md).
