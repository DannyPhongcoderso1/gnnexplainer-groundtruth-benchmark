# GNN XAI thesis benchmark

Evaluate GNNExplainer, PGExplainer and a Random baseline using GCN and GraphSAGE on BA-House (500/1000/1500 nodes), ShapeGGen V1 and Elliptic. Synthetic graphs have edge ground truth; Elliptic uses perturbation fidelity and stability without explanation ground truth.

## Organization

```text
src/gnnexplainer_benchmark/
  models/{synthetic,elliptic}/  separate backbone families
  explainers/                 accepted GNNExplainer, PGExplainer and Random
  data/                       dataset-specific preparation and splits
  evaluation/                 ground truth, fidelity, stability, sparsity, runtime
scripts/{data,train,evaluate,verify}/
configs/{ba_house,shapeggen,elliptic,shared}/
data/{ba_house,shapeggen,elliptic}/
artifacts/checkpoints/{ba_house,shapeggen,elliptic}/
results/{ba_house,shapeggen,elliptic}/
docs/{methodology,reproducibility,provenance}/
tests/
```

[Source and protocol map](docs/methodology/PROTOCOLS.md) distinguishes synthetic/Elliptic architectures and final protocols. [Artifact access](docs/reproducibility/ARTIFACTS.md), [command guide](docs/reproducibility/RUNBOOK.md) and [result index](results/README.md) provide the main navigation. Historical execution records keep their original paths; [the path catalog](docs/provenance/repository_paths.json) maps them to current files. Canonical module/CLI paths are authoritative; redundant old forwarding files were removed after redirecting active consumers. Historical reports and raw mask caches remain where needed for provenance; they are not required reading.

## Install

Use Python 3.11. Select a suitable PyTorch build, then install the project:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements/runtime.txt -r requirements/dev.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
```

[Windows dependency lock](requirements/lock.txt) and [historical Linux lock](docs/kaggle/task29/historical-linux-lock.txt) describe distinct environments. A fresh-machine installation was not tested during restructuring.

## Check the frozen release

These commands do not train or generate explanations:

```powershell
python scripts/verify/verify_thesis_artifacts.py
python scripts/verify/verify_release_evidence.py
python scripts/verify/checkpoints.py --check-load
```

All **71 final model identities** and **four synthetic graphs** are directly published with original SHA256 values. The one Elliptic graph is intentionally external (75/76 minimum model/graph artifacts in Git). The [mutable artifact catalog](docs/provenance/external_assets.json) and [SHA256 inventory](docs/provenance/artifact_sha256.json) identify exact files. The model checker loads CPU states strictly with `weights_only=True` and makes no forward/training calls.

## Preparation, training and evaluation

Examples below only show help; training and experiment drivers are separate from frozen verification:

```powershell
python scripts/data/generate_ba_house.py --help
python scripts/data/generate_shapeggen.py --help
python scripts/train/train_gcn.py --help
python scripts/train/run_shapeggen_pgexplainer.py --help
python scripts/evaluate/run_batch_gnnexplainer.py --help
python scripts/evaluate/run_random_baseline.py --help
python scripts/evaluate/elliptic_frozen_models.py --help
```

Dataset-specific configs live under [BA-House](configs/ba_house), [ShapeGGen V1](configs/shapeggen/v1) and [Elliptic](configs/elliptic). Checkpoints are under [BA-House](artifacts/checkpoints/ba_house), [ShapeGGen V1](artifacts/checkpoints/shapeggen/v1) and [Elliptic](artifacts/checkpoints/elliptic). Use new output directories; preserve frozen files. The [runbook](docs/reproducibility/RUNBOOK.md) identifies mixed training/evaluation drivers and historical guard dependencies.

## Elliptic dataset and pretrained models

Obtain [Elliptic / Weber et al. (2019)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) from its provider. **Raw CSVs, processed graphs and dataset archives are not redistributed.** The baseline derives from [Dang et al., RMIT-BDSL](https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection); its [MIT notice](docs/provenance/licenses/RMIT_BLOCKCHAIN_ANOMALY_DETECTION_MIT.txt) is retained.

```powershell
python scripts/data/preprocess_elliptic.py --raw-dir D:/user-acquired-elliptic --out data/elliptic/elliptic_data.pt
python scripts/evaluate/elliptic_frozen_models.py --check-load
```

[The Elliptic guide](docs/reproducibility/ELLIPTIC_MODELS.md) describes original preprocessing/temporal splits, frozen backbone loading, historical seed42 versus FRESH5 paths and no-training replay commands. GNNExplainer raw-score replay still needs the separately cataloged cached masks; it has no shared explainer model checkpoint.

## Results and limitations

[Saved results](results/README.md) map to [217 retained thesis cells](results/final/thesis_traceability.json). Numerical verification checks saved hashes/arithmetic, not a new experimental reproduction. Nine BA500 GraphSAGE PGExplainer cells remain withheld because natural106/35 and common48/16 cohorts conflict; both artifact families remain distinctly named. ShapeGGen diagnostic probes are supplementary and do not replace official results. Historical Elliptic seed42 and FRESH5 stability remain separate.

Restructuring changed organization and path resolution, not model algorithms, seeds, checkpoint bytes, graph identities or frozen measurements. No training, explanation generation or full scientific evaluation was run. The existing regression suite includes its original forward checks; the separate checkpoint gate performs no inference. Full fresh-machine/result reproduction and original figure generation are not claimed. See [validation](docs/release_review/RESTRUCTURE_VALIDATION.md), [before/after](docs/release_review/RESTRUCTURE_BEFORE_AFTER.md) and [third-party notices](THIRD_PARTY_NOTICE.md).

Current source cleanup was audited statically only. No tests, project imports, checkpoint loading, inference, optimization or dataset execution were run after cleanup. Earlier runtime results apply to the earlier commit, not a retest of this cleanup. See [the source cleanup report](docs/release_review/FINAL_SOURCE_CLEANUP_REPORT.md).
