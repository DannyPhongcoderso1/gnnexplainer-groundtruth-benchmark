# Elliptic dataset and pretrained models

The repository publishes 14 exact model artifacts: two frozen classifiers, two historical PGExplainer seed42 files and ten FRESH5 PG states. It does **not** redistribute Elliptic CSVs, processed graphs or their archives. Checkpoint-to-backbone/seed/source mapping and SHA256 are in `docs/provenance/elliptic_model_publication.json`; current logical resolution is in `external_assets.json`.

## Sources and licenses

- [Elliptic dataset / Weber et al. (2019)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), [paper](https://arxiv.org/abs/1908.02591).
- [Dang et al. baseline repository](https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection), MIT notice preserved at `docs/provenance/licenses/RMIT_BLOCKCHAIN_ANOMALY_DETECTION_MIT.txt`. Minimal architecture and preprocessing source comes from the pinned migrated baseline commit 95710a9852019104f90499cdba628c37f4483a2a, with upstream lineage 83ef2b9ed8b3233175bbc3f4231c593c365225b9. The initial publication normalized whitespace; repository organization subsequently extracted the reusable dataset/architecture modules and changed CLI imports/path resolution. All four backbone class ASTs and the complete Elliptic dataset module AST match the source release; preprocessing computation is unchanged.
- Kaggle metadata records CC BY-NC-ND4.0 for the dataset. That is separate from the MIT code license and the owner's authorization of their model artifacts. No unrestricted commercial model/data license is asserted. Model inspection found learned parameter/optimizer tensors and run/target-index metadata, not embedded features, labels, graph or transaction records. This is a model-only noncommercial thesis release, not redistribution or relicensing of the dataset.

## Acquire and preprocess locally

Obtain the three original CSVs through the official Kaggle dataset page under its terms: `elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`. Keep them outside Git. Do not commit the download archive.

```powershell
python scripts/data/preprocess_elliptic.py --raw-dir D:/user-acquired-elliptic --out data/elliptic/elliptic_data.pt
```

This is the existing preprocessing code, not a training command. It preserves original row/edge ordering, adds reverse edges for the undirected model, drops self-loops/duplicates, maps classes1→1,2→0,unknown→2, and excludes unknowns from supervised masks. Splits: time<30 train,30≤time<40 validation,time≥40 test. The timestep is removed from model features, leaving165 features; it remains as split metadata. Code and source hashes are recorded in the publication manifest.

The owner tested this command on already acquired raw CSVs in a temporary output directory:203769 nodes,468710 directed edge entries,165 features, and the exact original processed-file SHA256 `f807b83a7c739b16f69b5676226ce441b64d5fe8de95506a7727f86e8451291b`. All seven tensor fingerprints also matched. Neither input nor generated graph was staged/uploaded. A new installation/download on a separate machine was not tested. The portable replay command checks frozen tensor fingerprints, so harmless serialization differences cannot bypass edge/feature/label/split identity checks.

## Hash and load checks without any dataset

```powershell
python scripts/evaluate/elliptic_frozen_models.py
python scripts/evaluate/elliptic_frozen_models.py --check-load
```

The first command verifies14 model hashes and exact target CSV identities without loading a dataset. The second uses CPU `weights_only=True` loading and the known PyTorch TorchVersion string class; unknown globals fail, with no unsafe fallback. It strictly loads two original-compatible backbones and twelve PyG PGExplainer states. PyG, Torch and other dependencies remain pinned in `requirements/lock.txt`; the separate historical Linux lock is under `docs/kaggle/task29/`. Install using the root README before model loading.

Weights are in:

- `artifacts/checkpoints/elliptic/backbones/{gcn,graphsage}.pt`
- `artifacts/checkpoints/elliptic/pgexplainer/historical_seed42/{gcn,graphsage}.zip`
- `artifacts/checkpoints/elliptic/pgexplainer/fresh5/{gcn,graphsage}/seed_0.pt` through `seed_4.pt`

The historical `.zip` files are native Torch serialization archives, not full training packages. FRESH5 publication extracts only the original `checkpoint.pt` member from each COMPLETE ZIP; its original SHA256 is preserved. Final checkpoints retain their original incidental optimizer/config/target-index metadata because bytes must not be reconstructed. Full COMPLETE logs, epoch histories and standalone resume files are not published.

## Inference/evaluation without training

The adapter delegates to existing explanation-compatible architecture classes, the installed frozen PyG PGExplainer builder, two-hop candidate mapping, original RAW selection policies and existing fidelity evaluator. The frozen PyG constructor creates an Adam optimizer, but replay never steps or restores it and never invokes the PG training API or substitutes fresh states for seed42. GCN perturbation inference uses the pre-existing cache-disabled compatible wrapper; its numerical parity with the MIT architecture was unit-tested. Canonical normalized-LF target hashes support the original Windows/Linux CSV line endings without changing node order.

Use new output paths; existing files are rejected:

```powershell
python scripts/evaluate/elliptic_frozen_models.py --graph data/elliptic/elliptic_data.pt --backbone graphsage --family historical_seed42 --output outputs/elliptic_historical_sage.json
python scripts/evaluate/elliptic_frozen_models.py --graph data/elliptic/elliptic_data.pt --backbone gcn --family fresh5 --output outputs/elliptic_fresh5_gcn.json
```

Historical replay uses only seed42, RAW float32 scores and the original top-k policy, reporting the existing fidelity/sparsity definitions. FRESH5 defaults to all five seeds and uses the exact-tie global-ID policy from the original addendum, reporting per-target mean pairwise Jaccard; `--seed` selects a single fresh state, not a new training run. `--max-targets` is an optional reader smoke limit and must not be presented as the full 200-target result.

Before this static source cleanup, the model-load, architecture parity, targets and preprocessing tests passed. They were not rerun after cleanup; current runtime behavior is NOT_VERIFIED. The full real-graph200-target replay was not run for this publication; CPU/CUDA numeric differences may affect score ties. Use the historical Linux/T4 environment for comparison with frozen results, and compare any replay to the saved evidence before citing it. No fresh-clone full-reproduction claim is made.

Saved final numerical records and `python scripts/verify/verify_release_evidence.py` remain usable without a dataset. GNNExplainer raw-score/perturbation replay still requires its external cached edge masks, original graph and mappings; no shared GNNExplainer model checkpoint exists. The original historical guard drivers may separately require their COMPLETE/Release002 provenance archives. The new model loader does not need those full packages. This release does not add raw masks or redistribute the dataset.
