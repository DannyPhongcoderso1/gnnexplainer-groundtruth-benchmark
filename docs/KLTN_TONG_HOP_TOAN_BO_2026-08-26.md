# Full Undergraduate Thesis Summary

Summary date: 2026-08-26

This document summarizes the full thesis progress from the Codex working thread. It covers the Elliptic baseline, the synthetic XAI benchmark repository, BA-House, ShapeGGen, completed phases, frozen metrics, methodological lessons, current task status, and the next work items.

## 1. Overall Thesis Direction

The thesis currently focuses on:

- Reproducing two graph neural network baselines, `GCN` and `GraphSAGE`, on the Elliptic Bitcoin transaction dataset.
- Building a separate benchmark repository for evaluating GNN explainability methods on synthetic graphs with ground-truth explanations.
- Comparing `GNNExplainer`, `PGExplainer`, and a Random explanation baseline.
- Evaluating explanations with multiple complementary metrics instead of a single combined score.
- Extending from synthetic ground-truth datasets to the real-world Elliptic setting after the synthetic protocol is stable.

Core interpretation rules:

- Synthetic datasets such as `BA-House` and `ShapeGGen` provide ground-truth edge explanations, so `Edge AUROC` and `Edge AP` are valid explanation-quality metrics.
- `Elliptic` does not currently provide ground-truth explanatory edge masks, so explanation `Edge AUROC` and `Edge AP` must not be reported for Elliptic.
- The primary comparison is between explainers within the same frozen backbone.
- Cross-backbone comparisons between `GCN` and `GraphSAGE` are backbone-context observations, not causal claims.

## 2. Active Repositories

| Repository | Role | Current status |
|---|---|---|
| `D:\CS\KLTN\elliptic-gnn-xai` | Elliptic GCN and GraphSAGE baseline repository | Replaced with validated GCN/GraphSAGE baseline pipeline |
| `D:\CS\KLTN\GCN_GraphSAGE_base\Blockchain-Anomaly-Detection` | Source repository for Elliptic baseline code | Used as baseline reference source |
| `D:\CS\KLTN\gnnexplainer-groundtruth-benchmark` | Main XAI synthetic benchmark repository | BA-House complete, currently after ShapeGGen Task22 |
| `D:\CS\KLTN\docs` | Super-plan and task instruction directory | Following the revised Task20-Task33 V3 plan |

Current main XAI repository state:

```text
D:\CS\KLTN\gnnexplainer-groundtruth-benchmark
Branch: main
Latest commit: bed705d Freeze ShapeGGen backbones and targets
Remote: https://github.com/DannyPhongcoderso1/gnnexplainer-groundtruth-benchmark.git
```

Important tags:

| Tag | Meaning |
|---|---|
| `phase4c-gnnexplainer-v2` | Frozen Phase 4C GNNExplainer evaluation with Fid+ and Fid- |
| `phase4-evaluation-v1` | Frozen Phase 4 GNNExplainer, Random baseline, stability, and evaluation protocol |
| `ba-house-2x2-v1` | Frozen BA-House 2x2 explainer evaluation |
| `shapeggen-dataset-v1` | Frozen official GraphXAI ShapeGGen dataset |

## 3. Papers To Read

The paper files are stored under `D:\CS\KLTN\Paper`.

| Topic | File | Main use |
|---|---|---|
| GNN backbone | `D:\CS\KLTN\Paper\GNN\GCN.pdf` | Theoretical basis of Graph Convolutional Networks |
| GNN backbone | `D:\CS\KLTN\Paper\GNN\GraphSAGE.pdf` | GraphSAGE aggregation and inductive learning |
| Elliptic | `D:\CS\KLTN\Paper\GNN\Elliptic_GCN.pdf` | Applying GCN to the Elliptic Bitcoin dataset |
| Elliptic baseline | `D:\CS\KLTN\Paper\GNN\GCN_GraphSAGE_on_Elliptic_dataset.pdf` | Best hyperparameters and paper results for GCN/GraphSAGE on Elliptic |
| GNN XAI | `D:\CS\KLTN\Paper\XAI\GNNExplainer.pdf` | GNNExplainer and mask optimization |
| GNN XAI | `D:\CS\KLTN\Paper\XAI\PGNexplainer.pdf` | PGExplainer, explainer training, and fast inference |
| XAI evaluation | `D:\CS\KLTN\Paper\XAI\Evaluating_explainability_for_gnn.pdf` | Ground-truth agreement and perturbation-based explanation evaluation |
| XAI robustness | `D:\CS\KLTN\Paper\XAI\GNN_explaination_are_fragile.pdf` | Stability, fragility, and caution when interpreting explanations |

Key ideas to reuse in the thesis:

- `GCN` and `GraphSAGE` learn graph representations through different message-passing and aggregation mechanisms.
- `GNNExplainer` optimizes a separate explanation mask for each target instance, so its runtime is per-instance optimization time.
- `PGExplainer` requires explainer training first, then performs fast target-level inference.
- `Edge AUROC`, `Edge AP`, `Fid+`, `Fid-`, sparsity, stability, and runtime measure different aspects of explanation quality.
- No explainer should be called universally best based on a single metric.

## 4. Elliptic Baseline

Repository:

```text
D:\CS\KLTN\elliptic-gnn-xai
```

Baseline source:

```text
D:\CS\KLTN\GCN_GraphSAGE_base\Blockchain-Anomaly-Detection
https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection
Source commit: 83ef2b9ed8b3233175bbc3f4231c593c365225b9
```

### 4.1 Elliptic Data

Required raw CSV files:

```text
datasets/elliptic/elliptic_txs_features.csv
datasets/elliptic/elliptic_txs_edgelist.csv
datasets/elliptic/elliptic_txs_classes.csv
```

Preprocessed artifact:

```text
datasets/elliptic/elliptic_data.pt
SHA256: f807b83a7c739b16f69b5676226ce441b64d5fe8de95506a7727f86e8451291b
```

Preprocessed data statistics:

| Field | Value |
|---|---:|
| Nodes | 203769 |
| Edges | 468710 |
| Model input features | 165 |
| Train mask nodes | 26381 |
| Validation mask nodes | 8999 |
| Test mask nodes | 11184 |

Label mapping:

| Raw label | Internal label | Meaning |
|---|---:|---|
| `unknown` | 2 | Kept in the graph but excluded from supervised loss and evaluation |
| `"1"` | 1 | Positive class used for ROC-AUC and AP scoring |
| `"2"` | 0 | Negative class |

Temporal split:

| Split | Rule |
|---|---|
| Train | `time_step < 30` and known label |
| Validation | `30 <= time_step < 40` and known label |
| Test | `time_step >= 40` and known label |

### 4.2 Paper Baseline Hyperparameters

For the Elliptic baseline, Optuna is not needed because the goal is direct reproduction using the best parameters reported in the paper.

| Hyperparameter | GCN | GraphSAGE |
|---|---:|---:|
| Hidden dimension | 211 | 140 |
| Embedding dimension | 90 | 103 |
| Number of layers | 2 | 2 |
| Learning rate | 0.0008475 | 0.0005302 |
| Epochs | 497 | 397 |
| Dropout | 0.2361 | 0.1135 |
| Weight decay | 0.0005 | 0.0005 |
| Aggregator | Not applicable | mean |

Frozen interpretation:

- `100 trials` in the baseline result files are not 100 independent training runs.
- They are 100 bootstrap rows, meaning repeated evaluation on subsamples from the same checkpoint.
- The epoch counts are fixed from the paper: 497 epochs for GCN and 397 epochs for GraphSAGE.
- Optuna would create a new search experiment, not a direct baseline reproduction.

### 4.3 Elliptic Baseline Results

| Model | Paper AUC | Reproduced AUC | Paper AP | Reproduced AP |
|---|---:|---:|---:|---:|
| GCN | 0.8728 | 0.8652 +- 0.0072 | 0.5993 | 0.5697 +- 0.0213 |
| GraphSAGE | 0.8593 | 0.8743 +- 0.0075 | 0.6551 | 0.6607 +- 0.0163 |

Important artifacts:

| Artifact | Path | SHA256 |
|---|---|---|
| GCN checkpoint | `artifacts/checkpoints/gcn_baseline.pt` | `55004ffa1b712ae374fbf630d55f31ac69602d696dbeef8cb150d3f788a9cc07` |
| GraphSAGE checkpoint | `artifacts/checkpoints/graphsage_baseline.pt` | `f25d5cbeb10f9a27dbba1adcb2fdb21b406219956c107f8d51e8ecf5188866bf` |
| Elliptic data artifact | `datasets/elliptic/elliptic_data.pt` | `f807b83a7c739b16f69b5676226ce441b64d5fe8de95506a7727f86e8451291b` |

Reproduction commands:

```powershell
cd D:\CS\KLTN\elliptic-gnn-xai
python scripts\preprocess_elliptic.py --raw-dir datasets\elliptic --out datasets\elliptic\elliptic_data.pt
powershell -ExecutionPolicy Bypass -File scripts\run_baseline_paper_params.ps1
```

Test commands:

```powershell
cd D:\CS\KLTN\elliptic-gnn-xai
python -m pytest
python scripts\test.py --config config\testing.yaml --model GCN
python scripts\test.py --config config\testing.yaml --model SAGE
```

## 5. Experiment Environment

The local machine has a working GPU.

| Component | Value |
|---|---|
| Python | 3.11.9 |
| Torch | 2.11.0+cu128 |
| Torch Geometric | 2.8.0.post1 |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Operating system | Windows |

## 6. BA-House Dataset

Dataset artifact:

```text
artifacts/data/ba_house.pt
SHA256: 4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94
Metadata: artifacts/data/ba_house_metadata.json
```

Dataset statistics:

| Field | Value |
|---|---:|
| Nodes | 500 |
| Edges | 3398 |
| Node features | 10 |
| Label 0 nodes | 300 |
| Label 1 nodes | 80 |
| Label 2 nodes | 80 |
| Label 3 nodes | 40 |
| Ground-truth motif nodes | 200 |
| Ground-truth motif edges | 480 |
| Train nodes | 300 |
| Validation nodes | 100 |
| Test nodes | 100 |
| Seed | 42 |

Data split:

| Split | Count | Ratio |
|---|---:|---:|
| Train | 300 | 60% |
| Validation | 100 | 20% |
| Test | 100 | 20% |

Meaning:

- Label 0 represents background Barabasi-Albert nodes.
- Labels 1, 2, and 3 represent roles in the house motif.
- BA-House provides edge and node ground truth, so it is suitable for explanation evaluation.

## 7. BA-House Backbones

### 7.1 GCN

Frozen checkpoint:

```text
artifacts/checkpoints/ba_house_gcn_best.pt
SHA256: 4C81814172650E20B6EDEE5FC97BC66CAE46AC1E8BA76BAEC2226C59A7D1D507
```

Configuration:

| Field | Value |
|---|---|
| Architecture | PyG `GCNConv`, `cached=False` |
| Hidden channels | 32 |
| Layers | 3 |
| Dropout | 0.2 |
| Learning rate | 0.01 |
| Weight decay | 0.0005 |
| Maximum epochs | 300 |
| Patience | 60 |
| Class weighting | Enabled |
| Seed | 42 |
| Best epoch | 10 |
| Best validation accuracy | 0.65 |

Results:

| Split | Accuracy | Correct / Total |
|---|---:|---:|
| Train | 0.606667 | 182 / 300 |
| Validation | 0.650000 | 65 / 100 |
| Test | 0.620000 | 62 / 100 |
| Motif test | 0.400000 | 16 / 40 |

### 7.2 GraphSAGE

Frozen checkpoint:

```text
artifacts/checkpoints/ba_house_graphsage_best.pt
SHA256: 09992c5ac4d2cd80c10792c12be5f945b3a20d749d8433008eba9cf3694135eb
```

Configuration:

| Field | Value |
|---|---|
| Architecture | PyG `SAGEConv` |
| Aggregator | sum |
| Hidden channels | 32 |
| Layers | 3 |
| Dropout | 0.2 |
| Learning rate | 0.01 |
| Weight decay | 0.0005 |
| Maximum epochs | 300 |
| Patience | 60 |
| Class weighting | Enabled |
| Seed | 42 |
| Best epoch | 87 |
| Best validation accuracy | 0.95 |

Results:

| Split | Accuracy | Correct / Total |
|---|---:|---:|
| Test | 0.960000 | 96 / 100 |
| Motif test | 0.900000 | 36 / 40 |
| Majority baseline | 0.600000 | Not applicable |
| Random baseline | 0.250000 | Not applicable |

Important interpretation warning:

- GCN and GraphSAGE have very different classifier performance on BA-House.
- Therefore, cross-backbone explanation differences should be described as backbone-context observations.
- They should not be stated as pure causal effects of changing only the backbone.

## 8. Frozen XAI Evaluation Protocol

| Component | Frozen rule |
|---|---|
| Candidate-edge scope | Computation subgraph around the target node based on the model hop count |
| Edge AUROC and Edge AP | Computed from continuous scores over the full candidate-edge set |
| Edge-selection budget | Fixed candidate-edge ratio |
| Selected edge ratio | 20% |
| Rounding rule | `ceil(candidate_edge_count * 0.2)`, minimum 1 edge |
| Remove and retain perturbations | Use the same selected edge set |
| Legacy fidelity | `remove_delta = original_probability - remove_probability` |
| Fid+ | 1 if removing selected edges changes the original prediction |
| Fid- | 1 if retaining only selected edges changes the original prediction |
| 1-Fid- | `1 - Fid-`, higher means selected edges preserve the prediction better |
| Sparsity | `1 - selected_edge_count / candidate_edge_count` |
| Stability | Mean pairwise Jaccard similarity across selected edge sets from multiple seeds |

Why fixed `top_k=25` was replaced:

- Candidate-edge counts vary strongly across nodes.
- A fixed top-k budget is unfair between small and large candidate sets.
- Ratios 5%, 10%, and 20% were tested on validation targets only.
- The final frozen protocol uses a 20% candidate-edge ratio.

Node 465 sanity calibration:

| Explanation | Edge AUROC | Edge AP | Remove delta | Retain probability |
|---|---:|---:|---:|---:|
| GNNExplainer with top-k 25 | 0.933057 | 0.218982 | -0.011565 | 0.295433 |
| Ground-truth oracle | 1.000000 | 1.000000 | 0.020871 | 0.274899 |
| Random top-k mean | 0.533077 | 0.043400 | -0.000573 | 0.275232 |

## 9. BA-House XAI Results

Frozen 2x2 target set:

```text
configs/targets/ba_house_2x2_targets.csv
SHA256: 2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f
Target count: 12
Node ids: 306, 315, 316, 355, 365, 375, 376, 411, 445, 465, 486, 491
```

Validation targets used before final target freeze:

```text
311, 380, 385, 450
```

### 9.1 GCN With GNNExplainer

| Metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Edge AUROC | 0.870617 | 0.077797 | 0.892918 | 0.678571 | 0.935044 |
| Edge AP | 0.527731 | 0.236662 | 0.584178 | 0.106239 | 0.781677 |
| Fid+ | 0.666667 | 0.492366 | 1.000000 | 0.000000 | 1.000000 |
| Fid- | 0.250000 | 0.452267 | 0.000000 | 0.000000 | 1.000000 |
| 1-Fid- | 0.750000 | 0.452267 | 1.000000 | 0.000000 | 1.000000 |
| Legacy fidelity | -0.000357 | 0.006904 | 0.002986 | -0.015010 | 0.007155 |
| Sparsity | 0.793694 | 0.008901 | 0.797033 | 0.769231 | 0.800000 |
| Explanation-generation runtime | 0.129505 | 0.177384 | 0.075328 | 0.068638 | 0.692148 |

Batch result:

- 12/12 targets succeeded.
- 0 targets failed.
- 12/12 raw explanations were reused.
- Primary seed: 42.
- 12/12 edge-mask hashes matched.

### 9.2 GCN With Random Baseline

Random uses the same target nodes, checkpoint, candidate-edge scope, 20% budget, and evaluator.

| Metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Edge AUROC | 0.496727 | 0.102063 | 0.500951 | 0.196970 | 0.754630 |
| Edge AP | 0.277178 | 0.233067 | 0.216445 | 0.012323 | 0.726835 |
| Fid+ | 0.333333 | 0.473381 | 0.000000 | 0.000000 | 1.000000 |
| Fid- | 0.625000 | 0.486153 | 1.000000 | 0.000000 | 1.000000 |
| 1-Fid- | 0.375000 | 0.486153 | 0.000000 | 0.000000 | 1.000000 |
| Legacy fidelity | -0.000994 | 0.006809 | -0.000784 | -0.027685 | 0.020590 |
| Sparsity | 0.793694 | 0.008557 | 0.797033 | 0.769231 | 0.800000 |
| Random selection and evaluation runtime | 0.021375 | 0.022475 | 0.018677 | 0.017017 | 0.262515 |

Random aggregation rule:

- The raw table has 120 rows: 12 nodes times 10 seeds.
- Main quality comparison first averages metrics within each node.
- The final summary is then computed across nodes.
- Raw seed-level standard deviation and explainer node-level standard deviation must not be treated as identical sampling units.

### 9.3 GCN With PGExplainer

| Metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Edge AUROC | 0.747969 | 0.188227 | 0.843335 | 0.333333 | 0.977273 |
| Edge AP | 0.366563 | 0.253247 | 0.434493 | 0.045876 | 0.675609 |
| Fid+ | 0.416667 | 0.514929 | 0.000000 | 0.000000 | 1.000000 |
| Fid- | 0.583333 | 0.514929 | 1.000000 | 0.000000 | 1.000000 |
| 1-Fid- | 0.416667 | 0.514929 | 0.000000 | 0.000000 | 1.000000 |
| Legacy fidelity | -0.001014 | 0.004187 | -0.000095 | -0.009157 | 0.003353 |
| Sparsity | 0.793694 | 0.008901 | 0.797033 | 0.769231 | 0.800000 |
| Inference runtime | 0.005550 | 0.001033 | 0.005271 | 0.004727 | 0.008589 |

PGExplainer notes:

- The current PyG version does not support `explanation_type="model"` for PGExplainer.
- The supported path is `explanation_type="phenomenon"`.
- Implementation provenance was corrected so the API semantics are not misreported.

### 9.4 Three-Way Comparison In The GCN Context

| Method | Edge AUROC | Edge AP | Fid+ | Fid- | 1-Fid- | Sparsity | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.496727 | 0.277178 | 0.333333 | 0.625000 | 0.375000 | 0.793694 | 0.126536 |
| GNNExplainer | 0.870617 | 0.527731 | 0.666667 | 0.250000 | 0.750000 | 0.793694 | 0.441272 |
| PGExplainer | 0.747969 | 0.366563 | 0.416667 | 0.583333 | 0.416667 | 0.793694 | 0.951911 |

Interpretation:

- In the GCN context, GNNExplainer has stronger ground-truth edge agreement than PGExplainer.
- GNNExplainer is also stronger on Fid+ and 1-Fid-.
- PGExplainer is much more stable under Jaccard stability.
- Random is a sanity baseline, not a learned explainer.

### 9.5 GraphSAGE With GNNExplainer

| Metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Edge AUROC | 0.895612 | 0.083495 | 0.916028 | 0.678571 | 0.961490 |
| Edge AP | 0.594340 | 0.229296 | 0.642899 | 0.205847 | 0.867933 |
| Fid+ | 0.916667 | 0.288675 | 1.000000 | 0.000000 | 1.000000 |
| Fid- | 0.500000 | 0.522233 | 0.500000 | 0.000000 | 1.000000 |
| 1-Fid- | 0.500000 | 0.522233 | 0.500000 | 0.000000 | 1.000000 |
| Legacy fidelity | 0.412075 | 0.151457 | 0.457578 | 0.037357 | 0.573621 |
| Sparsity | 0.793694 | 0.008901 | 0.797033 | 0.769231 | 0.800000 |
| Explanation-generation runtime | 0.129616 | 0.188752 | 0.077888 | 0.051090 | 0.726345 |

### 9.6 GraphSAGE With PGExplainer

PGExplainer used the natural pool with 106 training nodes and 35 validation nodes.

| Metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Edge AUROC | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| Edge AP | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| Fid+ | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| Fid- | 0.500000 | 0.522233 | 0.500000 | 0.000000 | 1.000000 |
| 1-Fid- | 0.500000 | 0.522233 | 0.500000 | 0.000000 | 1.000000 |
| Legacy fidelity | 0.425826 | 0.126268 | 0.437633 | 0.101164 | 0.573621 |
| Sparsity | 0.793694 | 0.008901 | 0.797033 | 0.769231 | 0.800000 |
| Inference runtime | 0.014336 | 0.031808 | 0.005216 | 0.004820 | 0.115336 |

Because this result was extremely high, additional audit and sensitivity checks were performed:

- Check for target leakage.
- Check candidate-edge mapping.
- Check whether ground truth was accidentally used.
- Check whether the PGExplainer training pool created a confound.

### 9.7 BA-House Stability

| Backbone | Explainer | Mean Jaccard | Std | Median | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| GCN | GNNExplainer | 0.441272 | 0.279181 | 0.368264 | 0.159524 | 0.977273 |
| GCN | PGExplainer | 0.951911 | 0.047503 | 0.964542 | 0.866667 | 1.000000 |
| GraphSAGE | GNNExplainer | 0.479613 | 0.258427 | 0.421404 | 0.200000 | 0.977273 |
| GraphSAGE | PGExplainer natural pool | 0.805056 | 0.118526 | 0.830964 | 0.644444 | 1.000000 |

## 10. BA-House 2x2 Comparison

This is the frozen primary BA-House 2x2 table.

| Backbone | Explainer | Edge AUROC | Edge AP | Fid+ | Fid- | 1-Fid- | Stability |
|---|---|---:|---:|---:|---:|---:|---:|
| GCN | GNNExplainer | 0.870617 | 0.527731 | 0.666667 | 0.250000 | 0.750000 | 0.441272 |
| GCN | PGExplainer | 0.747969 | 0.366563 | 0.416667 | 0.583333 | 0.416667 | 0.951911 |
| GraphSAGE | GNNExplainer | 0.895612 | 0.594340 | 0.916667 | 0.500000 | 0.500000 | 0.479613 |
| GraphSAGE | PGExplainer natural pool | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 0.500000 | 0.805056 |

Recommended conclusion wording:

- In the GCN context, GNNExplainer has stronger ground-truth agreement than PGExplainer.
- In the GraphSAGE context, PGExplainer has stronger ground-truth agreement than GNNExplainer.
- PGExplainer is more stable than GNNExplainer in both backbone contexts.
- Cross-backbone differences should not be interpreted as pure causal backbone effects.

## 11. BA-House PGExplainer Common-Pool Sensitivity

Why this was needed:

- GraphSAGE PGExplainer originally used a larger natural training pool than GCN PGExplainer.
- A controlled check was needed to see whether the near-perfect GraphSAGE PGExplainer result was only caused by a larger training pool.

Common pool:

| Pool | Count |
|---|---:|
| Common PG train | 48 |
| Common PG validation | 16 |

Controlled common-pool result:

| Metric | PG GCN common | PG GraphSAGE common | GraphSAGE - GCN |
|---|---:|---:|---:|
| Edge AUROC | 0.747969 | 0.999869 | 0.251900 |
| Edge AP | 0.366563 | 0.993590 | 0.627027 |
| Fid+ | 0.416667 | 1.000000 | 0.583333 |
| Fid- | 0.583333 | 0.500000 | -0.083333 |
| 1-Fid- | 0.416667 | 0.500000 | 0.083333 |
| Stability | 0.951911 | 0.789956 | -0.161955 |
| Sparsity | 0.793694 | 0.793694 | 0.000000 |
| Inference runtime | 0.005550 | 0.003302 | -0.002248 |

GraphSAGE PGExplainer natural pool versus common pool:

| Metric | Natural 106/35 | Common 48/16 | Common - natural |
|---|---:|---:|---:|
| Edge AUROC | 1.000000 | 0.999869 | -0.000131 |
| Edge AP | 1.000000 | 0.993590 | -0.006410 |
| Fid+ | 1.000000 | 1.000000 | 0.000000 |
| Fid- | 0.500000 | 0.500000 | 0.000000 |
| 1-Fid- | 0.500000 | 0.500000 | 0.000000 |
| Stability | 0.805056 | 0.789956 | -0.015100 |
| Sparsity | 0.793694 | 0.793694 | 0.000000 |
| Inference runtime | 0.014336 | 0.003302 | -0.011034 |

Conclusion:

- The GraphSAGE PGExplainer result remained very strong under the common-pool condition.
- This reduces the concern that the high result was only due to GraphSAGE having a larger PGExplainer training pool.
- However, this remains a post-hoc sensitivity analysis and does not replace the original natural-pool result.
- This lesson was applied prospectively to ShapeGGen: common PG pools must be frozen before XAI.

## 12. ShapeGGen Dataset

ShapeGGen was replaced with the official GraphXAI implementation. The local reimplementation is no longer the source of truth.

Dataset artifact:

```text
artifacts/data/shapeggen.pt
SHA256: 9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e
Protocol: THESIS_SHAPEGGEN_OFFICIAL_V1
Upstream GraphXAI commit: a11e65ffbc4df737f35522a8accf2283a8aeaa37
```

Dataset statistics:

| Field | Value |
|---|---:|
| Nodes | 1298 |
| Directed edge entries | 4518 |
| Undirected edges | 2259 |
| Node features | 10 |
| Class 0 nodes | 502 |
| Class 1 nodes | 796 |
| Train nodes | 778 |
| Validation nodes | 259 |
| Test nodes | 261 |

Ground-truth and candidate statistics:

| Field | Value |
|---|---:|
| Targets | 1298 |
| Candidate edge entries | 156266 |
| Positive edge entries | 28316 |
| Positive node entries | 10420 |
| Eligible targets | 1298 |
| Minimum positive edges per target | 12 |
| Median positive edges per target | 26 |
| Maximum positive edges per target | 28 |
| Minimum candidate edges per target | 22 |
| Median candidate edges per target | 118 |
| Maximum candidate edges per target | 296 |

Label semantics:

- Class 0 means one nearby unique motif.
- Class 1 means two nearby unique motifs.
- ShapeGGen labels must not be described as licit or illicit transaction classes.

Edge policy:

- Use the term `directed edge entries` when referring to PyG `edge_index`.
- 4518 directed edge entries correspond to 2259 undirected edges with reverse entries.
- There are no self-loops.
- There are no duplicate ordered edge entries.
- Reverse-edge coverage is complete.

## 13. Task22 ShapeGGen Backbone And Target Freeze

Task22 is complete. This is the current thesis progress point.

Completed instruction file:

```text
D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\07_TASK22_SHAPEGGEN_BACKBONES.md
```

Task22 goals:

- Reuse the exact ShapeGGen dataset frozen in Task21.
- Train and freeze GCN and GraphSAGE on the same artifact and same split.
- Select checkpoints using validation only, then report final test performance once.
- Freeze a common 2x2 target set before any explainer is run.
- Freeze common PGExplainer train and validation pools using the intersection of nodes eligible under both backbones.
- Do not run GNNExplainer, Random, PGExplainer, or stability during Task22.

### 13.1 ShapeGGen GCN

Frozen checkpoint:

```text
artifacts/checkpoints/shapeggen_gcn_best.pt
SHA256: 15bfa8960fe4e0f0712c3a794db93ed4e90aad6fce3555a16db23a8a13b91e36
```

Results:

| Split | Accuracy | Correct / Total | Motif correct / Motif total |
|---|---:|---:|---:|
| Train | 0.692802 | 539 / 778 | 345 / 477 |
| Validation | 0.752896 | 195 / 259 | 125 / 159 |
| Test | 0.681992 | 178 / 261 | 119 / 160 |

Checkpoint reload parity:

```text
passed: true
max_abs_diff: 0.0
```

### 13.2 ShapeGGen GraphSAGE

Frozen checkpoint:

```text
artifacts/checkpoints/shapeggen_graphsage_best.pt
SHA256: 498a74875360a82749b78cb40a69b3e4ec04ec634b85e4713e85d96f34c666a0
```

Results:

| Split | Accuracy | Correct / Total | Motif correct / Motif total |
|---|---:|---:|---:|
| Train | 0.998715 | 777 / 778 | 476 / 477 |
| Validation | 0.884170 | 229 / 259 | 147 / 159 |
| Test | 0.869732 | 227 / 261 | 141 / 160 |

Checkpoint reload parity:

```text
passed: true
max_abs_diff: 0.0
```

### 13.3 ShapeGGen Targets And PG Pools

Final 2x2 target set:

```text
configs/targets/shapeggen_2x2_targets.csv
Target count: 167
SHA256: bd128dec7810421d592debe8177f3cb46d6c7d9b01fd3f562b8b13971517b41d
```

Backbone-specific eligible pools:

| Pool | Count | SHA256 |
|---|---:|---|
| GCN train eligible | 539 | `17a69bd999853a8487a1f85af7bec21409e171dad8f021d99918f98192279b4f` |
| GraphSAGE train eligible | 777 | `d5fc96cdd8e9c6b4f4dab6e707ecfbf2e1c3b36fcdf525c004b446a8d74ddd91` |
| GCN validation eligible | 195 | `c3ec03b26f3636266acc0d0872a53d605b28670125249a098f18fd843b2de47f` |
| GraphSAGE validation eligible | 229 | `4189685c8facd2c6ff24fd12fc85715c47c97691c5760c9b7877eb13d7f448ba` |

Frozen common PG pools:

| Pool | Count | SHA256 |
|---|---:|---|
| Common PG train | 539 | `c05a5be38a64b629c8bb837b1fb3408d8a008726713e70f0318aee3499746b70` |
| Common PG validation | 182 | `b988e40c2af234724b07802ec31d0831ef039a3ef4de3bfee1b80051b49cdc5d` |

Disjointness checks:

| Check | Result |
|---|---|
| Common train and common validation are disjoint | True |
| Common train and final 2x2 are disjoint | True |
| Common validation and final 2x2 are disjoint | True |

Task22 validation:

```text
pytest: 111 passed
ruff check: passed
git diff --check: passed
Commit: bed705d Freeze ShapeGGen backbones and targets
```

## 14. Progress By Task

| Stage | Content | Status |
|---|---|---|
| Elliptic baseline | Reproduce GCN and GraphSAGE baseline from source repository | Complete |
| Task01 | Create BA-House dataset | Complete |
| Task02 | Train GCN on BA-House | Complete |
| Task03 | Run single-node GNNExplainer | Complete |
| Phase 4B | Freeze metric protocol | Complete |
| Phase 4C | Run GNNExplainer batch on GCN | Complete |
| Phase 4D | Run stability and Random baseline | Complete |
| Task14 | Freeze PGExplainer protocol and smoke test | Complete |
| Task15 | Train PGExplainer on GCN BA-House | Complete |
| Task16 | Evaluate PGExplainer on GCN BA-House | Complete |
| Task16B | Stability and runtime for PGExplainer on GCN | Complete |
| Task17 | Three-way comparison on GCN | Complete |
| Task18 | Train GraphSAGE on BA-House | Complete |
| Task19 | Run explainers on GraphSAGE BA-House | Complete |
| Task19A | Audit near-perfect GraphSAGE PGExplainer result | Complete |
| Task19B | Common-pool sensitivity analysis | Complete |
| Task20 | Freeze BA-House 2x2 comparison | Complete |
| Task21 | Freeze ShapeGGen dataset and ground truth | Complete |
| Task21C | Replace local ShapeGGen with official GraphXAI implementation | Complete |
| Task22 | Train ShapeGGen GCN/GraphSAGE and freeze targets before XAI | Complete |

## 15. Next Work

Next instruction file:

```text
D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\08_TASK23_SHAPEGGEN_GNNEXPLAINER_RANDOM.md
```

Upcoming tasks:

| Task | File | Goal |
|---|---|---|
| Task23 | `08_TASK23_SHAPEGGEN_GNNEXPLAINER_RANDOM.md` | Run GNNExplainer and Random on ShapeGGen for both GCN and GraphSAGE |
| Task24 | `09_TASK24_SHAPEGGEN_PGEXPLAINER.md` | Train and evaluate PGExplainer on ShapeGGen using common PG pools |
| Task25 | `10_TASK25_SHAPEGGEN_2X2_CROSS_SYNTHETIC.md` | Summarize ShapeGGen 2x2 and compare with BA-House |
| Task26 | `11_TASK26_ELLIPTIC_HANDOFF_PARITY_PROTOCOL.md` | Transfer the protocol to Elliptic |
| Task27 | `12_TASK27_ELLIPTIC_GNNEXPLAINER.md` | Run GNNExplainer on Elliptic |
| Task28 | `13_TASK28_ELLIPTIC_PGEXPLAINER.md` | Run PGExplainer on Elliptic |
| Task29 | `14_TASK29_ELLIPTIC_CASE_STUDY.md` | Build TP, FP, FN, TN case studies |
| Task30 | `15_TASK30_ELLIPTIC_QUANTITATIVE_COMPARISON.md` | Summarize Elliptic quantitative comparison and lock experiments |
| Task31 | `16_TASK31_FINAL_ANALYSIS_WRITING.md` | Write final analysis |
| Task32 | `17_TASK32_SUPERVISOR_REVISION.md` | Revise based on supervisor feedback |
| Task33 | `18_TASK33_FINAL_REPRODUCIBILITY_PACKAGE.md` | Package reproducibility materials and final QA |

Planned deadlines:

| Date | Goal |
|---|---|
| 2026-10-18 | Lock all core experiments |
| 2026-10-19 to 2026-10-25 | Prioritize writing and revision |
| 2026-10-26 to 2026-10-28 | Final QA and reproducibility only |
| 2026-10-29 | Submission deadline |

## 16. Lessons Learned

1. Freeze the dataset, checkpoint, target list, metric protocol, and budget before final evaluation.
2. Do not reselect targets after seeing explainer results.
3. Do not tune ratios or hyperparameters on final test metrics.
4. For the Elliptic baseline, paper best parameters are enough; Optuna is not needed.
5. `100 trials` in the baseline are bootstrap rows, not 100 training runs.
6. `Edge AUROC` and `Edge AP` must use continuous scores over the full candidate-edge set.
7. Fidelity must clearly separate remove-side and retain-side behavior.
8. Legacy `fidelity` equals `remove_delta`, but Fid+ and Fid- are easier to interpret in the thesis.
9. Fixed `top_k=25` is not suitable for final evaluation when candidate-edge counts vary strongly.
10. The 20% candidate-edge budget is frozen and must be used consistently.
11. Runtime semantics must be explicit and should not be collapsed into a single winner column.
12. PGExplainer has separate upfront training time and per-target inference time.
13. GNNExplainer optimizes each target separately, so runtime is per-target generation time.
14. Random is only a sanity baseline.
15. Stability uses the same Jaccard metric but different stochastic sources for each method.
16. PGExplainer should use common train and validation pools when comparing across backbones.
17. BA-House and ShapeGGen have ground-truth edge explanations, so Edge AUROC/AP are valid.
18. Elliptic does not have ground-truth explanatory edges, so explanation Edge AUROC/AP must not be reported there.
19. Cross-backbone results should be described with phrases such as `backbone context` and `under the frozen protocol`.
20. Avoid phrases such as `pure backbone effect`, `only backbone changed`, or `universally best explainer`.

## 17. Thesis Document And Proposal Work

Document-related work already performed in the thread:

- Reformatted Section 3 to match Section 2 in the proposal document.
- Updated the proposal using the new XAI prompt.
- Updated the proposal toward explainer-comparison metrics.
- Replaced `faithfulness` with `fidelity` as requested.
- Checked spelling.
- Removed semicolons from the proposal.
- Fixed a broken table.
- Restored the cover-page frame.
- Performed parallel writing work on `D:\CS\KLTN\Luận.docx`.

Writing suggestions:

- The data chapter should separate Elliptic, BA-House, and ShapeGGen.
- The backbone chapter should separate GCN and GraphSAGE.
- The explanation-method chapter should separate GNNExplainer, PGExplainer, and Random baseline.
- The experiment chapter should separate synthetic datasets with ground truth from Elliptic without ground-truth explanations.
- The discussion chapter should emphasize multidimensional metrics and interpretation limits.

## 18. Common Commands

Check the XAI repository:

```powershell
cd D:\CS\KLTN\gnnexplainer-groundtruth-benchmark
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
git diff --check
git status --short --branch
```

Reproduce ShapeGGen Task22 if needed:

```powershell
cd D:\CS\KLTN\gnnexplainer-groundtruth-benchmark
.\.venv\Scripts\python.exe scripts\run_shapeggen_backbones_and_targets.py
```

Reproduce the Elliptic baseline:

```powershell
cd D:\CS\KLTN\elliptic-gnn-xai
python scripts\preprocess_elliptic.py --raw-dir datasets\elliptic --out datasets\elliptic\elliptic_data.pt
powershell -ExecutionPolicy Bypass -File scripts\run_baseline_paper_params.ps1
python -m pytest
```

## 19. Do Not Do These When Starting Task23

- Do not reselect ShapeGGen targets.
- Do not change `selected_edge_ratio=0.2`.
- Do not tune on final test results.
- Do not run PGExplainer in Task23.
- Do not start Task24 before Task23 is reviewed.
- Do not silently drop failed targets.
- Do not change the frozen GCN or GraphSAGE checkpoints.
- Do not change the frozen ShapeGGen dataset.
- Do not change the computation of `Edge AUROC` or `Edge AP`.
- Do not collapse GNNExplainer and Random runtime into a single runtime meaning.

## 20. Important Source Files

| Content | File |
|---|---|
| BA-House Phase 4 report | `docs/phase4_evaluation_report.md` |
| GCN three-way comparison | `docs/gcn_explainer_comparison_report.md` |
| GraphSAGE BA-House explainer report | `docs/graphsage_explainers_ba_house_report.md` |
| BA-House 2x2 report | `docs/ba_house_2x2_evaluation_report.md` |
| BA-House 2x2 freeze JSON | `docs/ba_house_2x2_freeze.json` |
| ShapeGGen protocol | `docs/shapeggen_dataset_protocol.md` |
| ShapeGGen ground-truth verification | `docs/shapeggen_gt_verification_report.md` |
| ShapeGGen backbone freeze report | `docs/shapeggen_backbones_and_targets_freeze.md` |
| ShapeGGen target freeze JSON | `docs/shapeggen_targets_freeze.json` |
| Elliptic baseline reproduction | `D:\CS\KLTN\elliptic-gnn-xai\docs\baseline_reproduction.md` |
| Elliptic baseline comparison | `D:\CS\KLTN\elliptic-gnn-xai\results\comparison.md` |

## 21. Very Short Progress Summary

The Elliptic GCN and GraphSAGE baselines have been reproduced using code migrated from the source repository and paper best hyperparameters. The reproduced results are close to the paper: GCN AUC 0.8652, GCN AP 0.5697, GraphSAGE AUC 0.8743, and GraphSAGE AP 0.6607.

In the XAI repository, BA-House is complete: dataset, GCN, GraphSAGE, GNNExplainer, PGExplainer, Random baseline, stability, and the frozen 2x2 comparison over 12 shared targets. In the GCN context, GNNExplainer has stronger Edge AUROC/AP than PGExplainer. In the GraphSAGE context, PGExplainer reaches near-perfect Edge AUROC/AP. PGExplainer is more stable than GNNExplainer in both backbone contexts.

Task22 for ShapeGGen is complete: the official GraphXAI ShapeGGen dataset is frozen, GCN and GraphSAGE are trained and frozen, 167 common 2x2 targets are frozen, and common PG train/validation pools are frozen before any ShapeGGen XAI run. The next task is Task23, which runs GNNExplainer and Random on ShapeGGen for both backbones using the frozen target set and frozen 20% budget.
