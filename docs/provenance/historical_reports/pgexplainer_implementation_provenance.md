# PGExplainer Implementation Provenance

## Purpose

This document records the source hierarchy, installed PyG API audit, and implementation-provenance decisions for the upcoming PGExplainer protocol. It is documentation only. No PGExplainer training, evaluation, environment change, checkpoint change, target-list change, or Phase 4 protocol change is performed in this step.

Phase 4 remains frozen:

- GCN checkpoint: unchanged.
- Frozen explainer-test targets: unchanged.
- Evaluation metrics: unchanged.
- Explanation budget: `selected_edge_ratio = 0.2`.
- Candidate-edge scope: unchanged.
- No PGExplainer experiment has been run for this provenance step.

## Source Hierarchy

When a PGExplainer implementation or configuration question arises, use this source order:

| Priority | Source | What It Controls |
|---:|---|---|
| 1 | NeurIPS 2020 main paper, "Parameterized Explainer for Graph Neural Network" | Method concept, post-hoc objective, explainer formulation |
| 2 | NeurIPS 2020 Supplementary Material | Reported training hyperparameters |
| 3 | Official authors' repository, `flyingdoog/PGExplainer` | Original implementation provenance |
| 4 | Installed PyG `2.8.0.post1` source in this environment | Actual API, supported explanation settings, implementation behavior |
| 5 | Frozen Phase 4 thesis evaluation protocol | Fairness decisions specific to this thesis: metrics, targets, candidate-edge scope, and 20% budget |

Primary URLs:

- NeurIPS abstract/proceedings: `https://papers.nips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html`
- Main paper PDF: `https://papers.nips.cc/paper_files/paper/2020/file/e37b08dd3015330dcbb5d6663667b8b8-Paper.pdf`
- Supplementary PDF: `https://papers.nips.cc/paper_files/paper/2020/file/e37b08dd3015330dcbb5d6663667b8b8-Supplemental.pdf`
- Official author repository: `https://github.com/flyingdoog/PGExplainer`
- PyG PGExplainer docs for version 2.8.0: `https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.explain.algorithm.PGExplainer.html`

## Paper-Level Method Basis

PGExplainer is a parameterized post-hoc explainer. The prediction model is trained first and remains frozen while a separate explanation network is optimized. For node classification, PGExplainer uses node embeddings and local graph structure to produce edge logits or edge distributions.

Conceptual flow:

```text
Frozen GNN
  -> node embeddings
  -> PGExplainer MLP
  -> edge logits / edge distribution
  -> Concrete-style sampling during training
  -> masked graph
  -> frozen GNN prediction
  -> loss preserving the already-trained GNN's prediction behavior
     + size regularization
     + entropy regularization
```

This thesis uses PGExplainer for structural edge explanations. Feature explanations are not part of the primary PGExplainer protocol.

## Installation Statement

PGExplainer is already available in the frozen PyG environment:

```powershell
.\.venv\Scripts\python.exe -c "from torch_geometric.explain.algorithm import PGExplainer; print(PGExplainer)"
```

Audit result:

```text
PASS
```

Therefore:

- No additional PGExplainer package is installed.
- No PyG upgrade is performed.
- No TensorFlow PGExplainer environment is introduced into the main experiment.

Reason: the thesis uses the existing PyTorch/PyG stack to preserve framework consistency with GNNExplainer and the frozen Phase 4 evaluator.

## Installed PyG Source Audit

Installed environment:

```text
torch_geometric version: 2.8.0.post1
PGExplainer class: torch_geometric.explain.algorithm.pg_explainer.PGExplainer
Installed source path:
D:\CS\KLTN\gnnexplainer-groundtruth-benchmark\.venv\Lib\site-packages\torch_geometric\explain\algorithm\pg_explainer.py
```

Inspected signatures:

```text
PGExplainer(epochs: int, lr: float = 0.003, **kwargs)
PGExplainer.train(epoch, model, x, edge_index, *, target, index=None, **kwargs) -> float
PGExplainer.forward(model, x, edge_index, *, target, index=None, **kwargs) -> Explanation
```

Important correction: `epochs` is a required constructor argument in PyG PGExplainer. It is not a PyG constructor default. The thesis-selected value `epochs = 30` comes from the NeurIPS 2020 Supplementary Material, with PyG documentation examples also using 30.

Installed PyG implementation coefficients:

```python
{
    "edge_size": 0.05,
    "edge_ent": 1.0,
    "temp": [5.0, 2.0],
    "bias": 0.01,
}
```

`bias = 0.01` is an installed-PyG implementation-specific coefficient. It is not treated as one of the main paper-reported PGExplainer hyperparameters used to define the thesis protocol.

## Hyperparameter Provenance

| Parameter | Selected value | Source | Source category | Paper alignment | Deviation | Rationale |
|---|---:|---|---|---|---|---|
| `epochs` | `30` | NeurIPS 2020 Supplementary Material. PyG docs examples also use 30. | PAPER / SUPPLEMENT | Aligned with paper-reported baseline training setting | No deviation from paper setting. Not a PyG constructor default. | Use the paper-reported training length; pass explicitly because PyG requires `epochs`. |
| `lr` | `0.003` | NeurIPS 2020 Supplementary Material and installed PyG constructor default | PAPER / SUPPLEMENT + PYG | Aligned | No | Matches the paper-reported learning rate and PyG default argument. |
| `edge_size` | `0.05` | NeurIPS 2020 Supplementary Material and installed PyG `coeffs` | PAPER / SUPPLEMENT + PYG | Aligned | No | Matches the paper-reported size regularization and installed PyG coefficient. |
| `edge_ent` | `1.0` | NeurIPS 2020 Supplementary Material and installed PyG `coeffs` | PAPER / SUPPLEMENT + PYG | Aligned | No | Matches the paper-reported entropy regularization and installed PyG coefficient. |
| `temp` | `[5.0, 2.0]` | NeurIPS 2020 Supplementary Material and installed PyG `coeffs` | PAPER / SUPPLEMENT + PYG | Aligned | No | Matches the paper-reported temperature schedule endpoints and installed PyG coefficient. |
| `bias` | `0.01` | Installed PyG PGExplainer implementation | PYG_IMPLEMENTATION | Not a main paper-reported hyperparameter | Implementation-specific PyG coefficient | Retain installed PyG `2.8.0.post1` behavior without introducing an undocumented custom override. |

## PyG-Compatible Explainer Configuration

These settings are PyG explainer configuration/API choices, not paper-reported PGExplainer hyperparameters:

| Setting | Selected value | Classification | Source | Rationale |
|---|---|---|---|---|
| `explanation_type` | `"phenomenon"` | PYG_IMPLEMENTATION / API CONSTRAINT | Installed PyG `supports()` audit | Installed PyG PGExplainer supports `phenomenon` explanations and rejects `model` explanations. |
| `node_mask_type` | `None` | PYG_IMPLEMENTATION / SUPPORTED CONFIGURATION | Installed PyG `supports()` audit | Installed PyG PGExplainer does not support explaining input node feature masks. |
| `edge_mask_type` | `"object"` | PYG EXPLAINER CONFIGURATION | PyG explain API + structural thesis protocol | Thesis compares structural edge explanations across GNNExplainer, PGExplainer, and Random. |
| `task_level` | `"node"` | THESIS TASK + PYG SUPPORTED CONFIGURATION | Thesis benchmark + installed PyG support | BA-House benchmark explains node-level predictions. |

Installed PyG PGExplainer support audit:

```text
explanation_type="model"       -> unsupported
explanation_type="phenomenon"  -> supported
```

The terms `model` and `phenomenon` are PyG explain API terminology. This document does not claim that the original PGExplainer paper introduced those terms.

## Target Semantics For PyG PGExplainer

Because installed PyG PGExplainer requires `explanation_type="phenomenon"`, the explainer must receive an explicit target.

For this thesis:

```text
target_for_pgexplainer = frozen GCN predicted class
```

The default target must not be:

```text
ground-truth class label
```

Rationale: the original PGExplainer is a post-hoc explainer whose purpose is to explain or preserve the prediction behavior produced by an already-trained GNN. Using the frozen GCN predicted class as the PyG `phenomenon` target preserves this model-prediction explanation objective while adapting the method to the supported PyTorch Geometric interface.

Important limitation: this implementation is paper-aligned but is not an exact reproduction of the original TensorFlow implementation. The original paper formulates the method around the trained GNN prediction distribution. The current PyG adaptation uses `explanation_type="phenomenon"` with the frozen model's predicted class as the explicit classification target. This should not be described as mathematically identical to the original TensorFlow objective.

Preferred thesis wording:

```text
The chosen target preserves the intended post-hoc semantics of explaining the frozen GNN prediction, while adapting the method to the supported interface of PyTorch Geometric.
```

## Thesis Fairness Protocol

The following choices are controlled by the frozen Phase 4 thesis protocol, not by the original PGExplainer paper:

| Decision | Selected value | Source category | Rationale |
|---|---|---|---|
| Explanation budget | `ceil(candidate_edge_count * 0.2), minimum 1` | THESIS FAIRNESS PROTOCOL | Keep comparison fair with GNNExplainer and Random baseline. |
| Edge AUROC/AP score scope | Continuous scores over full candidate-edge set | THESIS FAIRNESS PROTOCOL | Preserve Phase 4 metric semantics. |
| Candidate-edge scope | Computation subgraph around target node | THESIS FAIRNESS PROTOCOL | Match frozen GNNExplainer evaluation. |
| Fid+/Fid- semantics | Reuse Phase 4 definitions | THESIS FAIRNESS PROTOCOL | Enable three-way comparison. |
| Jaccard stability | Reuse Phase 4 pairwise selected-edge-set Jaccard | THESIS FAIRNESS PROTOCOL | Use the same stability implementation. |
| Test-target policy | Exclude 12 frozen explainer-test targets from PGExplainer training | THESIS FAIRNESS PROTOCOL | Prevent leakage into the final explainer-test comparison. |
| Output schema | Reuse Phase 4 metric schema where applicable | THESIS FAIRNESS PROTOCOL | Compare GNNExplainer, PGExplainer, and Random consistently. |

## Compact Provenance Summary

PAPER / SUPPLEMENT:

- `epochs = 30`
- `lr = 0.003`
- `edge_size = 0.05`
- `edge_ent = 1.0`
- `temperature = 5.0 -> 2.0`

PYG_IMPLEMENTATION:

- `bias = 0.01`
- `explanation_type="phenomenon"` support
- `explanation_type="model"` unsupported
- `node_mask_type=None`
- `edge_mask_type="object"`
- actual installed PGExplainer API behavior in PyG `2.8.0.post1`

THESIS FAIRNESS PROTOCOL:

- fixed candidate-edge ratio `0.2`
- ceil rounding with minimum 1 selected edge
- shared candidate-edge scope
- shared AUROC/AP evaluator
- shared Fid+/Fid- semantics
- shared Jaccard definition
- frozen test-target policy

## Required Task 14 Follow-Up

Task 14 must freeze the PGExplainer protocol before training. It should add or verify tests for:

- Constructing PyG `phenomenon` targets from frozen model predictions.
- Ensuring ground-truth labels are not used as PGExplainer training targets by default.
- Ensuring the 12 frozen explainer-test targets are excluded from PGExplainer explainer-training targets.
- Ensuring the frozen 20% budget and candidate-edge scope remain unchanged.

## Stop Condition For This Step

This provenance step only audits sources and records implementation basis. It does not:

- train PGExplainer;
- evaluate PGExplainer;
- change the frozen GCN checkpoint;
- change Phase 4 metrics;
- change target lists;
- install new packages;
- change the selected-edge ratio.
