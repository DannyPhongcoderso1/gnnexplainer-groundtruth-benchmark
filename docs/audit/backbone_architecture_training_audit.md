# Audit03 Backbone Architecture And Training Fairness

Date: 2026-09-04

Scope: deterministic evidence expansion and finalization only. Existing frozen
datasets, configs, histories, reports, and checkpoints were loaded read-only.
No backbone was retrained, no explainer was run, no Optuna study was rerun, and
no target/result artifact was modified.

## Stop Boundary

This audit does not authorize Audit04, Task28 Step04, full Task28 training,
final-target explanations, Task29, Random reruns, PGExplainer reruns,
GNNExplainer reruns, backbone retraining, thesis DOCX edits, README edits, or
baseline-repository edits.

## Amendment History

| Item | Value |
|---|---|
| Original Audit03 verdict | `AUDIT03 VERDICT: CONDITIONAL PASS` |
| External review verdict | `BLOCKED FOR DETERMINISTIC EVIDENCE EXPANSION` |
| Original report SHA256 before expansion | `07d815ad29df7a92eb9c4def1b749fdfc0ca396a271bceb19e9d85fba89c6a80` |
| Original CSV SHA256 before expansion | `6e041258f083adf79d791dc0561b461f2159753de9b4b7b60d89c814ac0023ce` |
| Expansion performed | Elliptic test exposure, final-refit semantics, update accounting, actual bias tensors, class weights, hyperparameter provenance, calibration definitions, comparison flags, and claim boundaries |

## Evidence

- Synthetic model code: `src/gnnexplainer_benchmark/models/gcn.py`,
  `src/gnnexplainer_benchmark/models/graphsage.py`,
  `src/gnnexplainer_benchmark/training/trainer.py`.
- Elliptic baseline code: `D:/CS/KLTN/elliptic-gnn-xai/model/GCN.py`,
  `D:/CS/KLTN/elliptic-gnn-xai/model/SAGE.py`,
  `D:/CS/KLTN/elliptic-gnn-xai/scripts/train.py`,
  `D:/CS/KLTN/elliptic-gnn-xai/utils/objectives.py`,
  `D:/CS/KLTN/elliptic-gnn-xai/data/dataset.py`.
- Elliptic baseline configs/results:
  `D:/CS/KLTN/elliptic-gnn-xai/config/training_baseline_paper_gcn.yaml`,
  `D:/CS/KLTN/elliptic-gnn-xai/config/training_baseline_paper_sage.yaml`,
  `D:/CS/KLTN/elliptic-gnn-xai/results/GCN/GCN_training_results.json`,
  `D:/CS/KLTN/elliptic-gnn-xai/results/SAGE/SAGE_training_results.json`.
- Output matrix: `docs/audit/backbone_fairness_matrix.csv`.

## Architecture Matrix

| Dataset | Backbone | Conv | Layers | Hidden | Embedding | Dropout | Aggregator | Self-loop / root behavior |
|---|---|---|---:|---:|---|---:|---|---|
| BA-House | GCN | GCNConv | 3 | 32 | same_as_output | 0.200000 |  | GCNConv default add_self_loops with normalize=True |
| BA-House | GraphSAGE | SAGEConv | 3 | 32 | same_as_output | 0.200000 | sum | SAGEConv root_weight=True, no explicit self-loop insertion |
| ShapeGGen | GCN | GCNConv | 3 | 32 | same_as_output | 0.200000 |  | GCNConv default add_self_loops with normalize=True |
| ShapeGGen | GraphSAGE | SAGEConv | 3 | 32 | same_as_output | 0.200000 | sum | SAGEConv root_weight=True, no explicit self-loop insertion |
| Elliptic | GCN | GCNConv | 2 | 211 | 90 | 0.236100 |  | GCNConv default add_self_loops with normalize=True |
| Elliptic | GraphSAGE | SAGEConv | 2 | 140 | 103 | 0.113500 | mean | SAGEConv root_weight=True, no explicit self-loop insertion |

## Actual Bias Audit

| Dataset | Backbone | Bias enabled | Bias tensor count | Bias parameter elements | Bias tensor names |
|---|---|---|---:|---:|---|
| BA-House | GCN | TRUE | 3 | 68 | ["convs.0.bias","convs.1.bias","convs.2.bias"] |
| BA-House | GraphSAGE | TRUE | 3 | 68 | ["convs.0.lin_l.bias","convs.1.lin_l.bias","convs.2.lin_l.bias"] |
| ShapeGGen | GCN | TRUE | 3 | 66 | ["convs.0.bias","convs.1.bias","convs.2.bias"] |
| ShapeGGen | GraphSAGE | TRUE | 3 | 66 | ["convs.0.lin_l.bias","convs.1.lin_l.bias","convs.2.lin_l.bias"] |
| Elliptic | GCN | TRUE | 3 | 303 | ["convs.0.bias","convs.1.bias","out.bias"] |
| Elliptic | GraphSAGE | TRUE | 3 | 245 | ["convs.0.lin_l.bias","convs.1.lin_l.bias","out.bias"] |

Repository evidence comes from the instantiated frozen model and its parameter
names, not from PyG defaults alone.

## Parameter-Count And Capacity Audit

| Dataset | GCN params | GraphSAGE params | Classification |
|---|---:|---:|---|
| BA-House | 1540 | 3012 | depth-matched, capacity-unmatched |
| ShapeGGen | 1474 | 2882 | depth-matched, capacity-unmatched |
| Elliptic | 54288 | 75491 | depth-matched, capacity-unmatched |

## Training Protocol Matrix

| Dataset | Backbone | Optimizer | LR | Weight decay | Class weights | Max/final epochs | Patience | Selected epoch/final epochs | Selection rule |
|---|---|---|---:|---:|---|---:|---|---:|---|
| BA-House | GCN | Adam | 0.010000 | 0.000500 | True | 300 | 60 | 10 | val_accuracy |
| BA-House | GraphSAGE | Adam | 0.010000 | 0.000500 | True | 300 | 60 | 87 | val_accuracy |
| ShapeGGen | GCN | Adam | 0.010000 | 0.000500 | True | 300 | 60 | 16 | val_accuracy |
| ShapeGGen | GraphSAGE | Adam | 0.010000 | 0.000500 | True | 300 | 60 | 263 | val_accuracy |
| Elliptic | GCN | Adam | 0.000847 | 0.000500 | False | 497 | none | 497 | Optuna best trial by validation AUPRC, then final train+val model |
| Elliptic | GraphSAGE | Adam | 0.000530 | 0.000500 | False | 397 | none | 397 | Optuna best trial by validation AUPRC, then final train+val model |

## Class Weights And Formula

| Dataset | Backbone | Enabled | Train class counts | Weight vector | Population | Same within dataset |
|---|---|---|---|---|---|---|
| BA-House | GCN | True | [180,48,48,24] | [0.4166666865348816,1.5625,1.5625,3.125] | BA-House train_mask labels only | TRUE |
| BA-House | GraphSAGE | True | [180,48,48,24] | [0.4166666865348816,1.5625,1.5625,3.125] | BA-House train_mask labels only | TRUE |
| ShapeGGen | GCN | True | [301,477] | [1.2923587560653687,0.8155136108398438] | ShapeGGen train_mask labels only | TRUE |
| ShapeGGen | GraphSAGE | True | [301,477] | [1.2923587560653687,0.8155136108398438] | ShapeGGen train_mask labels only | TRUE |
| Elliptic | GCN | False | not_used | not_used | not_used | TRUE |
| Elliptic | GraphSAGE | False | not_used | not_used | not_used | TRUE |

For synthetic datasets, class weights are computed from train labels only by
`train_labels.numel() / (num_classes * train_class_counts.clamp_min(1.0))`.
The vector is passed to `torch.nn.functional.cross_entropy` during training.
For Elliptic, class weights are not used because the inherited baseline
objective uses `nn.CrossEntropyLoss()` without a weight tensor. This is a
baseline-behavior inheritance, not a claim that accuracy alone is sufficient
under heavy class imbalance.

## Selected Checkpoint Versus Executed Updates

| Dataset | Backbone | Best checkpoint epoch | Training stop epoch | Updates/epoch | Updates in selected checkpoint | Total executed updates | Discarded updates after best | Hyperparameter-search updates | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| BA-House | GCN | 10 | 70 | 1 | 10 | 70 | 60 | not_applicable | D:\CS\KLTN\gnnexplainer-groundtruth-benchmark\results\metrics\gcn_training_history.csv |
| BA-House | GraphSAGE | 87 | 147 | 1 | 87 | 147 | 60 | not_applicable | D:\CS\KLTN\gnnexplainer-groundtruth-benchmark\results\metrics\graphsage_ba_house_training\history.csv |
| ShapeGGen | GCN | 16 | 76 | 1 | 16 | 76 | 60 | not_applicable | D:\CS\KLTN\gnnexplainer-groundtruth-benchmark\results\metrics\shapeggen_backbone_comparison\gcn_history.csv |
| ShapeGGen | GraphSAGE | 263 | 300 | 1 | 263 | 300 | 37 | not_applicable | D:\CS\KLTN\gnnexplainer-groundtruth-benchmark\results\metrics\shapeggen_backbone_comparison\graphsage_history.csv |
| Elliptic | GCN | 497 | 497 | 1 | 497 | 497 | 0 | UNKNOWN | final_refit_fixed_n_epochs_from_baseline_config |
| Elliptic | GraphSAGE | 397 | 397 | 1 | 397 | 397 | 0 | UNKNOWN | final_refit_fixed_n_epochs_from_baseline_config |

Synthetic training is full-graph with exactly one `optimizer.step()` per epoch.
The selected checkpoint contains the best validation epoch, while later executed
updates can be discarded by early stopping or by reaching `max_epochs`. Elliptic
final checkpoints are final train+validation refits for fixed `n_epochs`; total
historical Optuna-search compute is `UNKNOWN` because no complete original
search history is available in the audited artifacts.

## Elliptic Optuna Objective Trace

Trace: `scripts/train.py` creates an Optuna study with direction `maximize`,
wraps `objective_gnn`, passes `(train_mask, val_mask, test_mask)`, then
`objective_gnn` calls `GNN_features(selection_split="val",
evaluate_test=True)`. `GNN_features` returns validation AUPRC as `score` and
also computes test AUPRC when `evaluate_test=True`. `objective_gnn` stores
`test_score` in user attributes, asserts `objective_source == "val"`, and
returns `result["score"]`.

| Backbone | Programmatic test-based selection | Test metric computed per trial | Test metric logged/displayed per trial | Human-observable test exposure | Evidence human used test for tuning | Final one-time test evaluation |
|---|---|---|---|---|---|---|
| GCN | FALSE | TRUE | TRUE | TRUE | UNKNOWN | TRUE |
| GraphSAGE | FALSE | TRUE | TRUE | TRUE | UNKNOWN | TRUE |

Code evidence supports `programmatic_test_based_selection = FALSE`: test AUPRC
does not enter the Optuna return value, pruning, checkpoint selection, or
hyperparameter assignment. Observational exposure still exists because test
AUPRC is computed/stored per fixed trial and displayed in logs/results. Because
the repository does not contain a human decision log, whether a human used that
test exposure for tuning remains `UNKNOWN`.

## Elliptic Final Train+Validation Refit

After `best_trial` is selected by validation AUPRC, `scripts/train.py` builds a
new final model and sets `final_train_mask = train_mask OR val_mask`.
`GNN_features(selection_split="none", evaluate_test=True)` then trains for the
fixed final epoch count and evaluates the held-out test mask once. Test labels
are not used in the final refit loss. The benefit is that the final backbone can
use more labeled non-test data after hyperparameters are fixed. The trade-off is
that the original validation split is no longer an independent final-backbone
validation set. Later PGExplainer validation pools are explainer-validation
pools, not untouched backbone-validation evidence.

## Hyperparameter Provenance

| Dataset | Backbone | Provenance |
|---|---|---|
| BA-House | GCN | {"all_fields":"PROJECT_DEFINED: local synthetic config under configs/model with matched schedule fields and architecture-specific convolution"} |
| BA-House | GraphSAGE | {"all_fields":"PROJECT_DEFINED: local synthetic config under configs/model with matched schedule fields and architecture-specific convolution"} |
| ShapeGGen | GCN | {"all_fields":"PROJECT_DEFINED: local synthetic config under configs/model with matched schedule fields and architecture-specific convolution"} |
| ShapeGGen | GraphSAGE | {"all_fields":"PROJECT_DEFINED: local synthetic config under configs/model with matched schedule fields and architecture-specific convolution"} |
| Elliptic | GCN | {"aggregator":"not_applicable","dropout":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.dropout=0.2361; paper-table match=TRUE","embedding_dim":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.embedding_dim=90; paper-table match=TRUE","epochs":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.n_epochs=497; paper-table match=TRUE","hidden_dim":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.hidden_dim=211; paper-table match=TRUE","learning_rate":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.lr=0.0008475; paper-table match=TRUE","num_layers":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.num_layers=2; paper-table match=TRUE","optimizer":"INHERITED_BASELINE_CONFIGURATION: config optimizer.type=adam; not an Elliptic paper-specific value","seed":"PROJECT_DEFINED: config training.seed=42","weight_decay":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.weight_decay=0.0005; paper-table match=TRUE"} |
| Elliptic | GraphSAGE | {"aggregator":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.aggregator=mean; paper-table match=TRUE","dropout":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.dropout=0.1135; paper-table match=TRUE","embedding_dim":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.embedding_dim=103; paper-table match=TRUE","epochs":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.n_epochs=397; paper-table match=TRUE","hidden_dim":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.hidden_dim=140; paper-table match=TRUE","learning_rate":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.lr=0.0005302; paper-table match=TRUE","num_layers":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.num_layers=2; paper-table match=TRUE","optimizer":"INHERITED_BASELINE_CONFIGURATION: config optimizer.type=adam; not an Elliptic paper-specific value","seed":"PROJECT_DEFINED: config training.seed=42","weight_decay":"PAPER_REPORTED_EXACT_MATCH: local config fixed_hparams.weight_decay=0.0005; paper-table match=TRUE"} |

Elliptic uses two frozen backbone-specific configurations inherited from the
baseline workflow. The local values match the reviewed paper table values for
the listed hyperparameters, but this audit uses local config/result files as the
implementation evidence. Do not use the GCN or GraphSAGE architecture papers as
evidence for Elliptic-specific hyperparameter values.

## Performance And Calibration

| Dataset | Backbone | Params | Train acc | Val acc | Test acc | Test macro F1 | Test ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| BA-House | GCN | 1540 | 0.606667 | 0.650000 | 0.620000 | 0.350836 | 0.354853 |
| BA-House | GraphSAGE | 3012 | 0.953333 | 0.950000 | 0.960000 | 0.915314 | 0.179294 |
| ShapeGGen | GCN | 1474 | 0.692802 | 0.752896 | 0.681992 | 0.664249 | 0.072708 |
| ShapeGGen | GraphSAGE | 2882 | 0.998715 | 0.884170 | 0.869732 | 0.863678 | 0.064548 |
| Elliptic | GCN | 54288 | 0.987491 | 0.987221 | 0.963519 | 0.783143 | 0.025859 |
| Elliptic | GraphSAGE | 75491 | 0.994883 | 0.996555 | 0.969152 | 0.815375 | 0.022412 |

Calibration definitions: probabilities are `softmax(logits)` on the reported
evaluation split. ECE uses 10 equal-width confidence bins over max softmax
confidence and weights each non-empty bin by its sample fraction. Brier is
`mean(sum((probability - one_hot_label)^2))`, so the same multiclass formula is
used for binary and multiclass datasets. NLL is the mean negative log
probability of the true class, clamped at `1e-12`. These are post-hoc
descriptive diagnostics only. They do not participate in checkpoint selection,
do not prove perfect calibration, and should be read alongside probability-delta
metrics because those use the original predicted-class probability.

## Test Prediction And Class-Wise Metrics

| Dataset | Backbone | Test prediction distribution | Test class-wise metrics |
|---|---|---|---|
| BA-House | GCN | {"0": 70, "1": 30} | {"0": {"correct": 46, "predicted_count": 70, "recall": 0.7666666666666667, "total": 60}, "1": {"correct": 16, "predicted_count": 30, "recall": 1.0, "total": 16}, "2": {"correct": 0, "predicted_count": 0, "recall": 0.0, "total": 16}, "3": {"correct": 0, "predicted_count": 0, "recall": 0.0, "total": 8}} |
| BA-House | GraphSAGE | {"0": 60, "1": 18, "2": 13, "3": 9} | {"0": {"correct": 60, "predicted_count": 60, "recall": 1.0, "total": 60}, "1": {"correct": 16, "predicted_count": 18, "recall": 1.0, "total": 16}, "2": {"correct": 13, "predicted_count": 13, "recall": 0.8125, "total": 16}, "3": {"correct": 7, "predicted_count": 9, "recall": 0.875, "total": 8}} |
| ShapeGGen | GCN | {"0": 100, "1": 161} | {"0": {"correct": 59, "predicted_count": 100, "recall": 0.5841584158415841, "total": 101}, "1": {"correct": 119, "predicted_count": 161, "recall": 0.74375, "total": 160}} |
| ShapeGGen | GraphSAGE | {"0": 105, "1": 156} | {"0": {"correct": 86, "predicted_count": 105, "recall": 0.8514851485148515, "total": 101}, "1": {"correct": 141, "predicted_count": 156, "recall": 0.88125, "total": 160}} |
| Elliptic | GCN | {"0": 10836, "1": 348} | {"0": {"correct": 10488, "predicted_count": 10836, "recall": 0.9943117178612059, "total": 10548}, "1": {"correct": 288, "predicted_count": 348, "recall": 0.4528301886792453, "total": 636}} |
| Elliptic | GraphSAGE | {"0": 10843, "1": 341} | {"0": {"correct": 10523, "predicted_count": 10843, "recall": 0.9976298824421691, "total": 10548}, "1": {"correct": 316, "predicted_count": 341, "recall": 0.4968553459119497, "total": 636}} |

## Common-Correct Intersection

| Dataset | Split | Split total | GCN correct | GraphSAGE correct | Common-correct intersection |
|---|---|---:|---:|---:|---:|
| BA-House | train | 300 | 182 | 286 | 182 |
| BA-House | validation | 100 | 65 | 95 | 65 |
| BA-House | test | 100 | 62 | 96 | 62 |
| ShapeGGen | train | 778 | 539 | 777 | 539 |
| ShapeGGen | validation | 259 | 195 | 229 | 182 |
| ShapeGGen | test | 261 | 178 | 227 | 167 |
| Elliptic | train | 26381 | 26051 | 26246 | 25995 |
| Elliptic | validation | 8999 | 8884 | 8968 | 8869 |
| Elliptic | test | 11184 | 10776 | 10839 | 10729 |

Exact target eligibility and sampling flow: `PENDING_AUDIT04`. Audit03 records
common-correct counts and final target counts but does not explain why
BA-House has 62 common-correct test nodes and 12 final targets. Explanation
metrics on common-correct targets are conditioned on model correctness. BA-House
GCN does not predict classes 2 or 3 on the test split, so cross-backbone
explanation differences are confounded by predictive competence.

## Comparison Classification

| Dataset | Architecture-matched | Depth-matched | Update-matched | Capacity-unmatched | Conditional |
|---|---|---|---|---|---|
| BA-House | False | True | False | True | True |
| ShapeGGen | False | True | False | True | True |
| Elliptic | False | True | False | True | True |

## Comparison Match Flags

| Dataset | Backbone | Hyperparameter schedule matched | Depth matched | Capacity matched | Selected checkpoint update matched | Executed compute matched | Architecture matched | Target selection flow |
|---|---|---|---|---|---|---|---|---|
| BA-House | GCN | TRUE | TRUE | FALSE | FALSE | FALSE | FALSE | PENDING_AUDIT04 |
| BA-House | GraphSAGE | TRUE | TRUE | FALSE | FALSE | FALSE | FALSE | PENDING_AUDIT04 |
| ShapeGGen | GCN | TRUE | TRUE | FALSE | FALSE | FALSE | FALSE | PENDING_AUDIT04 |
| ShapeGGen | GraphSAGE | TRUE | TRUE | FALSE | FALSE | FALSE | FALSE | PENDING_AUDIT04 |
| Elliptic | GCN | FALSE | TRUE | FALSE | FALSE | FALSE | FALSE | PENDING_AUDIT04 |
| Elliptic | GraphSAGE | FALSE | TRUE | FALSE | FALSE | FALSE | FALSE | PENDING_AUDIT04 |

## Findings

1. Trong mỗi synthetic dataset, GCN và GraphSAGE dùng cùng optimizer, learning
   rate, weight decay, maximum epochs, patience, seed, class-weighting policy và
   full-graph update semantics. Tuy nhiên, chúng không capacity-matched và các
   selected checkpoints chứa số optimizer updates khác nhau do best validation
   epochs khác nhau.
2. Synthetic GCN and GraphSAGE are depth-matched but not architecture-matched and
   not capacity-matched. GraphSAGE has more parameters because `SAGEConv` has
   separate root/neighbor transformations, while `GCNConv` uses normalized graph
   convolution.
3. Synthetic selected epochs differ because early stopping selects the best
   validation checkpoint. This is an optimization outcome, not an equal-update
   guarantee.
4. Elliptic sử dụng hai frozen backbone-specific configurations kế thừa hoặc tạo
   bởi baseline workflow. GCN và GraphSAGE khác hidden dimension, embedding
   dimension, dropout, learning rate, aggregator, parameter count và final epoch
   count. Vì vậy, phép so sánh là giữa hai frozen backbone contexts, không phải
   controlled architecture-only comparison.
5. Elliptic final checkpoints are capacity-unmatched and update-unmatched.
   Cross-backbone comparisons must therefore remain descriptive backbone-context
   comparisons, not causal architecture effects.
6. No programmatic test-based checkpoint or hyperparameter selection was found.
   The Elliptic baseline still has human-observable per-trial test exposure
   because debug test AUPRC is computed/stored during the Optuna-compatible
   fixed-trial path.

## Valid Interpretation

- Within a fixed dataset and backbone, explainer comparisons remain cleaner than
  cross-backbone comparisons.
- Cross-backbone result differences can be reported as observations under frozen
  backbone contexts.
- Probability-delta comparisons should mention calibration diagnostics because
  confidence distributions differ by backbone.
- A design choice should be reported with what, why, evidence, chronology,
  benefit, trade-off, valid interpretation, and forbidden interpretation.

## Forbidden Interpretation

- Do not claim a pure causal effect of GCN versus GraphSAGE.
- Do not claim equal capacity where parameter counts differ.
- Do not claim equal optimization where selected epochs or final epoch counts
  differ.
- Do not treat Elliptic GCN and GraphSAGE as hyperparameter-matched.
- Do not call the Elliptic path absolute “no test leakage”; the precise wording
  is no programmatic test-based selection, with observational test exposure.
- Do not claim target-selection flow beyond `PENDING_AUDIT04`.

## External Semantics

- Kipf and Welling, `Semi-Supervised Classification with Graph Convolutional
  Networks`, ICLR 2017, DOI `10.48550/arXiv.1609.02907`, supports GCN as a
  graph convolution model using local graph structure and node features.
- Hamilton, Ying, and Leskovec, `Inductive Representation Learning on Large
  Graphs`, NIPS 2017, DOI `10.48550/arXiv.1706.02216`, supports GraphSAGE as
  neighborhood feature aggregation.
- PyG 2.8.0 `GCNConv` documentation records self-loop insertion under
  normalization and `cached` semantics for transductive use.
- PyG 2.8.0 `SAGEConv` documentation records configurable aggregation and
  root-weight behavior.

These sources explain model semantics. They do not prove this repository's
checkpoint quality, fairness, or implementation correctness.

## Stop-Condition Review

| Stop condition | Result |
|---|---|
| Test metric enters Optuna objective | Not observed |
| Test metric enters pruning | Not observed |
| Test metric selects checkpoint or best hyperparameters | Not observed |
| Test labels enter final refit | Not observed |
| Test metric logged/displayed per trial | Observed, limitation documented |
| Unexplained training asymmetry | Not a blocker, but Elliptic asymmetry is inherited/frozen baseline configuration and documented |
| Checkpoint/config mismatch | Not observed in loaded artifacts |
| Cached adjacency contradicts evaluation | Not observed for synthetic GCN. Elliptic baseline GCN uses cached=True, while XAI-compatible wrapper parity was addressed earlier |
| Missing backbone provenance | Not observed |
| Class-weight implementation differs unexpectedly between synthetic backbones | Not observed |
| Result values require non-trivial correction | Not observed |

## Severity Classification

| Finding | Severity |
|---|---|
| Wording/provenance/calibration documentation expanded | L1 |
| Deterministic update-count, test-exposure, class-weight, and checkpoint-selection audit expanded | L2 |
| Exact target-selection flow | PENDING_AUDIT04 |
| Test metric programmatically influences Optuna/pruning/checkpoint/hyperparameter assignment | Not observed, so no POTENTIAL L4 blocker |

## Verdict

AUDIT03 VERDICT: CONDITIONAL PASS.

Reason: no validity blocker was found, but cross-backbone comparisons are
capacity-unmatched and, for Elliptic, hyperparameter/update-unmatched. Claims
must remain descriptive and conditioned on the frozen backbone contexts.

Audit04 authorized: NO until external review.

Task28 Step04: NOT AUTHORIZED.

Full Task28 training: NOT AUTHORIZED.

Task29: NOT AUTHORIZED.

STOP FOR REVIEW.
