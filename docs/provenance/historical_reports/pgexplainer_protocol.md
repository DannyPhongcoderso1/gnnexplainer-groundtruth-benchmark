# PGExplainer GCN Protocol Freeze

## Scope

This protocol freezes the PGExplainer experiment design before training. It does not train PGExplainer and does not evaluate the final 12 frozen explainer-test targets.

Frozen Phase 4 inputs remain unchanged:

- Dataset: `artifacts/data/ba_house.pt`
- GCN checkpoint: `artifacts/checkpoints/ba_house_gcn_best.pt`
- Final explainer-test targets: `results/metrics/explainer_test_targets.csv`
- Candidate-edge scope: computation subgraph around the target node
- Edge ranking scope: continuous scores over the full candidate-edge set
- Explanation budget: fixed candidate-edge ratio `0.2`
- Rounding rule: `ceil(candidate_edge_count * 0.2), minimum 1`
- Metrics: shared Phase 4 implementation

## Source And Provenance

Implementation provenance is recorded in:

```text
docs/pgexplainer_implementation_provenance.md
```

The thesis implementation is paper-aligned and PyG-native. It uses the original paper and supplement for method/protocol intent and reported training hyperparameters, while the installed PyG `2.8.0.post1` source controls actual API behavior.

## PGExplainer Target Splits

PGExplainer is a learned explainer, so it uses explainer-training and explainer-validation pools that are separate from the final frozen test nodes.

Train pool:

```text
base_model split = train
AND motif/ground-truth-explanation eligible
AND frozen GCN prediction correct
AND candidate-edge mapping valid
AND not in frozen 12 explainer-test targets
```

Validation pool:

```text
base_model split = validation
AND motif/ground-truth-explanation eligible
AND frozen GCN prediction correct
AND candidate-edge mapping valid
AND not in frozen 12 explainer-test targets
```

Final test:

```text
results/metrics/explainer_test_targets.csv
```

The previous `explainer_validation_targets.csv` was created for Phase 4 budget selection. It must not be reused blindly as PGExplainer train/validation data unless its base split membership is explicitly verified.

Canonical tracked PGExplainer target files:

```text
configs/targets/pgexplainer_train_targets.csv
configs/targets/pgexplainer_val_targets.csv
```

Generated copies may also exist under `results/metrics/` for reporting, but they are not the authoritative protocol inputs.

## Frozen PGExplainer Target Sets

Canonical train target file:

```text
configs/targets/pgexplainer_train_targets.csv
```

Count:

```text
48
```

SHA256:

```text
05e4574515fb7a03b846adb677295c940eb7af380216c15dfedf65bb9430bc3c
```

Canonical validation target file:

```text
configs/targets/pgexplainer_val_targets.csv
```

Count:

```text
16
```

SHA256:

```text
57fc6135495d9a6f8c7b8c1c3babcddd6b76187ebb353ed6fd8c289771980776
```

Frozen final test target file:

```text
results/metrics/explainer_test_targets.csv
```

Count:

```text
12
```

SHA256:

```text
2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f
```

Frozen target ids:

```text
train = 300, 301, 305, 310, 320, 321, 330, 331, 340, 341, 345, 346, 351, 356, 360, 361, 366, 370, 371, 381, 390, 395, 400, 401, 405, 406, 410, 416, 420, 421, 430, 431, 435, 436, 440, 441, 455, 460, 461, 466, 471, 476, 480, 481, 485, 490, 495, 496
validation = 325, 326, 335, 336, 350, 386, 391, 396, 415, 425, 426, 446, 451, 456, 470, 475
final_test = 306, 315, 316, 355, 365, 375, 376, 411, 445, 465, 486, 491
```

Disjointness audit:

```text
train ∩ validation = empty
train ∩ frozen_test = empty
validation ∩ frozen_test = empty
```

Do not change target membership after looking at PGExplainer test performance.

Phase 4 `explainer_validation_targets.csv` audit:

```text
nodes: 311, 380, 385, 450
base_model_split: test
decision: do not use as PGExplainer train/validation data
```

Required columns:

```text
node_id
ground_truth_label
frozen_model_prediction
target_for_pgexplainer
confidence
base_model_split
is_ground_truth_motif
gcn_prediction_correct
candidate_edge_count
candidate_edge_mapping_valid
```

## Target Semantics

Installed PyG PGExplainer supports `explanation_type="phenomenon"` and does not support `explanation_type="model"`.

Therefore, each PGExplainer training instance must receive an explicit target:

```text
target_for_pgexplainer = frozen_model_prediction[node_id]
```

The default target must not be `data.y[node_id]` merely because the selected node is correctly predicted. The CSV stores both `ground_truth_label` and `frozen_model_prediction` for audit, but the training target for PGExplainer is the frozen model prediction.

This is a paper-aligned PyG adaptation: it preserves the post-hoc goal of explaining the frozen GNN prediction while using the supported PyG `phenomenon` API.

## Primary Configuration

Config files:

```text
configs/explainer/pgexplainer.yaml
configs/evaluation/pgexplainer_gcn.yaml
```

Paper-aligned starting configuration:

```yaml
algorithm: PGExplainer
implementation: torch_geometric
explanation_type: phenomenon
explanation_level: node
edge_mask_type: object
node_mask_type: null
epochs: 30
lr: 0.003
edge_size: 0.05
edge_ent: 1.0
temperature:
  start: 5.0
  end: 2.0
```

Installed-PyG implementation coefficient:

```yaml
bias: 0.01
source: installed PyG 2.8.0.post1 PGExplainer source
```

`bias` is not treated as one of the main paper-reported PGExplainer hyperparameters.

## Hyperparameter Policy

No large hyperparameter search is allowed.

Allowed path:

```text
paper default
-> smoke
-> validation-only sanity check
```

If the paper-default setting runs and the training loss is finite:

```text
KEEP DEFAULT
```

Changes are allowed only for:

- API incompatibility
- numerical failure
- zero or NaN masks
- obvious non-training

Any change must be decided using train/validation only. Final test metrics must not be used for tuning.

## Checkpoint Selection

Preferred checkpoint protocol:

```text
fixed 30 epochs paper protocol
save final epoch
```

No early stopping based on final test metrics is allowed. If validation checkpointing is introduced later, the validation metric and checkpoint rule must be defined before final test evaluation.

## Runtime Accounting

PGExplainer runtime reporting must use separate fields:

```text
explainer_training_runtime_seconds
explanation_inference_runtime_seconds
evaluation_runtime_seconds
```

Runtime must not be collapsed into a single ambiguous field. PGExplainer has a separate training phase, so the thesis must report both training cost and inference/evaluation cost.

## Stability Semantics

PGExplainer stability is defined across independently trained explainer checkpoints:

```text
same frozen GCN
same explainer train pool
same config
different PGExplainer training seeds
-> independently trained explainer checkpoints
-> explanation edge sets on same test nodes
-> pairwise Jaccard
```

Stability is not defined as merely changing an inference seed.

Planned stability seeds:

```text
0, 1, 2, 3, 4
```

This matches the GNNExplainer Phase 4 stability seed count.

## Acceptance Gate

Before PGExplainer training starts, the following must be true:

- PGExplainer train, validation, and final test target lists are disjoint.
- Final 12 test targets remain unchanged.
- Paper hyperparameters are documented.
- PyG implementation-specific deviations are documented.
- `target_for_pgexplainer` is the frozen model prediction.
- Runtime semantics are frozen.
- Stability semantics are frozen.
- No final PGExplainer training has started.
