# GraphSAGE BA-House Freeze

## Scope

Task 18 trains and freezes a second BA-House prediction backbone using
GraphSAGE. The frozen dataset is reused exactly from
`artifacts/data/ba_house.pt` with SHA256
`4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94`.
No BA-House data, split masks, features, labels, or edge ordering were
regenerated.

## Architecture

GraphSAGE is configured in `configs/model/graphsage_ba_house.yaml`.

| Field | GCN | GraphSAGE |
|---|---|---|
| Layers | 3 | 3 |
| Hidden channels | 32 | 32 |
| Dropout | 0.2 | 0.2 |
| Task | 4-class node classification | 4-class node classification |
| Convolution | `GCNConv(cached=False)` | `SAGEConv(aggr="sum")` |

The unavoidable architecture difference is the message-passing operator.
`SAGEConv(sum)` is used because `SAGEConv(mean)` was degenerate on this
synthetic BA-House feature setting: mean aggregation lost neighborhood-size
signal that is useful for identifying structural motifs. This choice is frozen
before any GraphSAGE explanation experiment.

## Training Protocol

- Seed: `42`
- Optimizer: Adam
- Learning rate: `0.01`
- Weight decay: `0.0005`
- Max epochs: `300`
- Patience: `60`
- Class weighting: enabled
- Device: CUDA when available
- Checkpoint selection: validation split only
- Monitor: `val_accuracy`
- Test metrics: reported once after checkpoint freeze

## Frozen Checkpoint

- Checkpoint: `artifacts/checkpoints/ba_house_graphsage_best.pt`
- Manifest: `artifacts/checkpoints/ba_house_graphsage_best_manifest.json`
- Best epoch: `87`
- Best validation accuracy: `0.9500`
- Test accuracy: `0.9600`
- Test correct: `96/100`
- Motif test correct: `36/40`
- Majority baseline accuracy: `0.6000`
- Random baseline accuracy: `0.2500`
- Reload parity: passed with max absolute logit difference `0.0`

## Target Parity

The existing frozen 12 GCN explainer-test targets were inspected using
classifier outputs and ground-truth/candidate-mapping eligibility only.
No explainer output was used.

- Parity table: `results/metrics/model_target_parity_gcn_graphsage.csv`
- Frozen targets inspected: `12`
- GCN correct: `12/12`
- GraphSAGE correct: `12/12`
- Candidate mapping valid for both backbones: `12/12`
- Eligible for BA-House 2x2 comparison: `12/12`

## Outputs

- `configs/model/graphsage_ba_house.yaml`
- `artifacts/checkpoints/ba_house_graphsage_best.pt`
- `artifacts/checkpoints/ba_house_graphsage_best_manifest.json`
- `results/metrics/graphsage_ba_house_training/history.csv`
- `results/metrics/graphsage_ba_house_training/summary.json`
- `results/metrics/model_target_parity_gcn_graphsage.csv`
- `docs/graphsage_ba_house_freeze.md`

## Stop Boundary

GraphSAGE explainers were not run in Task 18. The checkpoint and target parity
are frozen for review before any GraphSAGE XAI work begins.
