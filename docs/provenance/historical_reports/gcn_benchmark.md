# BA-House GCN Benchmark

Phase 2 trains a small benchmark GCN on the generated BA-House dataset. It uses
PyG `GCNConv`, raw logits, `cached=False`, and the node masks stored in the
dataset artifact. The best checkpoint is selected only by validation accuracy;
test metrics are computed after the best checkpoint is selected.

## Generate Dataset First

```powershell
.\.venv\Scripts\python.exe scripts\generate_ba_house.py `
  --config configs\dataset\ba_house.yaml `
  --output artifacts\data\ba_house.pt `
  --metadata artifacts\data\ba_house_metadata.json
```

## Train

```powershell
.\.venv\Scripts\python.exe scripts\train_gcn.py `
  --artifact artifacts\data\ba_house.pt `
  --metadata artifacts\data\ba_house_metadata.json `
  --config configs\model\gcn.yaml
```

Outputs are local and ignored by git:

```text
artifacts/checkpoints/ba_house_gcn_best.pt
results/metrics/gcn_training_history.csv
results/metrics/gcn_model_summary.json
```

## Evaluate

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_gcn.py `
  --artifact artifacts\data\ba_house.pt `
  --checkpoint artifacts\checkpoints\ba_house_gcn_best.pt
```

No GNNExplainer run is part of this phase.
