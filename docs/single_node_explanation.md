# Single-Node GNNExplainer

Phase 3 runs PyG GNNExplainer for one BA-House node only. The target node is
selected from the test split, must belong to a house motif, and must be correctly
predicted by the saved GCN checkpoint. Target selection is based on logits and
masks, not by inspecting the visualization.

## Select Targets

```powershell
.\.venv\Scripts\python.exe scripts\select_target_nodes.py `
  --artifact artifacts\data\ba_house.pt `
  --checkpoint artifacts\checkpoints\ba_house_gcn_best.pt `
  --output results\metrics\target_nodes.csv
```

## Explain One Node

```powershell
.\.venv\Scripts\python.exe scripts\explain_one_node.py `
  --artifact artifacts\data\ba_house.pt `
  --metadata artifacts\data\ba_house_metadata.json `
  --checkpoint artifacts\checkpoints\ba_house_gcn_best.pt `
  --config configs\explainer\gnnexplainer.yaml
```

The output directory is:

```text
results/explanations/node_<id>/
```

It contains raw `edge_mask.pt`, raw `node_mask.pt`, `metadata.json`,
`top_edges.csv`, and `explanation.png`.

This phase validates that masks are finite, model weights remain unchanged during
explanation, and the computation subgraph edge mapping aligns with the full
graph. It does not run batch explanations and does not support scientific claims
from one node.
