# BA-House Dataset Pipeline

Phase 1 generates a synthetic BA-House node-classification graph with PyTorch
Geometric `ExplainerDataset`. The background graph is Barabasi-Albert, the motif
is `HouseMotif`, and node features are constant by design so explanation quality
can be compared against PyG ground-truth node and edge masks.

## Configuration

The source of truth is:

```text
configs/dataset/ba_house.yaml
```

The current pilot configuration uses 300 background nodes, 40 house motifs, seed
42, 10 constant node features, and a 60/20/20 stratified node split.

## Generate

PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\generate_ba_house.py `
  --config configs\dataset\ba_house.yaml `
  --output artifacts\data\ba_house.pt `
  --metadata artifacts\data\ba_house_metadata.json
```

Generated data and metadata are local artifacts and are ignored by git.

## Inspect

```powershell
.\.venv\Scripts\python.exe scripts\inspect_ba_house.py --artifact artifacts\data\ba_house.pt
```

The inspector validates tensor shapes, finite values, non-empty ground-truth
masks, edge-mask length, and split coverage before printing a compact summary.

## Guarantees

- Python, NumPy, and PyTorch are seeded before generation.
- `edge_mask.numel()` matches `edge_index.size(1)`.
- Train, validation, and test masks are non-overlapping and cover every node.
- Metadata stores config, seed, shapes, label distribution, ground-truth counts,
  split counts, library versions, and the SHA-256 checksum of the artifact.
- No model training is part of this phase.
