# Frozen artifact access

The minimum catalog contains 71 model identities and five graphs. All 71 models and four synthetic graphs are in Git; only the Elliptic dataset is intentionally external. [The catalog](../provenance/external_assets.json) records exact dataset/backbone/explainer/seed/pool identities, hashes, original paths and current published paths.

| Dataset | Frozen graph | Models |
| --- | --- | --- |
| BA500 | [graph](../../data/ba_house/ba500/graph.pt) | [checkpoints](../../artifacts/checkpoints/ba_house/ba500) |
| BA1000 | [graph](../../data/ba_house/ba1000/graph.pt) | [checkpoints](../../artifacts/checkpoints/ba_house/ba1000) |
| BA1500 | [graph](../../data/ba_house/ba1500/graph.pt) | [checkpoints](../../artifacts/checkpoints/ba_house/ba1500) |
| ShapeGGen V1 | [graph](../../data/shapeggen/v1/graph.pt) | [checkpoints](../../artifacts/checkpoints/shapeggen/v1) |
| Elliptic | [provider acquisition](ELLIPTIC_MODELS.md) | [checkpoints](../../artifacts/checkpoints/elliptic) |

BA500 GraphSAGE PG common48_16 and natural106_35 directories preserve the actual training cohorts. Point seed42 and Stability seeds0-4 remain distinct. Original incidental optimizer metadata remains inside exact checkpoint bytes; no additional epoch/history package is added.

```powershell
python scripts/verify/checkpoints.py --check-load
python scripts/verify/verify_thesis_artifacts.py
```

The first command checks all 71 checkpoint hashes/four synthetic graph hashes and strictly loads model states on CPU without forward or training calls. The second is standard-library artifact discovery, including existing optional owner/source/restore paths. Use `--help` for explicit restore options; different existing files are never overwritten.

GNNExplainer has no shared model checkpoint. Its [raw cache inventory](../provenance/gnn_cached_artifacts.json) retains 5268 edge-evaluation identities: 2076 already published and3192 require separate access. No raw cache is added by this restructuring. Saved final numerical records permit aggregate verification without those caches; independent score/perturbation replay needs the original graph, targets, backbone and masks.

The old dataset-containing private Elliptic package remains an owner-local backup, never a public dataset channel. Obtain original CSVs through the provider and follow [preprocessing instructions](ELLIPTIC_MODELS.md). Repository access does not grant a new dataset or commercial license.
