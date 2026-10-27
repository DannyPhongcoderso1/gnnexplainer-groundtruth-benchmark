# Final saved results

| Dataset | Accepted evidence | Notes |
| --- | --- | --- |
| BA500 | [metrics](ba_house/ba500/metrics), [common-pool sensitivity](ba_house/ba500/common_pool_sensitivity) | Keep GraphSAGE PG natural/common cohorts distinct |
| BA1000 / BA1500 | [scale comparison](ba_house/scale_comparison) | Original combined per-target CSVs retain their scale column; no rows were split/recomputed |
| ShapeGGen V1 | [metrics](shapeggen/v1/metrics), [supplementary diagnostics](shapeggen/v1/diagnostics) | Diagnostics do not replace accepted point results |
| Elliptic | [historical final200](elliptic/final200), [FRESH5 stability](elliptic/stability), [GNN saved metrics](elliptic/gnnexplainer) | Historical seed42 is separate from FRESH5 |

[The unchanged thesis trace](final/thesis_traceability.json) retains 217 cells and nine withheld BA500 cohort-conflict cells. Its original release_evidence paths are resolved through the [path catalog](../docs/provenance/repository_paths.json). [The artifact catalog](../docs/provenance/external_assets.json) identifies checkpoint, seed, pool and dataset lineage. No saved measurement was edited or substituted.

Older results/metrics and results/explanations directories retain raw caches and historical supporting records needed by existing protocols. Removing them solely to shorten the tree would discard evidence. Current navigation starts with the table above; the original release remains immutable for comparison.

```powershell
python scripts/verify/verify_release_evidence.py
```

This verifies saved hashes and arithmetic only. Original figure builders remain unlocated, and this task did not generate new results.
