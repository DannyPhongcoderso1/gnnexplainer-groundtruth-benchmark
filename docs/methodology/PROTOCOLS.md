# Source and scientific protocols

| Component | Accepted source | Invariant |
| --- | --- | --- |
| Synthetic GCN / GraphSAGE | [models/synthetic](../../src/gnnexplainer_benchmark/models/synthetic) | Shared architecture code across BA sizes and ShapeGGen; dataset features/splits remain separate |
| Elliptic GCN / GraphSAGE | [models/elliptic](../../src/gnnexplainer_benchmark/models/elliptic) | Original layer/state names and cache-compatible forward semantics; separate from synthetic classifiers |
| GNNExplainer | [gnnexplainer.py](../../src/gnnexplainer_benchmark/explainers/gnnexplainer.py) | Original mask optimization and seed control |
| PGExplainer | [pgexplainer.py](../../src/gnnexplainer_benchmark/explainers/pgexplainer.py) | Accepted PyG builder; training/evaluation implementations remain separate |
| Random | [random_baseline.py](../../src/gnnexplainer_benchmark/explainers/random_baseline.py) | Original candidate scope and per-target seeds |
| Candidate mapping | [subgraph_mapping.py](../../src/gnnexplainer_benchmark/explain/subgraph_mapping.py) | Dataset-specific hop count and global directed edge identity |
| Metrics | [evaluation](../../src/gnnexplainer_benchmark/evaluation) | Existing edge AUROC/AP, recovery, fidelity, stability, sparsity and runtime definitions |

[Dataset modules](../../src/gnnexplainer_benchmark/data) retain separate BA-House and ShapeGGen generation, splitting, target selection and serialization. The Elliptic BCDataset implementation comes from the MIT baseline; preprocessing preserves node/edge order, reverse-edge handling, labels, features and temporal masks. [Elliptic details](../reproducibility/ELLIPTIC_MODELS.md) identify its license/source and graph fingerprints.

BA500 GraphSAGE PG has separate natural106/35 and common48/16 pools. The existing nine-cell thesis conflict stays unresolved; never exchange their checkpoint/result labels. Elliptic historical seed42 uses its original RAW ranking; FRESH5 seeds0-4 use the existing exact-tie global-ID addendum. Rejected and exploratory variants are not exported as the accepted PG API. ShapeGGen loss/gradient diagnostics are supplementary.

State dictionaries are loaded into the same layer definitions. Class bodies and generation logic are preserved during relocation; thin import aliases retain historical callers. Original upstream architectures remain under external/elliptic_baseline as the attributed parity reference rather than a second active classifier family. Historical protocol receipts and producer source snapshots remain in provenance/audit folders and the immutable prior release.
