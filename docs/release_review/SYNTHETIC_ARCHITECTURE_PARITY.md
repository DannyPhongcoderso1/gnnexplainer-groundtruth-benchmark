# Synthetic architecture parity - static audit

Scope: BA500, BA1000, BA1500 and ShapeGGen V1. No project execution or checkpoint loading. Thesis SHA256 `120cb35c8b7b23af4c8eb067e8844cd2658b3ff505fdf189aec62afc633fe5bc`; PDF page numbers below are 1-based and the printed page is one lower.

| Canonical source | Accepted scientific definition | Thesis |
| --- | --- | --- |
| `src/gnnexplainer_benchmark/models/synthetic/gcn.py:8-32` (`GCN`) |3 GCNConv layers, channels10->32->32->C; ReLU/dropout.2 on first2, raw logits from final layer; cached=false from accepted config |3.3.1 PDF42 / printed41 |
| `src/gnnexplainer_benchmark/models/synthetic/graphsage.py:8-32` (`GraphSAGE`) |3 SAGEConv layers, same channels; configured sum aggregation, root linear contribution, no L2 output normalization or GraphNorm; first2 ReLU/dropout.2 |3.3.2 PDF43 / printed42 |

| Dataset | C | Feature/split identity | Accepted configuration evidence |
| --- | ---: | --- | --- |
| BA500 |4 |500 nodes, constant10 ones, seed42,300/100/100 split |configs/ba_house/ba500/gcn.yaml and graphsage_ba_house.yaml; saved GraphSAGE manifest artifacts/checkpoints/ba_house_graphsage_best_manifest.json |
| BA1000 |4 |1,000 nodes, same feature convention,600/200/200 |Original source receipt ba1000_ext_gcn_best_manifest.json / GraphSAGE counterpart and dataset-first config; architecture.channels=[10,32,32,4] |
| BA1500 |4 |1,500 nodes, same feature convention,900/300/300 |Original ba1500 counterpart source receipts and dataset-first config; same architecture family |
| ShapeGGen V1 |2 |1,298 nodes;10 features comprising6 informative and4 noise; generation/split seed912;778/259/261 |configs/shapeggen/v1; data/shapeggen/v1/manifest.json; original frozen backbone/target report |

Frozen training settings agree with the methodology: Adam lr=.01, weight_decay=.0005, maximum300 epochs, patience60, seed42. Saved synthetic configs also enable class_weighting; that additional optimization detail is not explicit in the short thesis architecture paragraph. No loss weighting was changed. Static config comparison is not evidence that training was rerun.

The generic GraphSAGE class default is mean; actual accepted configs and checkpoint builders supply sum. The default alone would lead to the wrong audit conclusion. Aggregation cannot be inferred from tensor shape, because sum and mean share parameter dimensions.

For GCN, expected weights are[32,10],[32,32],[C,32] with biases[32],[32],[C]. SAGE expected lin_l/lin_r weights use those layer dimensions and lin_l biases. These are source/config expectations. The available published synthetic receipts record architecture/config/byte identity but do not provide a complete tensor-by-tensor JSON schema for every synthetic backbone. A fresh complete stored-shape/load check is **NOT_VERIFIED** and was not attempted. Earlier strict-load attestations are historical, not this task's execution evidence.

All four datasets legitimately use the same two synthetic classes with different output dimensions/configs. BA-House and ShapeGGen generation, feature conventions, labels and ground-truth functions remain separate. `data/ba_house.py`, `data/shapeggen.py`, `data/splitting.py` and the scale extension retain their original computational ASTs. Three-hop candidate neighborhoods match the synthetic depth; Elliptic uses two hops. Input graph storage retains reverse directed columns; no new edge coalescing or self-loop policy was introduced by cleanup.

Source class ASTs match the pre-cleanup frozen implementations. Root models/gcn.py and models/graphsage.py were forwarding files with no model computation and have been removed after consumer redirects; models/__init__.py now exports canonical classes directly. No per-BA-size model copy was added.

**Final static verdict: STATIC_MATCH** for inspected accepted source/config/recorded architecture. **NOT_VERIFIED** for current runtime behavior and unavailable complete synthetic tensor-schema evidence. The existing BA500 GraphSAGE PG natural106/35 versus common48/16 conflict remains a **THESIS_MISMATCH** in cohort attribution (3.2.1/Table3.1, PDF37); it is not an architecture mismatch. Both saved artifact families and the nine withheld trace cells remain unchanged.
