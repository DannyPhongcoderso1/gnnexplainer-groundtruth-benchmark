# Elliptic architecture parity - static audit

No project import, torch.load, inference, optimizer or test execution was performed. Static comparison is not a new runtime-equivalence result. Thesis SHA256 `120cb35c8b7b23af4c8eb067e8844cd2658b3ff505fdf189aec62afc633fe5bc`; PDF page numbers below are 1-based and the printed page is one lower.

## Source identity and configuration

- Current GCN: `src/gnnexplainer_benchmark/models/elliptic/gcn.py:14-45` (`EllipticGCN`). Current GraphSAGE: `src/gnnexplainer_benchmark/models/elliptic/graphsage.py:14-45` (`EllipticSAGE`).
- Historical validated source: `$ELLIPTIC_BASELINE_ROOT/model/GCN.py` and `model/SAGE.py`, migrated baseline commit `95710a9852019104f90499cdba628c37f4483a2a`.
- Pinned upstream: [GCN](https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection/blob/83ef2b9ed8b3233175bbc3f4231c593c365225b9/model/GCN.py#L21) and [SAGE](https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection/blob/83ef2b9ed8b3233175bbc3f4231c593c365225b9/model/SAGE.py#L24), commit `83ef2b9ed8b3233175bbc3f4231c593c365225b9`. The two complete baseline class ASTs match that pinned revision. The URL was checked at this exact revision; current upstream main was not substituted.
- Accepted training configs: `configs/elliptic/backbone_training_gcn.yaml` and `backbone_training_graphsage.yaml`, moved without changing a byte. Their baseline preset explicitly sets graphnorm=false and init_mode=default; upstream optional defaults are not evidence of the accepted settings.
- Model bytes are bound to pre-existing tensor-schema metadata in `docs/provenance/elliptic_model_publication.json`. Checkpoint loading was not repeated. Original path/commit/hash and current published path/hash remain distinct.

| Property | GCN | GraphSAGE |
| --- | --- | --- |
| Thesis methodology |3.3.1, PDF42 / printed41; freezing3.3.3 PDF43 |3.3.2 and3.3.3, PDF43 / printed42 |
| Input/model features |165 |165 |
| Message-passing layers |GCNConv165->211; GCNConv211->90 |SAGEConv165->140; SAGEConv140->103 |
| Head and output |Linear90->2; raw class logits |Linear103->2; raw class logits |
| Nonlinearity/dropout |ReLU then p=.2361 after conv0 only |ReLU then p=.1135 after conv0 only |
| Aggregation/normalization |PyG GCN degree normalization; improved=false, normalize=true |Configured mean; root_weight=true, project=false, output L2 normalize=false |
| GraphNorm |Absent/disabled in accepted config; no gns state keys |Absent/disabled in accepted config; no gns state keys |
| Bias |Both conv biases and head bias |lin_l bias in both SAGE layers, lin_r has no bias; head bias |
| Self-loops/root |GCNConv normalization adds remaining self-loops; no extra residual |No explicit self-loop insertion by SAGEConv; separate root linear contribution is not an identity residual |
| Edges/forward |forward(x,edge_index); supplied global directed columns |forward(x,edge_index); supplied global directed columns |
| Parameter names |convs.0.*,convs.1.*,out.* |convs.0.lin_l/lin_r.*,convs.1.lin_l/lin_r.*,out.* |
| Accepted training metadata |lr=.0008475, weight_decay=.0005, seed42,497 epochs |lr=.0005302, weight_decay=.0005, seed42,397 epochs |
| Final static verdict |**INTENTIONAL_ADAPTATION** |**STATIC_MATCH**, configured frozen forward/state subset |

Operator defaults above were read from the locked local PyG2.8.0.post1 source, not inferred from class names. Both published dependency locks name that same version. No claim is made about a newer PyG version.

## Saved checkpoint schema

Dimensions are read from existing validated JSON metadata; this task only hashes the binary files.

| Backbone | Parameter | Recorded shape |
| --- | --- | --- |
| gcn | `convs.0.bias` | `[211]` |
| gcn | `convs.0.lin.weight` | `[211, 165]` |
| gcn | `convs.1.bias` | `[90]` |
| gcn | `convs.1.lin.weight` | `[90, 211]` |
| gcn | `out.weight` | `[2, 90]` |
| gcn | `out.bias` | `[2]` |
| graphsage | `convs.0.lin_l.weight` | `[140, 165]` |
| graphsage | `convs.0.lin_l.bias` | `[140]` |
| graphsage | `convs.0.lin_r.weight` | `[140, 165]` |
| graphsage | `convs.1.lin_l.weight` | `[103, 140]` |
| graphsage | `convs.1.lin_l.bias` | `[103]` |
| graphsage | `convs.1.lin_r.weight` | `[103, 140]` |
| graphsage | `out.weight` | `[2, 103]` |
| graphsage | `out.bias` | `[2]` |

These6 GCN and8 GraphSAGE state entries agree with the configured layers/head and contain no GraphNorm state. Metadata-to-file SHA256 binding is checked statically; no new load or numerical comparison is claimed.

## Differences and consequences

| Finding | Exact source / thesis evidence | Scientific consequence / compatibility |
| --- | --- | --- |
| Intentional GCN cache adaptation |Canonical GCN constructor above; pinned model/GCN.py:51-70 hard-codes cached=true; thesis3.3.1 PDF42 explicitly disables caching for perturbation |Layer/state names and dimensions are unchanged. Cached and uncached reuse are not equivalent across changing edge sets; cached=false is necessary for intended removal/retention semantics. Earlier parity evidence is archived in docs/elliptic_targets_freeze.json and provenance/historical_reports/elliptic_model_parity_report.md; its reported zero difference is historical evidence, not a rerun. |
| Configured GraphNorm differs from generic defaults |Pinned SAGE constructor defaults graphnorm=true, but the accepted YAML preset is false; GCN docstring mentions true while its signature defaults false |Read actual accepted config/schema, not docstrings. Canonical absence of GraphNorm matches these checkpoints; enabling it would add parameters and change behavior. |
| Frozen-compatible implementation is not a general upstream constructor replacement |Pinned reset_parameters and optional output_dim=0/GraphNorm/Xavier branches versus canonical constructors |The accepted domain is two layers, positive2-class head, no GraphNorm, loaded frozen state. RNG consumption/reset order and unsupported upstream modes are not claimed equivalent for fresh training. No model computation was rewritten. |
| Thesis GCN head description is less explicit |Thesis3.3.1 PDF42 names the90-dimensional representation; current out Linear and saved schema include2-class head |Description gap, not evidence of a90-class model or a checkpoint mismatch. GraphSAGE paragraph explicitly mentions its Linear head. Clarify in thesis prose if desired; do not remove either head. |
| Upstream annotation is misleading |Pinned GCN.forward:93-106 annotates Tuple but returns one out tensor |Canonical Tensor annotation matches the executable return; no tuple-return behavior is inferred from annotation. |

## Dataset and scope

Data3.2.3 PDF39-41 agrees with `src/gnnexplainer_benchmark/data/elliptic.py`: provider row ordering; reverse edges when directed=false; invalid/self-loop/duplicate removal; class1->1,2->0,unknown->2; known-label masks for time<30,30<=time<40,time>=40; time_step excluded from model features. Saved metadata gives203,769 nodes,468,710 directed entries,165 features and26,381/8,999/11,184 supervised nodes. The dataset module AST is unchanged, not executed. Unknown nodes remain available for aggregation. Absence of future structural/feature leakage cannot be proved from masks alone and is **NOT_VERIFIED** here.

No incompatible accepted architecture was found in the inspected source/config/schema evidence. This is not proof of runtime correctness. All runtime parity, successful loading, numerical reproduction and alternative-mode behavior after cleanup are **NOT_VERIFIED**. Canonical class bodies were left unchanged.

The raw loader maps edge endpoints by transaction ID but consumes mapped class_df values in provider row order (data/elliptic.py:64-70). Consistent feature/class row ordering is an input assumption; this static task does not verify original raw CSV row alignment or independently audit labels. Frozen graph hashes and the original provider ordering must be respected.
