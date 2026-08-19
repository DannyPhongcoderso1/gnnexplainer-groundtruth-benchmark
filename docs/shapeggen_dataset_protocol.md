# ShapeGGen Dataset Protocol

## Frozen Config

The frozen config is `configs/dataset/shapeggen.yaml`.

```yaml
dataset:
  protocol_name: THESIS_SHAPEGGEN_V1
  seed: 912
  model_layers: 3
  shape: house
  variant: 1
  num_subgraphs: 100
  prob_connection: 0.09
  subgraph_size: 13
  n_features: 10
  n_informative: 6
  class_sep: 0.5
  add_sensitive_feature: false
  split:
    method: stratified_disjoint
    train_ratio: 0.6
    val_ratio: 0.2
    test_ratio: 0.2
```

This config was frozen before any ShapeGGen model training or explainer
evaluation.

## Artifact

- Dataset artifact: `artifacts/data/shapeggen.pt`
- Manifest: `artifacts/data/shapeggen_manifest.json`
- Dataset SHA256: `ecf0975a1a38ed89225b91992331165b14879b943dc96360ca4406dbbaacc6ac`

## Dataset Summary

| Field | Value |
|---|---:|
| Nodes | 1279 |
| Directed edges | 4390 |
| Undirected edges | 2195 |
| Node features | 10 |
| Class 0 nodes | 508 |
| Class 1 nodes | 771 |
| Train nodes | 766 |
| Validation nodes | 255 |
| Test nodes | 258 |

## Ground-Truth Summary

| Field | Value |
|---|---:|
| Positive node-mask entries | 10250 |
| Positive edge-mask entries | 27700 |
| Candidate edge entries | 148852 |
| Eligible targets | 1279 |
| Min positive edges per target | 12 |
| Max positive edges per target | 28 |
| Min candidate edges per target | 18 |
| Max candidate edges per target | 306 |

Every node is eligible as a future explanation target because every target has
at least one positive explanatory edge and at least one negative candidate edge.

## Split Protocol

The split protocol is stratified and disjoint:

- Train: 60%
- Validation: 20%
- Test: 20%

The exact counts differ slightly from decimal ratios because counts are computed
per class before assigning the remainder to test.

The official GraphXAI split implementation was not copied because it samples
train, validation, and test masks independently. Independent sampling can produce
overlapping masks, which is incompatible with the thesis training protocol.

## Edge Representation

The graph is stored as a PyG directed `edge_index` with reverse-edge copies for
each undirected NetworkX edge. Candidate masks and ground-truth edge masks are
indexed by this exact `edge_index`.

For future explainers:

- Edge AUROC and Edge AP must use continuous scores over the full candidate-edge
  set for the target.
- Fidelity and sparsity budgets must select edges from the same candidate mapping.
- Reverse-edge handling must remain deterministic and must not collapse edge
  indices unless the same conversion is documented and applied to all methods.

## Scope Exclusion

Structured feature explanations are available through `feature_mask`, but feature
explanation evaluation is excluded from the core thesis comparison. The core
comparison remains edge/subgraph explanation quality.
