# ShapeGGen Ground-Truth Verification Report

## Verification Status

Task21C official GraphXAI migration status: passed candidate audit.

Generated production artifact:

- `artifacts/data/shapeggen.pt`
- SHA256: `9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e`
- Upstream commit: `a11e65ffbc4df737f35522a8accf2283a8aeaa37`

## Official Ground-Truth Source

Official `ShapeGGen(..., make_explanations=True)` returns
`dataset.explanations`, a list with one `Explanation` object per target node.
Each object contains:

- `node_idx`: target node id;
- `feature_imp`: global feature-importance mask;
- `node_imp`: node GT over the enclosing subgraph nodes;
- `edge_imp`: edge GT over the enclosing subgraph edge order;
- `enc_subgraph.nodes`: full-graph node ids in the target enclosing subgraph;
- `enc_subgraph.edge_mask`: full-graph edge-index aligned candidate mask;
- `enc_subgraph.edge_index`: enclosing-subgraph edge order.

## Local Conversion Checks

The adapter verifies:

- exactly one upstream explanation exists per target node;
- upstream `enc_subgraph.edge_mask` length equals full graph `edge_index` length;
- upstream `edge_imp` length equals enclosing-subgraph edge count;
- upstream `node_imp` length equals enclosing-subgraph node count;
- feature masks are globally consistent across targets;
- local `edge_mask[target]` is a subset of `candidate_edge_mask[target]`;
- reverse-edge candidate and GT masks are aligned;
- split masks are boolean, disjoint, and complete;
- all tensors are finite where applicable.

## Frozen Statistics

| Metric | Value |
|---|---:|
| Targets | 1298 |
| Directed edge entries | 4518 |
| Undirected edges | 2259 |
| Candidate edge entries | 156266 |
| Positive edge entries | 28316 |
| Positive node entries | 10420 |
| Eligible targets | 1298 |
| Min candidate edges per target | 22 |
| Median candidate edges per target | 118 |
| Max candidate edges per target | 296 |
| Min positive edges per target | 12 |
| Median positive edges per target | 26 |
| Max positive edges per target | 28 |

## Feature Ground Truth

Feature GT is stored as a global boolean `feature_mask` of shape `[10]`.
Positive upstream feature indices after official feature shuffling are:

```text
0, 1, 3, 6, 8, 9
```

Feature explanation metrics remain excluded from the core thesis comparison.

## Evaluator Contract

Future ShapeGGen evaluation must retrieve edge GT through:

```python
get_target_edge_ground_truth(data, node_id)
get_target_candidate_edge_mask(data, node_id)
```

This isolates ShapeGGen target-specific storage from BA-House's 1D global
ground-truth masks.

## Stop Boundary

No ShapeGGen GCN or GraphSAGE training, final target selection, GNNExplainer,
PGExplainer, Random baseline, or stability experiment was started during this
freeze.
