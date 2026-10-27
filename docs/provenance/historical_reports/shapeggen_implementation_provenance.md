# ShapeGGen Official GraphXAI Provenance

## Status

Task21C replaces the first local ShapeGGen reimplementation with the official
GraphXAI ShapeGGen implementation.

The historical Task21 commit
`e7d753361fbebc205aa574c62c2b2520cc93d61b` remains in Git history as the first
local-generation attempt. Its dataset SHA256 was
`ecf0975a1a38ed89225b91992331165b14879b943dc96360ca4406dbbaacc6ac`.
That artifact is superseded. No ShapeGGen backbone, explainer, Random baseline,
stability run, or PGExplainer pool was created from that historical artifact.

## Authoritative Source

- Upstream project: GraphXAI
- Repository: `https://github.com/mims-harvard/GraphXAI`
- Pinned commit: `a11e65ffbc4df737f35522a8accf2283a8aeaa37`
- License: MIT
- Upstream source class: `graphxai.datasets.shape_graph.ShapeGGen`
- Upstream source file: `graphxai/datasets/shape_graph.py`

The official repository is vendored unmodified under `src/graphxai` because its
`setup.py` at the pinned commit installs only the top-level package and does not
install required subpackages such as `graphxai.datasets` and `graphxai.utils`.

## Upstream Files Audited

- `graphxai/datasets/shape_graph.py`
- `graphxai/datasets/utils/bound_graph.py`
- `graphxai/datasets/utils/bound_graph_pref_att.py`
- `graphxai/datasets/utils/label_generators.py`
- `graphxai/datasets/utils/feature_generators.py`
- `graphxai/datasets/feature/structured_feature.py`
- `graphxai/datasets/utils/verify.py`
- `graphxai/datasets/utils/shapes.py`
- `graphxai/utils/explanation.py`

## Upstream Behavior

Official `ShapeGGen`:

- constructs a NetworkX graph from inserted motifs and surrounding subgraphs;
- supports variant `1`, the preferential-attachment construction;
- uses house motifs for this frozen benchmark;
- labels nodes by the number of nearby unique motifs minus one;
- creates structured label-correlated node features;
- optionally creates sensitive features and homophily-adjusted features;
- creates one target-specific `Explanation` object per node when
  `make_explanations=True`;
- stores each explanation over the target node's model-layer enclosing subgraph.

## Local Adapter Behavior

The local file `src/gnnexplainer_benchmark/data/shapeggen.py` is now an adapter.
It does not implement ShapeGGen graph construction, motif creation, label
generation, feature generation, homophily behavior, or explanation generation.

The adapter performs only benchmark integration:

- calls official `ShapeGGen`;
- copies the official PyG `Data` tensors;
- converts official `Explanation` objects to dense target-specific masks;
- attaches local stratified disjoint train/validation/test splits;
- validates edge/index/GT consistency;
- serializes the artifact and manifest.

## Ground-Truth Conversion

For each upstream explanation:

1. `Explanation.node_idx` gives the target node.
2. `Explanation.enc_subgraph.edge_mask` gives the target candidate edge set,
   aligned to the official full-graph `edge_index`.
3. `Explanation.edge_imp` gives binary edge GT over the enclosing subgraph edge
   order.
4. The adapter writes those values into `edge_mask[target_id, edge_id]`.
5. `Explanation.enc_subgraph.nodes` and `Explanation.node_imp` are converted to
   `node_mask[target_id, node_id]`.
6. `Explanation.feature_imp` is checked to be globally consistent and stored as
   `feature_mask`.

No motif reconstruction is used in the adapter.

## Dependency Compatibility

Direct source import from the official clone initially failed because upstream
imports a debug dependency, `ipdb`, through `graphxai.gnn_models...testing`.
The ShapeGGen path does not use `ipdb` behavior. A minimal compatibility shim is
provided in `src/ipdb.py` so the vendored upstream source can import without
modifying upstream files or adding a debug dependency to the thesis runtime.

## Future Protocol Note

ShapeGGen uses target-specific GT masks. Future evaluator code must access edge
GT through the adapter function `get_target_edge_ground_truth(data, node_id)`;
it must not assume that every dataset stores `data.edge_mask` as a 1D global
mask like BA-House.
