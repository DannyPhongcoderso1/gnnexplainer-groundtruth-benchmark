# ShapeGGen Implementation Provenance

## Purpose

ShapeGGen is introduced as the second synthetic benchmark after BA-House. This
document freezes the implementation basis before any ShapeGGen backbone training
or explainer evaluation.

## Audited Sources

- GraphXAI paper: *GraphXAI: Graph Neural Networks for Explainable AI*.
- Official project page: `https://zitniklab.hms.harvard.edu/projects/GraphXAI/`.
- Official source repository: `https://github.com/mims-harvard/GraphXAI`.
- Audited official repository commit: `a11e65ffbc4df737f35522a8accf2283a8aeaa37`.
- Local package audit: `graphxai` is not installed in the current virtual environment.

## Source Files Audited

- `graphxai/datasets/shape_graph.py`
- `graphxai/datasets/utils/bound_graph_pref_att.py`
- `graphxai/datasets/utils/label_generators.py`
- `graphxai/datasets/utils/verify.py`
- `graphxai/datasets/utils/shapes.py`
- `graphxai/datasets/utils/feature_generators.py`
- `graphxai/datasets/feature/structured_feature.py`

The temporary audited clone was kept outside this repository at
`D:\CS\KLTN\_tmp_graphxai_source_task21` and is not a project artifact.

## Official Semantics Captured

The local implementation follows the official ShapeGGen variant-1 construction:

- Start from house motifs.
- Grow each motif with a Barabasi-Albert-like preferential attachment process.
- Connect subgraphs probabilistically while keeping each node near one or two motifs.
- Label each node by the number of unique nearby motifs minus one.
- Build structured Gaussian features from labels.
- Store PyG directed `edge_index` with reverse-edge copies.
- Define target-specific ground truth over the model-layer computation subgraph.

Official target-specific explanation semantics were preserved:

- Important nodes are motif nodes whose motif id appears in the target node's
  one-hop motif set.
- Important edges are candidate edges whose endpoints are both important motif
  nodes.
- If the target node is not itself a motif node, direct source-to-motif candidate
  edges are also marked important.

## Thesis Protocol Decisions

These are intentional frozen thesis-protocol decisions and are not explainer-driven:

- `ShapeGGen` is implemented locally because the `graphxai` package is not
  installed in the project environment.
- The official independently sampled train, validation, and test masks were not
  reused because they can overlap. The thesis protocol uses stratified disjoint
  masks.
- Feature explanations are stored as metadata via `feature_mask`, but feature
  explanation metrics are excluded from core thesis metrics.
- Candidate and ground-truth edge masks use the same directed PyG edge indexing
  that future GNNExplainer, PGExplainer, and Random evaluations will consume.
- Future PGExplainer comparisons must use a common candidate pool within the same
  backbone and dataset. Natural-pool results may be reported only as diagnostic
  results, not as the primary fair comparison protocol.

## Frozen Implementation Files

- `configs/dataset/shapeggen.yaml`
- `src/gnnexplainer_benchmark/data/shapeggen.py`
- `scripts/generate_shapeggen.py`
- `scripts/inspect_shapeggen.py`
- `artifacts/data/shapeggen.pt`
- `artifacts/data/shapeggen_manifest.json`

No ShapeGGen backbone training, GNNExplainer, PGExplainer, Random baseline, or
GraphSAGE evaluation has been started as part of this freeze.
