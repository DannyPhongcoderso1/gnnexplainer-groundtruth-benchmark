# Project Plan

## Phase 1: Environment And PyG Smoke Test

- Create isolated virtual environment.
- Install stable PyTorch and PyTorch Geometric.
- Verify `GCNConv`, `Explainer`, and `GNNExplainer` on a toy graph.

## Phase 2: BA-House Dataset

- Generate BA-House with PyG `ExplainerDataset`.
- Validate ground-truth explanation masks.

## Phase 3: Benchmark GCN

- Train a small benchmark GCN.
- Select correctly predicted motif nodes for explanation.

## Phase 4: GNNExplainer

- Run PyG GNNExplainer.
- Save raw edge and node masks.

## Phase 5: Metrics

- Edge ROC-AUC.
- Edge Average Precision.
- Precision@k.
- Recall@k.
- Jaccard@k.
- Faithfulness.
- Sparsity.
- Runtime.

## Phase 6: Stability

- Stability across explainer seeds.
- Stability across model seeds.

## Phase 7: GraphSAGE

- Add a GraphSAGE benchmark after the GCN path is validated.

## Phase 8: Attributed Synthetic Benchmark

- Add a synthetic benchmark with feature ground truth.

## Phase 9: Elliptic Transfer

- Transfer validated generic XAI modules to an Elliptic feature branch.

Synthetic benchmarks do not prove that GNNExplainer is absolutely reliable. They
check correctness, ground-truth overlap, faithfulness, sparsity, and stability in
controlled conditions.
