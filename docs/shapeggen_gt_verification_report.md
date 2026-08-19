# ShapeGGen Ground-Truth Verification Report

## Verification Status

ShapeGGen protocol freeze status: passed.

Generated artifact:

- `artifacts/data/shapeggen.pt`
- SHA256: `ecf0975a1a38ed89225b91992331165b14879b943dc96360ca4406dbbaacc6ac`

## Checks Performed

The dataset generator and tests verify:

- Each planted motif id corresponds to exactly five nodes.
- Each planted motif subgraph is isomorphic to the NetworkX house motif.
- Labels encode one or two nearby motifs as `shapes_in_khop - 1`.
- Train, validation, and test masks are boolean, complete, and disjoint.
- `edge_index` contains reverse-edge copies.
- Candidate edge masks are target-specific model-layer computation edges.
- Positive edge masks are strict subsets of candidate edge masks.
- Every target has positive explanatory edges and negative candidate edges.
- Reverse-edge candidate masks and ground-truth masks are aligned.
- Dataset artifact SHA256 is locked in tests.

## Candidate And Ground-Truth Mapping

For target node `t`:

1. Compute the one-hop motif id set around `t`.
2. Compute the `model_layers=3` computation node set around `t`.
3. Candidate edges are directed PyG edges whose endpoints are both in that
   computation node set.
4. Positive nodes are motif nodes inside the computation node set whose motif id
   appears in the one-hop motif id set.
5. Positive edges are candidate edges whose endpoints are both positive nodes.
6. If `t` is not a positive motif node, source-to-motif candidate edges touching
   `t` and a positive motif node are also positive.

This mirrors the audited GraphXAI target-specific ShapeGGen explanation
semantics.

## Frozen Statistics

| Metric | Value |
|---|---:|
| Targets | 1279 |
| Candidate edge entries | 148852 |
| Positive edge entries | 27700 |
| Positive node entries | 10250 |
| Min candidate edges per target | 18 |
| Median candidate edges per target | 110 |
| Max candidate edges per target | 306 |
| Min positive edges per target | 12 |
| Median positive edges per target | 26 |
| Max positive edges per target | 28 |

## Test Coverage

ShapeGGen-specific tests:

- `tests/test_shapeggen_generation.py`
- `tests/test_shapeggen_gt_masks.py`
- `tests/test_shapeggen_metadata.py`
- `tests/test_shapeggen_reproducibility.py`

The tests are intended to fail if the frozen artifact changes unexpectedly, if
the split stops being disjoint, or if edge/candidate/ground-truth mapping drifts.

## Stop Boundary

No ShapeGGen backbone training or explainer evaluation was run during this task.
The next phase may train ShapeGGen backbones only after this protocol/data/GT
freeze is reviewed.
