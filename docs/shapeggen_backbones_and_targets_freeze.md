# ShapeGGen Backbones And Targets Freeze

## Scope

Task22 trains and freezes GCN and GraphSAGE backbones on the official GraphXAI
ShapeGGen artifact, then freezes final 2x2 XAI targets and common PGExplainer
train/validation pools before any ShapeGGen explanation is inspected.

## Dataset

- Artifact: `artifacts\data\shapeggen.pt`
- SHA256: `9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e`
- Protocol: `THESIS_SHAPEGGEN_OFFICIAL_V1`
- Upstream GraphXAI commit: `a11e65ffbc4df737f35522a8accf2283a8aeaa37`
- Split counts: `{'test': 261, 'train': 778, 'validation': 259}`

## Backbone Performance

| Backbone | Train Acc | Val Acc | Test Acc | Checkpoint SHA256 |
|---|---:|---:|---:|---|
| GCN | 0.692802 | 0.752896 | 0.681992 | `15bfa8960fe4e0f0712c3a794db93ed4e90aad6fce3555a16db23a8a13b91e36` |
| GraphSAGE | 0.998715 | 0.884170 | 0.869732 | `498a74875360a82749b78cb40a69b3e4ec04ec634b85e4713e85d96f34c666a0` |

Reload parity passed for both checkpoints.

## Frozen Target Counts

| Set | Count | SHA256 |
|---|---:|---|
| Final 2x2 targets | 167 | `bd128dec7810421d592debe8177f3cb46d6c7d9b01fd3f562b8b13971517b41d` |
| GCN train eligible | 539 | `17a69bd999853a8487a1f85af7bec21409e171dad8f021d99918f98192279b4f` |
| GraphSAGE train eligible | 777 | `d5fc96cdd8e9c6b4f4dab6e707ecfbf2e1c3b36fcdf525c004b446a8d74ddd91` |
| GCN validation eligible | 195 | `c3ec03b26f3636266acc0d0872a53d605b28670125249a098f18fd843b2de47f` |
| GraphSAGE validation eligible | 229 | `4189685c8facd2c6ff24fd12fc85715c47c97691c5760c9b7877eb13d7f448ba` |
| Common PG train | 539 | `c05a5be38a64b629c8bb837b1fb3408d8a008726713e70f0318aee3499746b70` |
| Common PG validation | 182 | `b988e40c2af234724b07802ec31d0831ef039a3ef4de3bfee1b80051b49cdc5d` |

Common PG pools are symmetric cross-backbone intersections by `node_id`.
They are disjoint from each other and from the final 2x2 test target set.

## Stop Boundary

No ShapeGGen GNNExplainer, PGExplainer, Random baseline, stability run, or final
XAI metric was executed in this task.
