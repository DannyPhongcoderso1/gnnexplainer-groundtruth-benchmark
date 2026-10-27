# PGExplainer GCN Stability And Runtime Freeze

## Scope

Task 16B trains independent PGExplainer checkpoints for seeds `0,1,2,3,4`
and evaluates the same 12 frozen final targets only for pre-specified stability
and sensitivity diagnostics. The primary quality result remains seed `42` and
is not replaced by any stability seed.

## Frozen Inputs

- Dataset SHA256: `4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94`
- Frozen GCN checkpoint SHA256: `4c81814172650e20b6edee5fc97bc66cae46ac1e8ba76baec2226c59a7d1d507`
- PG train targets: `48 / 05e4574515fb7a03b846adb677295c940eb7af380216c15dfedf65bb9430bc3c`
- PG validation targets: `16 / 57fc6135495d9a6f8c7b8c1c3babcddd6b76187ebb353ed6fd8c289771980776`
- Final targets: `12 / 2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f`
- Primary seed42 checkpoint SHA256: `a075cd84c52e14a65b042f99f33e04e6b37e4919a70132ccb71e505b08c12eda`

## Stability Protocol

- Stability seeds: `[0, 1, 2, 3, 4]`
- Epochs per checkpoint: `30`
- Selected-edge budget: fixed candidate-edge ratio `0.2`
- Pairwise Jaccard per node: `C(5,2) = 10` seed pairs
- Node aggregation: equal-weight mean of per-node Jaccard means

## Acceptance Checks

- Independent checkpoints trained: `5/5`
- Final-target evaluation batches successful: `5/5`
- Total explained final targets: `60/60`
- Protocol parity: candidate counts, GT counts, selected counts, and original
  predictions match Phase 4 for `12/12` targets per seed.
- Post-test tuning: `false`
- Best-seed selection: `false`

## Code Provenance

- Stability code base commit: `0f8fff7acf8d9fe4ed976ecefd4236c0fdc54a0a`
- Stability script SHA256: `201e38631cc88d8621bdd551d9982a177f911105b8f52a5798e902b4b43b9e66`
- Training module SHA256: `50a4f5c2d123b4d5c250d31cc81c8b6f5baf26f04816ab4823ab5fc8b70c34be`
- Evaluation module SHA256: `423082d620a177b911397c8c4d059e78709e032a351ce59dc99a65f7ea9e8f16`

## Jaccard Stability

| Metric | Value |
|---|---:|
| Node count | 12 |
| Pairwise rows | 120 |
| Mean | 0.951911 |
| Std | 0.047503 |
| Median | 0.964542 |
| Min | 0.866667 |
| Max | 1.000000 |

## Runtime Semantics

PGExplainer runtime is separated into upfront training, trained-explainer
inference, and per-seed evaluation batch wall time. These stages are not
collapsed into one ambiguous runtime.

| Runtime | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Training per checkpoint | 10.542066 | 0.648016 | 10.524398 | 9.846523 | 11.489800 |
| Inference per target | 0.003200 | 0.000622 | 0.003027 | 0.002115 | 0.004901 |
| Evaluation per seed batch | 0.270408 | 0.033898 | 0.257610 | 0.247712 | 0.329742 |

## Freeze Statement

The PGExplainer GCN stability/runtime run is frozen for seeds `0..4`. Seed42
remains the primary quality checkpoint. The stability results are diagnostics
for sensitivity and must not be used to tune hyperparameters, alter target
membership, choose a better seed, or replace the primary seed42 result.
