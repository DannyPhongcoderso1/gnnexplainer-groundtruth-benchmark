# Audit02 Dataset, Split, Graph Visibility, And Preprocessing Audit

Date: 2026-09-03

Repository HEAD at audit start: `67a2e80c46c1bacda8df7ba7ec0fd336b1bae651`

Scope: deterministic evidence expansion and finalization only. No experiment was rerun. No dataset, split mask, target pool, checkpoint, config, metric, or result artifact was modified.

## Stop Boundary

This audit does not authorize Audit03, Task28 Step04, full Task28 training, final-target evaluation, Task29, GNNExplainer reruns, PGExplainer reruns, Random reruns, GCN retraining, GraphSAGE retraining, raw Elliptic edits, or baseline-repository edits.

The read-only baseline repository remained outside the edit scope. Its pre-existing `README.md` modification was not touched.

## Amendment History

| Item | Status |
|---|---|
| Original Audit02 verdict | `PASS WITH DOCUMENTED UNKNOWN ELLIPTIC FEATURE-PREPROCESSING PROVENANCE` |
| External review verdict | `BLOCKED FOR DETERMINISTIC EVIDENCE EXPANSION` |
| Original audit report SHA256 | `47a3406ef1006c81be49f019167aa04adbd74a4c938e6c7797da5be6b7228b00` |
| Original inventory SHA256 | `099e42b9f3d5a355da255a95a92f320325c594a0b12a476e0873552be5c9c876` |
| Expansion performed | Elliptic edge-boundary, time-transition, receptive-field, training-graph trace, symmetrization, feature provenance, split rationale, and `PENDING_AUDIT04` target-pool wording |

New evidence below must be read as a reviewed Audit02 expansion, not as evidence that existed in the first Audit02 draft.

## Dataset Inventory

| Dataset | Artifact SHA256 | Nodes | Directed edge entries | Unique undirected relations | Features | Label counts |
|---|---|---:|---:|---:|---:|---|
| BA-House | `4daeb6b5ca55ee1663fa79aa6be5a7fe1ebb86159aa9d60045aceb16fe271a94` | 500 | 3,398 | 1,699 | 10 | `{0:300, 1:80, 2:80, 3:40}` |
| ShapeGGen | `9a0d5c4b5ffc21a97e372d9e00f6a8971d7c88b063f2582b3a64332b9b0af39e` | 1,298 | 4,518 | 2,259 | 10 | `{0:502, 1:796}` |
| Elliptic | `f807b83a7c739b16f69b5676226ce441b64d5fe8de95506a7727f86e8451291b` | 203,769 | 468,710 | 234,355 | 165 | `{0:42019, 1:4545, 2:157205}` |

## Graph Structure Checks

| Dataset | Duplicate directed entries | Self-loops | Reverse coverage | Isolated nodes | Components | Largest component | Degree mean/median/min/max |
|---|---:|---:|---:|---:|---:|---:|---|
| BA-House | 0 | 0 | 3,398/3,398 | 0 | 1 | 500 | 6.796 / 5 / 2 / 71 |
| ShapeGGen | 0 | 0 | 4,518/4,518 | 0 | 1 | 1,298 | 3.481 / 3 / 1 / 15 |
| Elliptic | 0 | 0 | 468,710/468,710 | 0 | 49 | 7,880 | 2.300 / 2 / 1 / 473 |

## Split Counts And Class Distributions

| Dataset | Split rule | Source | Seed or boundary | Train | Validation | Test | Overlap |
|---|---|---|---|---:|---:|---:|---:|
| BA-House | Stratified random 60/20/20 by label | Project-defined config and `make_stratified_node_splits` | seed `42` | 300 | 100 | 100 | 0 |
| ShapeGGen | Stratified random 60/20/20 by label after official generator output | Project-defined local adapter and `make_stratified_node_splits` | seed `912` | 778 | 259 | 261 | 0 |
| Elliptic | Temporal known-label split | Inherited baseline loader | `<30`, `30-39`, `>=40` | 26,381 | 8,999 | 11,184 | 0 |

### BA-House Class Split

| Label | Total | Train | Validation | Test |
|---:|---:|---:|---:|---:|
| 0 | 300 | 180 | 60 | 60 |
| 1 | 80 | 48 | 16 | 16 |
| 2 | 80 | 48 | 16 | 16 |
| 3 | 40 | 24 | 8 | 8 |

The `300/100/100` counts are the exact 60/20/20 stratified split of 500 nodes.

### ShapeGGen Class Split

| Label | Total | Train | Validation | Test |
|---:|---:|---:|---:|---:|
| 0 | 502 | 301 | 100 | 101 |
| 1 | 796 | 477 | 159 | 160 |

The base split is `778/259/261`. The later `539/182/167` numbers are explainer-stage eligible/common target-pool counts, not base dataset split counts.

### Elliptic Class Split

| Split | Time-step rule | Count | Class distribution |
|---|---|---:|---|
| Train | `time_step < 30` and known label | 26,381 | `{0:23510, 1:2871}` |
| Validation | `30 <= time_step < 40` and known label | 8,999 | `{0:7961, 1:1038}` |
| Test | `time_step >= 40` and known label | 11,184 | `{0:10548, 1:636}` |
| Unknown | excluded from supervised masks | 157,205 | `{2:157205}` |

Unknown labels in supervised masks: `0`.

## Split Rationale, Source, And Chronology

| Dataset | Split design | Actual source | Rationale | Chronology | Benefit | Trade-off |
|---|---|---|---|---|---|---|
| BA-House | Stratified 60/20/20 | `configs/dataset/ba_house.yaml` and `src/gnnexplainer_benchmark/data/splitting.py` | `INFERENCE`: deterministic stratification preserves class proportions in a controlled synthetic benchmark | `ORIGINALLY_PRE_SPECIFIED` | Reproducible balanced masks | Random transductive split is not deployment-temporal |
| ShapeGGen | Official graph/features first, local stratified 60/20/20 after conversion | `configs/dataset/shapeggen.yaml`, `src/gnnexplainer_benchmark/data/shapeggen.py` | `INFERENCE`: separates official generator provenance from local backbone masks | `ORIGINALLY_PRE_SPECIFIED` | Reproducible split with official generator retained | Target-pool counts later differ from raw split counts |
| Elliptic | Temporal known-label split | `D:/CS/KLTN/elliptic-gnn-xai/data/dataset.py` | `INFERENCE_WITH_EXTERNAL_SOURCE_SUPPORT`: chronological transaction periods make temporal evaluation more appropriate than random mixing | `INHERITED_BASELINE_IMPLEMENTATION` | Avoids random label mixing across time | Full graph object is still passed to the model |

External sources support the Elliptic temporal/data semantics, but local code and artifacts are the implementation evidence.

## Transductive And Inductive Semantics

| Dataset | Training graph object | Loss mask | Inductive hidden-future isolation proven |
|---|---|---|---|
| BA-House | Full synthetic graph | `train_mask` for training, validation mask for selection | Not applicable to temporal future |
| ShapeGGen | Full synthetic graph | `train_mask` for training, validation mask for selection | Not applicable to temporal future |
| Elliptic | Full serialized graph object | temporal known-label train mask for training | `false` |

Elliptic is best described as `transductive_full_graph_temporal_supervision`. The model forward receives the full graph object, but deterministic edge and time audits show no stored cross-time relations in the frozen Elliptic graph. Therefore, train-period nodes have no observed future-period receptive-field exposure up to the checked 1-3 hop range. This is stronger than the first draft wording, but it is still not a strict induced-subgraph hidden-future training protocol.

## Elliptic Label Leakage Versus Context Exposure

Definitions used by this audit:

| Term | Definition |
|---|---|
| Future-label leakage | Validation/test/future labels used in train loss, target construction, or model selection |
| Future-feature exposure | Train-node representation can receive feature information from future-period nodes |
| Future-topology exposure | Train-node representation can depend on connectivity involving future-period nodes |
| Inductive hidden-future isolation | Future nodes and edges are absent from the training computation graph |

| Field | Result |
|---|---|
| Future-label leakage observed | `false` |
| Future-feature exposure observed for train nodes within 1-3 hops | `false` |
| Future-topology exposure observed for train nodes within 1-3 hops | `false` |
| Inductive hidden-future isolation proven | `false` |
| Training mode | `transductive_full_graph_temporal_supervision` |

Valid interpretation: future labels are not used for train loss, and the stored Elliptic graph has no cross-time receptive-field path from train-period nodes to validation/test-period nodes.

Forbidden interpretation: do not claim this is a strict inductive hidden-future protocol, because the training code passes the full graph object to the model.

## Elliptic Edge-Boundary Matrix

Canonicalization rule for unique relations: sort endpoint node ids inside each non-self-loop edge and de-duplicate.

### Stored Directed Edge Entries

| From/to group | TRAIN_KNOWN | VAL_KNOWN | TEST_KNOWN | UNKNOWN |
|---|---:|---:|---:|---:|
| TRAIN_KNOWN | 40,302 | 0 | 0 | 35,158 |
| VAL_KNOWN | 0 | 14,056 | 0 | 12,507 |
| TEST_KNOWN | 0 | 0 | 18,890 | 18,288 |
| UNKNOWN | 35,158 | 12,507 | 18,288 | 263,556 |

Directed matrix total: `468,710`.

### Canonical Unique Undirected Relations

| Endpoint group pair | Count |
|---|---:|
| TRAIN_KNOWN -- TRAIN_KNOWN | 20,151 |
| TRAIN_KNOWN -- VAL_KNOWN | 0 |
| TRAIN_KNOWN -- TEST_KNOWN | 0 |
| TRAIN_KNOWN -- UNKNOWN | 35,158 |
| VAL_KNOWN -- VAL_KNOWN | 7,028 |
| VAL_KNOWN -- TEST_KNOWN | 0 |
| VAL_KNOWN -- UNKNOWN | 12,507 |
| TEST_KNOWN -- TEST_KNOWN | 9,445 |
| TEST_KNOWN -- UNKNOWN | 18,288 |
| UNKNOWN -- UNKNOWN | 131,778 |

Unique relation total: `234,355`.

Boundary counts requested by review:

| Boundary | Unique relations |
|---|---:|
| Within-train | 20,151 |
| Train-validation | 0 |
| Train-test | 0 |
| Train-unknown | 35,158 |
| Validation-test | 0 |
| Validation-unknown | 12,507 |
| Test-unknown | 18,288 |

The `TRAIN_KNOWN -- UNKNOWN` boundary is same-period context because the time-step audit below found no cross-time relations.

## Elliptic Time-Step Edge Audit

| Quantity | Count/value |
|---|---:|
| Canonical unique relations checked | 234,355 |
| Same-time relations | 234,355 |
| Cross-time relations | 0 |
| Minimum absolute time gap | 0 |
| Maximum absolute time gap | 0 |
| Mean absolute time gap | 0.0 |
| Median absolute time gap | 0.0 |
| Relations crossing `29 -> 30+` | 0 |
| Relations crossing `39 -> 40+` | 0 |
| Relations connecting train-period to test-period | 0 |

Stored directed entries:

| Stored direction category | Count |
|---|---:|
| Earlier to later | 0 |
| Later to earlier | 0 |
| Same time | 468,710 |

Because the stored graph is symmetrized, stored reverse entries are message-passing directions, not raw transaction directions. In this artifact all stored relations stay inside the same time step, so symmetrization does not create cross-time future-to-past message paths.

## Receptive-Field Future Exposure

Frozen Elliptic backbones both use `num_layers = 2`, so the model-depth interpretation uses 1-hop and 2-hop neighborhoods. The helper also recorded 3-hop as a supplemental sanity bound.

| Hop definition | Hop | Train nodes checked | Nodes reaching future-period groups | Fraction |
|---|---:|---:|---:|---:|
| Exact frontier | 1 | 26,381 | 0 | 0.0 |
| Exact frontier | 2 | 26,381 | 0 | 0.0 |
| Supplemental exact frontier | 3 | 26,381 | 0 | 0.0 |
| Cumulative within-k | 1 | 26,381 | 0 | 0.0 |
| Cumulative within-k | 2 | 26,381 | 0 | 0.0 |
| Supplemental cumulative within-k | 3 | 26,381 | 0 | 0.0 |

Future groups counted: `VAL_KNOWN`, `TEST_KNOWN`, `UNKNOWN_VAL_PERIOD`, `UNKNOWN_TEST_PERIOD`.

This audit found train-to-unknown same-period edges, but not train-to-future-period receptive-field exposure.

## Backbone Training Graph Trace

| Backbone | Layers | Forward graph | Nodes passed | Edges passed | Loss mask | Validation access | Test access | Cached/self-loop behavior |
|---|---:|---|---:|---:|---|---|---|---|
| GCN | 2 | Full `data.edge_index` | 203,769 | 468,710 | `train_mask` during Optuna trial, then `train_mask OR val_mask` for final checkpoint | Validation labels used for objective/model selection | Test metric is logged/evaluated, but objective source is asserted as validation | Baseline `GCNConv(cached=True)`, PyG default self-loop handling. XAI wrapper uses `cached=False` |
| GraphSAGE | 2 | Full `data.edge_index` | 203,769 | 468,710 | `train_mask` during Optuna trial, then `train_mask OR val_mask` for final checkpoint | Validation labels used for objective/model selection | Test metric is logged/evaluated, but objective source is asserted as validation | `SAGEConv`, cache not applicable, root transform plus neighbor aggregation |

Evidence:

- `D:/CS/KLTN/elliptic-gnn-xai/scripts/train.py` loads `BCDataset`, calls `dataset.get_masks`, converts `dataset.to_torch_data().to(device)`, and passes full `graph.edge_index`.
- `D:/CS/KLTN/elliptic-gnn-xai/utils/objectives.py` computes `model(graph.x, graph.edge_index.to(device))` and applies `criterion(y_hat[train_mask], graph.y[train_mask])`.
- `D:/CS/KLTN/elliptic-gnn-xai/model/GCN.py` uses `GCNConv(..., cached=True)`.
- `D:/CS/KLTN/elliptic-gnn-xai/model/SAGE.py` uses `SAGEConv(..., aggr=aggregator)`.

Train forward includes the full graph tensor, but the frozen graph has no cross-time edges, so train-period message passing does not reach validation/test-period nodes within the model's 2-layer receptive field.

## Edge Symmetrization Provenance

| Question | Answer |
|---|---|
| Raw relationship semantics | Directed Bitcoin payment-flow edges |
| Raw local CSV rows read | 234,355 |
| Raw valid non-self-loop unique directed pairs | 234,355 |
| Raw reverse pairs originally present | 0 |
| Reverse pairs added by preprocessing | 234,355 |
| Stored directed pairs | 468,710 |
| Stored canonical unique relations | 234,355 |
| Stored equals raw plus reverse after de-duplication | `true` |
| Code path | `BCDataset._load_elliptic(directed=False)` in `D:/CS/KLTN/elliptic-gnn-xai/data/dataset.py` |
| Code symbol | `edges_rev`, `pd.concat`, `drop_duplicates` |
| Chronology | `INHERITED_BASELINE_IMPLEMENTATION` |

Rationale: `INFERENCE`. Symmetrization supports undirected neighborhood aggregation for GCN and GraphSAGE.

Benefit: incoming and outgoing transaction neighbors can both contribute to message passing.

Trade-off: raw transaction-direction information is reduced in the stored message-passing graph.

Symmetrization can reverse raw transaction direction as a message-passing direction. In this artifact, however, all relations are same-time, so this does not create later-period-to-earlier-period paths.

## Feature Preprocessing Provenance

| Field | Status | Evidence |
|---|---|---|
| Repository stores `time_step` separately | `REPOSITORY_CONFIRMED` | `BCDataset._load_elliptic` stores `_time_step` |
| Repository drops `time_step` from model input | `REPOSITORY_CONFIRMED` | `BCDataset.to_torch_data` uses `self.features[:, 1:]` before saving from raw |
| Repository fits scaler/imputer | `REPOSITORY_CONFIRMED_FALSE` | no local scaler/imputer fit found in preprocessing path |
| Original feature count including time step | `SOURCE_CONFIRMED` | Elliptic paper/source: 166 node features |
| Local transaction features | `SOURCE_CONFIRMED` | first 94 features include time step and transaction-local information |
| Aggregated neighborhood features | `SOURCE_CONFIRMED` | remaining 72 are one-hop backward/forward aggregates |
| Upstream scaling fit population | `UNKNOWN` | primary/official sources reviewed here do not disclose fit population |
| Whether aggregate features cross time boundaries | `SOURCE_LIMITED`, locally no cross-time graph relations observed | source says no edges connect different time steps, local artifact confirms cross-time relation count 0 |

Repository interpretation: the project uses the benchmark-provided feature table as frozen input and does not fit an additional project-level scaler or imputer.

Forbidden interpretation: do not infer train-only or all-data feature scaling from mean near zero and standard deviation near one.

## Synthetic Ground-Truth Structure

| Dataset | Generator / source | Ground-truth structure |
|---|---|---|
| BA-House | PyG `ExplainerDataset`, `BAGraph`, `HouseMotif` | Global `edge_mask` aligned to 3,398 directed edge entries and `node_mask` over 500 nodes |
| ShapeGGen | Official GraphXAI `ShapeGGen` at commit `a11e65ffbc4df737f35522a8accf2283a8aeaa37` | Target-specific dense `edge_mask`, `candidate_edge_mask`, `node_mask`, and global feature mask |
| Elliptic | Real transaction dataset | No motif-level explanatory-edge ground truth |

## Target-Pool Boundary

Audit02 verifies target file existence, hash, row count, node uniqueness, split membership, and pairwise overlap only.

| Dataset | Pool | Count | Hash | Split membership | Overlap status | Exact selection flow |
|---|---|---:|---|---|---|---|
| BA-House | Final 2x2 | 12 | `2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f` | test | 0 | `PENDING_AUDIT04` |
| BA-House | PG common train | 48 | `05e4574515fb7a03b846adb677295c940eb7af380216c15dfedf65bb9430bc3c` | train | 0 | `PENDING_AUDIT04` |
| BA-House | PG common validation | 16 | `57fc6135495d9a6f8c7b8c1c3babcddd6b76187ebb353ed6fd8c289771980776` | validation | 0 | `PENDING_AUDIT04` |
| ShapeGGen | Final 2x2 | 167 | `bd128dec7810421d592debe8177f3cb46d6c7d9b01fd3f562b8b13971517b41d` | test | 0 | `PENDING_AUDIT04` |
| ShapeGGen | PG common train | 539 | `c05a5be38a64b629c8bb837b1fb3408d8a008726713e70f0318aee3499746b70` | train | 0 | `PENDING_AUDIT04` |
| ShapeGGen | PG common validation | 182 | `b988e40c2af234724b07802ec31d0831ef039a3ef4de3bfee1b80051b49cdc5d` | validation | 0 | `PENDING_AUDIT04` |
| Elliptic | Final common test | 200 | `e414941e6943b06484ecd5028c51daf80ff2bf9f1bbb06e6a5fd816de8dfb9c0` | test | 0 | `PENDING_AUDIT04` |
| Elliptic | PG common train | 2,000 | `36c2f9b3a81336d304556702b02b6524411f308ee75f2152d79101586f9c1a22` | train | 0 | `PENDING_AUDIT04` |
| Elliptic | PG common validation | 500 | `1a747030b9e903acd13d1ad22462cf7e3daf22c4f60fb475e421d2ecd87c8da6` | validation | 0 | `PENDING_AUDIT04` |

Do not use Audit02 to claim that the exact target eligibility/filter/sampling flow has been fully explained.

## Before / After Filtering Counts

| Dataset | Before | After | Reason |
|---|---:|---:|---|
| BA-House dataset nodes | 500 | 500 | no dataset-node filtering |
| BA-House final targets | 100 test nodes | 12 targets | controlled final target subset, exact flow `PENDING_AUDIT04` |
| ShapeGGen dataset nodes | 1,298 | 1,298 | no dataset-node filtering |
| ShapeGGen final targets | 261 test nodes | 167 targets | common eligible target pool, exact flow `PENDING_AUDIT04` |
| Elliptic nodes | 203,769 total | 46,564 known-label supervised nodes | unknown labels excluded from supervision |
| Elliptic final targets | 11,184 known-label test nodes | 200 targets | frozen final XAI pool, exact flow `PENDING_AUDIT04` |

## Valid And Forbidden Interpretation

Valid interpretations:

- BA-House and ShapeGGen use reproducible synthetic splits with ground-truth explanation structures.
- Elliptic uses temporal known-label supervision and preserves unknown nodes as graph context.
- Elliptic future labels are not used for train loss.
- The frozen Elliptic graph has no cross-time relations, so train-period nodes do not reach validation/test-period nodes through the checked message-passing receptive field.
- Edge symmetrization is inherited from the baseline preprocessing and changes raw direction semantics into undirected message-passing semantics.

Forbidden interpretations:

- Do not say Elliptic has explanatory-edge ground truth.
- Do not say the Elliptic setup is strict inductive hidden-future training.
- Do not say target selection is fully explained before Audit04.
- Do not call feature/topology exposure label leakage.
- Do not infer upstream scaling fit scope without source evidence.

## External Sources And Limitations

| Source | Metadata | Claim supported | Limitation |
|---|---|---|---|
| Weber, Domeniconi, Chen, Weidele, Bellei, Robinson, Leiserson, `Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics`, 2019, KDD Anomaly Detection in Finance Workshop, DOI `10.48550/arXiv.1908.02591`, https://arxiv.org/abs/1908.02591 | Accessed 2026-09-03 | Elliptic contains over 200K transaction nodes, 234K directed payment-flow edges, and 166 node features | Does not prove local preprocessing or scaling fit population |
| Bellei, `The Elliptic Data Set: opening up machine learning on the blockchain`, Elliptic, 2019, https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14 | Accessed 2026-09-03 | Dataset has 203,769 nodes, 234,355 edges, 49 time steps, local and one-hop aggregate features, and no edges between different time steps | Official explanatory article, not local implementation evidence |
| Kaggle, `Elliptic Data Set`, https://www.kaggle.com/datasets/ellipticco/elliptic-data-set | Accessed 2026-09-03 | Official dataset distribution and feature description | Browser extraction is limited, and scaling fit population remains undisclosed |

## Severity Classification

| Finding | Severity |
|---|---|
| Split-rationale wording and citations needed expansion | L1 |
| Edge-boundary, time-transition, receptive-field, symmetrization, and deterministic exposure audits were missing | L2 |
| Elliptic upstream raw feature scaling fit population remains undisclosed | DOCUMENTED UNKNOWN |
| Exact target and PG pool eligibility/filter/sampling flow is not fully audited here | PENDING_AUDIT04 |

No L4/L5 blocker was found in Audit02, provided the thesis does not claim strict inductive hidden-future Elliptic training or Elliptic explanatory-edge correctness.

## Stop-Condition Review

| Stop condition | Result |
|---|---|
| Edge-boundary totals fail to reconcile | Not triggered |
| Split masks overlap | Not triggered |
| Unknown labels enter supervised masks | Not triggered |
| Future labels used in train loss/model selection | Not observed |
| Dataset hash mismatch | Not triggered |
| Backbone train path differs from reported path | Not triggered |
| Stored graph cannot be traced to preprocessing | Not triggered |
| Symmetrization contradicts frozen protocol | Not triggered |
| Inductive hidden-future claim contradicts full-graph training | Not triggered because this audit forbids that claim |
| Raw feature source contradicts current feature interpretation | Not triggered |
| Result values would need to change | Not triggered |
| Experiment rerun appears necessary | Not triggered |
| Unexpected user files modified | Not triggered |
| Baseline repository changes caused by this audit | Not triggered |

## Validation Commands

Required validation to run before freeze:

```powershell
.\.venv\Scripts\python.exe -m json.tool docs\audit\dataset_split_preprocessing_inventory.json > $null
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
git diff --check
```

## Final Verdict

AUDIT02 FINAL VERDICT: PASS.

Experiment rerun: NO.

Audit03 started: NO.

Task28 Step04 authorized: NO.

Full Task28 training authorized: NO.

Task29 authorized: NO.

STOP FOR REVIEW.
