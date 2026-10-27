# Elliptic model publication Git receipt - 2026-10-27

**ELLIPTIC_MODEL_PUBLICATION_COMPLETE**

- Branch: `release/thesis-final-20261026`. Verified pre-publication live HEAD: `11e44790f0da4e5cb2e662aac3b2996afd104406`.
- Model/source commit: `1b0eaaeac71c1937150a1680017caad2c16a3486`.
- Final model/catalog/documentation payload commit: `99137a5ee08b8ed6e863285eccb7b8d3596ca20f`. Normal push succeeded and live release HEAD was verified exactly equal to this SHA.
- Independent HTTPS readback at that immutable GitHub commit: **14/14 model files match original SHA256 and size**, verified UTC `2026-10-27T06:14:01.357690+00:00`.
- This report is a subsequent documentation-only attestation of that successful payload push; its own commit cannot contain its own SHA. The final branch receipt SHA is returned by `git log -1` and recorded in the delivery response/local receipt.
- GitHub main remains `6034c13bfab601bef823c07194c70574cd49ec41`. No main merge, force push, reset/rebase or upstream push.

## Validation

- Existing 66 tests plus five publication tests: **71 passed**, two existing Torch deprecation warnings,7.89s. Ruff and new-source AST/import checks passed.
- Clean Git extraction: evidence verifier PASS (248 payload files,217 trace cells, nine unchanged withheld cells),14 checkpoint hash checks PASS, strict weights-only CPU loads PASS (two backbones,twelve PG). No dataset/baseline checkout was required for those model checks; owner dependency environment was reused.
- Original raw-data preprocessing was checked only in an owner temporary folder: seven tensor fingerprints and exact frozen graph SHA matched. No data was copied to the repository. Full real-graph final200 replay and fresh-machine dependency install were not run.
- Exact staging:38 reviewed paths;14 new model binaries,total3,480,122 bytes;0 dataset files;0 new raw GNN masks;0 secret hits; no deletion. Staged/published bytes match the reviewed payload.
- All114 protected prior numerical/result artifacts are byte-identical. Historical seed42 and FRESH5 remain distinct; algorithms/training/results were not changed. Both research checkout HEADs/indexes and original user-modified files are unchanged.
- Dataset basenames, processed graph and old dataset-containing private ZIP are absent from the entire tracked tree. Only the14 exact approved model binaries were added. No COMPLETE archive was uploaded.
- Minimum artifact resolution is75/76: all71 model artifacts plus four synthetic graphs. The one Elliptic graph is intentionally acquired from its provider. Missing published model files:0. Blocked model files:0. No broader full scientific reproducibility claim.

## Independently verified published model files

| Published path | Bytes | SHA256 |
| --- | ---: | --- |
| `artifacts/elliptic/backbones/gcn.pt` | 220129 | `55004ffa1b712ae374fbf630d55f31ac69602d696dbeef8cb150d3f788a9cc07` |
| `artifacts/elliptic/backbones/graphsage.pt` | 305513 | `f25d5cbeb10f9a27dbba1adcb2fdb21b406219956c107f8d51e8ecf5188866bf` |
| `artifacts/elliptic/pgexplainer/fresh5/gcn/seed_0.pt` | 231081 | `193d175a0b49529d9f1a17d17f8b4388cb43c375d90e9d09657afa87c8c69259` |
| `artifacts/elliptic/pgexplainer/fresh5/gcn/seed_1.pt` | 231081 | `cb1d2a786bcb9b4b64be2728c056fd20191226c8d4e81b4890bf29412969d0e6` |
| `artifacts/elliptic/pgexplainer/fresh5/gcn/seed_2.pt` | 231081 | `f8d8f2d51a65568f6e101aaa3d2310ba55453469f4d190a1d9b4aa1b706eed2a` |
| `artifacts/elliptic/pgexplainer/fresh5/graphsage/seed_0.pt` | 261229 | `30d88fd22e9c44e85d51584c4cff582783c5d21c254fa9b483fba9dd49e33734` |
| `artifacts/elliptic/pgexplainer/fresh5/graphsage/seed_1.pt` | 261229 | `99847560d9709200ddffb7988b98a22c967948bac743f0ef9d3f8a93171e46ab` |
| `artifacts/elliptic/pgexplainer/fresh5/graphsage/seed_2.pt` | 261229 | `bc64c8eaee2c64bff32ddd69e0e814b2a6cab4ecfdf78535b935ba4484e73537` |
| `artifacts/elliptic/pgexplainer/fresh5/gcn/seed_3.pt` | 231081 | `4a67e143d4b093161e4ca3e10e6dc8d6b225b826233d7e7f7bdfb9ab41cc5ebb` |
| `artifacts/elliptic/pgexplainer/fresh5/gcn/seed_4.pt` | 231081 | `4faabab5a2ec917bfd12639fd0c2756de6dc0c6332da1e4debfad63b7898db64` |
| `artifacts/elliptic/pgexplainer/fresh5/graphsage/seed_3.pt` | 261229 | `6553f6dfec0e7aa5b5cb7a84066e0d5d5bd1ec00d0d9f471f067bcae888ab519` |
| `artifacts/elliptic/pgexplainer/fresh5/graphsage/seed_4.pt` | 261229 | `54351edf384c0ddf517137faaee8f4198b95f32381faac43a7d8756e0e105fd7` |
| `artifacts/elliptic/pgexplainer/historical_seed42/gcn.zip` | 231423 | `6035f9b343862a2c5fde376ba6160166c61ed916f5b497ee12498fe8726d3988` |
| `artifacts/elliptic/pgexplainer/historical_seed42/graphsage.zip` | 261507 | `ef9e81c399283834aba6bc4d45b72e25e334f00674f1e389ba5003a77eb6e827` |


Original path/archive member, source manifest, backbone/explainer/seed identity, tensor schema, source licenses and preprocessing fingerprints: [publication manifest](../provenance/elliptic_model_publication.json). Dataset acquisition and no-training commands: [Elliptic guide](../reproducibility/ELLIPTIC_MODELS.md). SHA/regression details: [report14](14_ELLIPTIC_MODEL_VERIFICATION.md).
