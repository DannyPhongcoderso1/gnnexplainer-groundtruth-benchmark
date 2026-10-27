# Elliptic XAI Handoff Provenance

## Scope

Task26 prepares Elliptic for final real-world XAI. It does not retrain GCN or GraphSAGE, does not run GNNExplainer, does not run PGExplainer, and does not inspect final explanation results.

## Cross-Repo Provenance

- Baseline repo path: `D:\CS\KLTN\elliptic-gnn-xai`
- Baseline remote:

```text
origin	https://github.com/DannyPhongcoderso1/elliptic-graphfraud-xai.git (fetch)
origin	https://github.com/DannyPhongcoderso1/elliptic-graphfraud-xai.git (push)
```

- Baseline commit: `95710a9852019104f90499cdba628c37f4483a2a`
- Baseline tag at HEAD: `baseline-v1.0`
- Baseline status note: `README.md` has a pre-existing tracked modification and was not modified by Task26.
- XAI benchmark commit before Task26 commit: `4434789c3f099dbe75bb3c7456addf0512e183c5`

## Frozen Inputs

| Artifact | Path | SHA256 |
|---|---|---|
| GCN checkpoint | `D:\CS\KLTN\elliptic-gnn-xai\artifacts\checkpoints\gcn_baseline.pt` | `55004ffa1b712ae374fbf630d55f31ac69602d696dbeef8cb150d3f788a9cc07` |
| GraphSAGE checkpoint | `D:\CS\KLTN\elliptic-gnn-xai\artifacts\checkpoints\graphsage_baseline.pt` | `f25d5cbeb10f9a27dbba1adcb2fdb21b406219956c107f8d51e8ecf5188866bf` |
| GCN config | `D:\CS\KLTN\elliptic-gnn-xai\config\training_baseline_paper_gcn.yaml` | `db271b1485f7f92d6e4a630394fe709af2f77f9e6390b06d8a1bcc65b675a368` |
| GraphSAGE config | `D:\CS\KLTN\elliptic-gnn-xai\config\training_baseline_paper_sage.yaml` | `2118d7836632468ce20cfd1b34a7dc6507199db1356320d71f0f7f0d53fbaf2a` |
| Processed Elliptic data | `D:\CS\KLTN\elliptic-gnn-xai\datasets\elliptic\elliptic_data.pt` | `f807b83a7c739b16f69b5676226ce441b64d5fe8de95506a7727f86e8451291b` |

No silent model-code copying is used. The Task26 script defines explanation-compatible wrappers that match the baseline checkpoint state dicts and records original-graph parity.
