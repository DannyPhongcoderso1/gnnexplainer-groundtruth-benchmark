# Elliptic data is acquired separately

Download the original CSVs from [Elliptic / Weber et al. (2019)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) under the provider's terms. Baseline source: [Dang et al., RMIT-BDSL](https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection). Raw CSVs, Kaggle archives, processed graphs and dataset-containing backup ZIPs must remain outside Git; the generated local graph is ignored.

```powershell
python scripts/data/preprocess_elliptic.py --raw-dir D:/user-acquired-elliptic --out data/elliptic/elliptic_data.pt
```

Run from the repository root. [The full guide](../../docs/reproducibility/ELLIPTIC_MODELS.md) describes exact temporal splits, feature/label/edge semantics, expected SHA256/tensor fingerprints, model-only checks and no-training replay. This organization task did not regenerate the dataset.
