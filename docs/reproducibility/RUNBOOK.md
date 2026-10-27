# Commands and frozen verification

Run from the repository root with the project installed. Each current command below supports `--help`. Inspect help/configs before an experiment and choose new outputs; no training or full experiment was run in the organization task.

| Purpose | Entry points |
| --- | --- |
| Dataset preparation | scripts/data/generate_ba_house.py; generate_shapeggen.py; preprocess_elliptic.py |
| Target preparation | scripts/data/select_target_nodes.py; select_pgexplainer_targets.py; freeze_ba_house_2x2_targets.py |
| Synthetic backbone training | scripts/train/train_gcn.py; train_graphsage.py; run_shapeggen_backbones_and_targets.py |
| PG training / mixed pipelines | scripts/train/train_pgexplainer_gcn.py; run_shapeggen_pgexplainer.py; run_pgexplainer_gcn_stability.py; run_pgexplainer_graphsage_stability.py |
| Historical Elliptic PG training | scripts/train/run_elliptic_pgexplainer_training.py (original baseline/guard assets required) |
| GNN/Random evaluation | scripts/evaluate/run_batch_gnnexplainer.py; run_shapeggen_gnnexplainer_random.py; run_elliptic_gnnexplainer.py; run_random_baseline.py |
| Frozen PG evaluation | scripts/evaluate/evaluate_pgexplainer_gcn.py; elliptic_frozen_models.py |
| Other accepted evaluation | scripts/evaluate/evaluate_ba_house_model.py; evaluate_gcn.py; evaluate_single_explanation.py; evaluate_validation_budget.py; run_stability.py |
| Verification | scripts/verify/verify_thesis_artifacts.py; verify_release_evidence.py; checkpoints.py; check_environment.py |

BA scale generation/training/verification entry points also appear in their matching scripts/data, train and verify directories. These thin entry points delegate to scripts/extensions, preserving original producer identity and fixed historical output paths. They have not been converted into a new portable training protocol. Elliptic backbone training remains attributed to the original baseline repository; no replacement trainer is invented.

```powershell
python scripts/verify/verify_thesis_artifacts.py
python scripts/verify/verify_release_evidence.py
python scripts/verify/checkpoints.py --check-load
```

Verification resolves 75/76 minimum artifacts, with exactly one intentional provider-acquired Elliptic graph. `--require-complete` still fails until that exact original graph is available through the resolver's existing explicit access options. The Elliptic-specific adapter separately checks seven frozen tensor fingerprints when replaying a locally preprocessed graph. No checks are weakened by restructuring.

[Artifact catalog](../provenance/external_assets.json), [current path mapping](../provenance/repository_paths.json), [results](../../results/README.md) and [Elliptic instructions](ELLIPTIC_MODELS.md) identify exact inputs. Original guard drivers may require owner-local COMPLETE/Release002 archives and raw GNN caches; they remain historical producers, not fresh-clone commands. Current model/hash/numerical-evidence verification has no dependency on the owner's old disk layout.

Former root script locations re-export current implementations; old artifact paths resolve through the explicit path catalog at supported loaders. Original execution JSONs, archived source identities and trace records are never rewritten to pretend that experiments used the new paths.
