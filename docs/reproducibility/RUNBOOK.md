For current Elliptic model-only publication and provider-acquired data, see [ELLIPTIC_MODELS.md](ELLIPTIC_MODELS.md). Historical guard drivers below retain their original archive expectations.

See [frozen artifact access](ARTIFACTS.md) before any historical pipeline command. Published checkpoints do not require replaying generation/training.

# Reproduction scope

Use the repository root as the working directory. Preserve frozen outputs and work in a separate checkout for new experiments. The release tests exercise saved evidence and selected implementation units. They do not retrain or regenerate scientific results.

## BA-House

The frozen BA500 benchmark uses the existing configs and scripts. `python scripts/run_random_baseline.py --backbone graphsage --help` exposes checkpoint/config/target arguments. The target identity and seed-dependent dispatch are covered by regression tests.

BA extension configs are under `configs/extensions/ba_house_scale/`. The historical generation entry point is `python scripts/extensions/generate_ba_house_scale.py`, followed by `python scripts/extensions/train_ba_house_scale_backbones.py`, target freezing, and the source entry points below. These are **executable pipeline commands**, not help commands; they create data or train models, check original hashes/state, and require the original inputs and phase receipts. No such pipeline command was run during release preparation.

- Targets: `docs/extension/ba_house_scale/phase3_targets/freeze_targets.py`
- GNNExplainer/Random: `docs/extension/ba_house_scale/phase4a_gnn_random/run_xai.py`
- PG primary: `docs/extension/ba_house_scale/phase4b1_pg_primary/run_pg.py`
- PG Stability: `docs/extension/ba_house_scale/phase4b2_pg_stability/run_pg_stability.py`
- Runtime: `docs/extension/ba_house_scale/phase4c_runtime_supplement_20261024/run_runtime.py`

Primary BA1000/1500 point metrics in the thesis are seed42, with 24/36 final targets. Stability uses the separate multi-seed protocol. Later master means must not replace seed42 thesis values. The three-configuration runtime CSV preserves the BA500 historical natural-pool entry; it does not resolve the BA500 mixed-cohort conflict. Natural106/35 and common48/16 remain separately identified; do not combine their metric cells as one accepted run.

## ShapeGGen V1

Official entry points are `scripts/generate_shapeggen.py`, `scripts/run_shapeggen_backbones_and_targets.py`, `scripts/run_shapeggen_gnnexplainer_random.py`, and `scripts/run_shapeggen_pgexplainer.py`. Inspect their CLI help and frozen configs before use. Exact original graph/checkpoints and target identity are required for identity reproduction; rebuilding a graph is a new experiment unless hashes agree.

`results/final/shapeggen_v1/` retains saved mask diagnostics, counterfactual loss records and intervention decisions. `docs/thesis_evidence/shapeggen_v1_pg_gcn_loss_probe/run_loss_probe.py` is historical diagnostic producer code. It has import-time directory creation and must not be imported as a read-only probe. Recomputing gradients is outside release verification.

## Elliptic

Keep the backbone dependency alongside this repository under the name `elliptic-gnn-xai`. Pin `95710a9852019104f90499cdba628c37f4483a2a`; live main at release preflight was `9cd7a02339aa6f08427700b87a1a8e49fbbd195b`, with the same 38 file blobs. The baseline repository is private. Cached origin/main was stale and was not used as live evidence.

GNNExplainer entry: `python scripts/run_elliptic_gnnexplainer.py --config configs/evaluation/elliptic_gnnexplainer.yaml --device cpu`. Requires the frozen processed dataset, target manifests and baseline checkpoints. Do not run before restoring approved assets.

Historical PG full30 uses hard CE/SUM and RAW-logit final200 selection. Its original producer strings are preserved in `docs/provenance/archived_producers/` with original artifact hashes. These are archived text, not a promise of a portable CLI. GCN is diagnostic; GraphSAGE is the validation-supported case study. V2/V3/V4 diagnostic candidates are not official replacements.

FRESH5 evaluates ten independently trained PGExplainer checkpoints (frozen classifier backbones; seeds0–4), on the same 200 targets. `scripts/evaluate_task29_tie_v1.py` needs the exact Release002 archives and completed training packages. `docs/kaggle/task29/` preserves the Kaggle control plane and separate Linux lock. `/kaggle/working` paths are platform contracts. `ACTIVE_RELEASE.json` is a preflight-only historical snapshot, not proof of completion or permission to train. Restore the original archives through the maintainer; verify expected hashes first. The evaluator's historical filenames/receipts are not altered to bypass guards.

## Identity and availability

`docs/provenance/external_assets.json` lists excluded asset hashes and restoration paths. Path tokens in compact derivatives are `$BENCHMARK_ROOT` and `$ELLIPTIC_BASELINE_ROOT`; replace them only when restoring local paths, not in scientific numeric values. Historical original JSON/CSV may retain original project paths. Their hashes and producer identity are provenance, not portable runtime configuration.

`OUTPUT_SHA256.json` and other copied historical integrity receipts refer to the original run. The release manifest covers current release bytes, including explicitly recorded approval-source redactions. Missing archives, guard receipts, original figure builders and the BA500 cohort conflict remain limitations. Do not interpret an archived status string as a newly executed safety gate.
