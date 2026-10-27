# Final source cleanup report

Status: **SOURCE_STATIC_AUDIT_AND_CLEANUP_COMPLETE**. Runtime after cleanup: **NOT_RETESTED**. Work branch refactor/repository-organization; source `9233a58d28688dae0419824b60d7e45ee4b67dfe`.

1. external/elliptic_baseline existed as the minimal attributed MIT source/config/preprocessing subset used for the earlier Elliptic publication and a parity-reference unit check. It was not a submodule or a required production classifier implementation.
2. The whole redundant vendor directory is removed. Full MIT notice moved unchanged to docs/provenance/licenses/RMIT_BLOCKCHAIN_ANOMALY_DETECTION_MIT.txt. Two accepted training YAMLs moved byte-identically to configs/elliptic/backbone_training_gcn.yaml and backbone_training_graphsage.yaml. Pinned upstream URL/SHA, original source hashes, copyright and model lineage remain recorded.
3. Removed old model files: src/gnnexplainer_benchmark/models/gcn.py and graphsage.py (forwarders). Canonical models/synthetic/{gcn,graphsage}.py and models/elliptic/{gcn,graphsage}.py remain. models/__init__.py exports the canonical synthetic classes directly. Two audit-local class copies were replaced with imports.
4. Removed27 root script forwarders and the GNN runner alias; current callers now use categorized commands and canonical GNN source. The four wrappers genuinely needed for guarded scale producers remain. No numerical algorithm was shortened or rewritten.
5.37 files removed (34 redundant source wrappers/reference copies plus3 empty result-directory markers);3 unique license/config files relocated.40 old tracked paths disappear. Exact path/SHA/action records are in the removal CSV. No checkpoint, graph, numerical result or frozen target/config byte was deleted or altered.
6. Independent backup: `D:\CS\KLTN\_thesis_release_backups\20261027\source_cleanup_9233a58d.zip`, SHA256 `ce3fcc726cdfecede66834cb0f2f05b8cb973f6db231c5af0d29d968c437a419`,256 exact original Git members; CRC and every member SHA256 verified before deletion. Immutable source commit also remains available.
7. No model architecture rewrite was made. GCN cached=false is an intentional frozen perturbation adaptation; GraphSAGE uses accepted mean on Elliptic and sum on synthetic data. Complete constructor equivalence for unused GraphNorm/init/output modes is not claimed. All four canonical class ASTs match the source commit.
8. Scientific description gaps remain: GNN joint feature-mask scope, synthetic sigmoid versus Elliptic RAW PG score postprocessing, the FRESH5-only tie addendum, exact temperature endpoint interpretation, the less explicit GCN Linear head description, and the existing BA500 natural/common cohort mismatch. The false optimizer-construction wording was corrected in current source docs; no model/training behavior was changed.
9. Runtime behavior after cleanup has **NOT BEEN RETESTED**. No tests, project imports, model/checkpoint loading, inference, optimization, evaluation or data pipeline execution occurred. Earlier76-test and remote artifact receipts belong to earlier source commits and are not current runtime PASS evidence.
10. Static checks:177 Python files parsed,365 owned import paths checked,79 current relative links resolved,75 frozen model/graph SHA256 values checked,4 canonical class ASTs unchanged, existing Elliptic target IDs200/2,000/500 disjoint. No active reference blocker was found in these scopes; unknown historical execution environments remain NOT_VERIFIED. Exact staged Git-byte checks, commits, push status and live remote SHA follow below. Main/original release and both original research working trees must remain unchanged.

[Elliptic parity](ELLIPTIC_ARCHITECTURE_PARITY.md) · [Synthetic parity](SYNTHETIC_ARCHITECTURE_PARITY.md) · [Explainer consistency](EXPLAINER_SOURCE_CONSISTENCY.md) · [Duplication audit](SOURCE_DUPLICATION_AUDIT.md) · [Removal manifest](DUPLICATE_REMOVAL_MANIFEST.csv).

Static path correction: scripts/train/run_shapeggen_pgexplainer.checkpoint_paths now routes its old formatted checkpoint name through the existing exact path resolver. This repairs current frozen-file discovery without changing a model, loss, seed, config or tensor; runtime is NOT_RETESTED.

## Completed static Git gates

- 87 explicit staged paths reviewed;37 source/placeholder removals and3 exact-byte relocations (40 old paths removed).
- All33,738 retained scientific result/model/graph Git blobs and43 original config/target file blobs are unchanged; the only removed result paths are three zero-byte placeholders.
- 425 staged payload SHA256 entries verified directly from Git blobs;0 dataset files added,0 secret hits. All six requested reports exist.
- No project code, tests, CLI help, checkpoint loader, model, optimizer, dataset generator/preprocessor or metric evaluator was executed. All runtime properties after cleanup remain NOT_VERIFIED.
- Original research HEAD/index/user-modified bytes remain unchanged. No upstream checkout or protected branch was modified.
- A stale, zero-byte worktree index.lock older than20 minutes was recoverably moved to the task's temporary directory after confirming no active Git process and exclusive open; no source checkout index was removed.

Publication completed; exact content-commit/live-remote receipt follows below. This report is excluded from the hash manifest's self-referential scope.

## Commit and live remote receipt

| Scope | Commit |
| --- | --- |
| cleanup: remove redundant Elliptic vendor sources and preserve MIT lineage | `9b2de73714130fc3fdceeb3ce1671ffff3c3c65d` |
| cleanup: remove duplicate backbone and explainer forwarding definitions | `af0dbd840fd4c65d74ba73ace254a798b4a46cb8` |
| cleanup: retire obsolete script wrappers and redirect canonical consumers | `b3d24c2424b0732c0541b9a981758ca93b7214d4` |
| docs: record static scientific source parity and cleanup provenance | `5234f2c1decae7209712fee78cf5606e2f13cbf4` |

Normal push succeeded to `refactor/repository-organization`. Final cleanup-content HEAD and verified live remote SHA: `5234f2c1decae7209712fee78cf5606e2f13cbf4`, verified UTC 2026-10-27T11:57:30.003895+00:00. No force push or merge.

This subsequent report-only attestation commit does not change code, configs, artifacts or numerical results. Its own SHA cannot be embedded in itself; the final attestation HEAD and verified live remote SHA are provided in the delivery response and by git log -1 / git ls-remote.

Protected GitHub main remains `6034c13bfab601bef823c07194c70574cd49ec41`; original release remains `2b29e57784217269dcabc5e4d950468a759034d7`; baseline main remains `9cd7a02339aa6f08427700b87a1a8e49fbbd195b`. Both original research HEADs, indexes and user-modified source/README bytes match the initial snapshots. No upstream checkout was modified.

No unresolved active reference blocker was found by static dependency/path/link checks. Historical guarded producers, complete synthetic checkpoint-schema validation, raw CSV alignment/future-structure assumptions and all current runtime behavior remain NOT_VERIFIED. Scientific description gaps and the nine-cell cohort conflict were recorded, not repaired by changing measurements or models.
