# Repository organization: before and after

Source release: `2b29e57784217269dcabc5e4d950468a759034d7`. Work branch: `refactor/repository-organization`. The prior release/main and both original research working trees are preserved.

Before: synthetic classifiers lived directly in models/, Elliptic classes were embedded in a preparation script, active commands/configs were grouped by historical implementation type, model/data paths were spread across artifacts/, accepted results shared results/metrics and docs/extension, and root docs mixed current instructions with historical prose.

After: four canonical backbone modules live under models/synthetic and models/elliptic; the old imports are aliases. GNNExplainer implementation moves to explainers/gnnexplainer.py, accepted PG has a concise API, and the original MIT Elliptic dataset module lives under data/elliptic.py. Current CLI implementations are grouped under scripts/data, train, evaluate and verify. Configs are dataset-first with a shared protocol directory. All71 model identities live under artifacts/checkpoints by dataset; four exact graph blobs are under data/. Accepted saved evidence has dataset-specific result directories and a concise index. Root documentation now starts with the README, protocol map, runbook, artifact access and provenance index.

## Mapped relocations

- checkpoint: 71
- configuration: 43
- data: 1
- data_manifest: 1
- explainer: 1
- frozen_result: 113
- graph: 4
- historical_document: 34
- historical_release_report: 16
- model: 2
- script: 28

Historical import wrappers preserve existing callers. The explicit repository_paths catalog maps former files to current locations; supported loaders and saved-evidence verification resolve old paths without editing immutable execution JSON or the 217-cell thesis trace. Old source identities and hashes remain available in the source release and mapping CSV.

Historical producer scripts with embedded execution/source guards remain under scripts/extensions and at their original TASK29 paths. Category entry points delegate to those exact producers. Raw explanation caches and legacy metric/protocol records remain because they contain unique evidence or runtime dependencies. Combined BA1000/BA1500 CSVs remain combined: scale-specific result indexes explain filtering the original scale column. The external MIT architectures remain parity references and license evidence. Synthetic and Elliptic architectures are not merged.

## Cleanup and recoverability

Every tracked path is classified in RESTRUCTURE_PATH_MAPPING.csv. Unique historical Markdown reports move to provenance/historical_reports; earlier release reviews move to provenance/release_history/20261027. No byte-identical duplicate Markdown report was found. One superseded conversation-style progress summary is ARCHIVE: docs/KLTN_TONG_HOP_TOAN_BO_2026-08-26.md is absent from the new publication but is preserved at the immutable source commit and in the full verified backup. No unique numerical evidence, checkpoint, graph, source receipt or critical producer is discarded. Comments/docstrings retain mathematical, edge/seed/cache, checkpoint/security and copyright rationale; no executable algorithm was shortened or redesigned.

Backup before deletion/relocation: `D:/CS/KLTN/_thesis_release_backups/20261027/pre_repository_organization_2b29e577.zip`; SHA256 `ecfe5cb72cc99adb09e5d6154837f138f6a727c8d5f47d70c0790f1f1590b553`;34,273 files; ZIP CRC/readback PASS. The deleted summary's exact archived bytes matched its recorded SHA256 before removal. Existing root/code references were reviewed; no active source/config dependency requires that summary. It is not published again as an unnecessary backup ZIP.

Scientific limitations remain unchanged: nine BA500 GraphSAGE PG cohort-conflict cells are withheld; Elliptic dataset/raw GNN cache access and original figure builders remain separate limitations. No training, explanation generation or full scientific evaluation is part of this refactor.

Five pre-existing BA500 GraphSAGE natural-cohort Stability checkpoints remain at legacy artifacts/explainers paths as unique cohort-conflict evidence. They are outside the71 final model inventory, are not newly uploaded or relabeled as accepted common-pool results, and their Git blobs remain unchanged.

Exact original Git blobs were additionally read and SHA256-hashed through git cat-file (34,273 paths,21,792 unique blobs). The ZIP preserves the initial Windows checkout and may contain CRLF text conversion; the unchanged source commit preserves exact original Git text bytes. Current relocated result/target/history blobs use the exact Git bytes. Checkpoint and graph bytes match in both representations. The path mapping CSV uses original Git SHA256 values.

REMOVE_FROM_RELEASE: artifacts/data/.gitkeep, artifacts/checkpoints/.gitkeep. All are zero-byte placeholders with no runtime/evidence information; exact empty entries were verified in the backup before deletion. The new data/model folders contain real published files and require no placeholders.

## Initial local inventory (read-only)

| Research checkout | Tracked | Untracked | Ignored | Scientific tracked / untracked / ignored |
| --- | ---: | ---: | ---: | --- |
| benchmark | 34063 | 21789 | 66610 | 33690 / 16376 / 26680 |
| baseline | 38 | 1 | 102 | 12 / 0 / 26 |

Scientific inventory covers model/graph/archive/CSV/JSON/YAML file types outside environment/runtime directories. Full path/size/Git-state inventories are retained owner-locally; their private datasets and ignored files were not copied to the refactor. The release source tree had34,273 tracked paths, each classified in the mapping CSV.
