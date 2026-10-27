# Repository organization: Git and remote receipt

**REPOSITORY_RESTRUCTURE_COMPLETE**

Working branch: `refactor/repository-organization`.
Source release: `2b29e57784217269dcabc5e4d950468a759034d7` (live remote verified before work).
Final restructuring content HEAD / verified remote payload HEAD: `e534763e33c6ad584c91a0b9da4af662e4cc3b8d`.

Normal push created only the new refactor branch. Independent HTTPS readback at that immutable commit verified **all71 final checkpoint files and all4 synthetic graph files** against their original SHA256/size (75/75 PASS), UTC `2026-10-27T08:50:17.177721+00:00`.

This file is the subsequent documentation-only attestation of the completed content push. Its own commit cannot contain its own SHA; the final attestation HEAD and exact live remote SHA are supplied by `git log -1`, `git ls-remote` and the delivery response. No source/artifact/result change is part of the attestation.

## Scoped commits

| Group | Commit | Reviewed source/destination paths | Purpose |
| --- | --- | ---: | --- |
| 1 | `4d80b3fa62fd32f5b0406af04b3535611cbe661f` | 15 | refactor: organize synthetic and elliptic model source |
| 2 | `8d66e7e6e2a64e7c0620c5b155eb42fd693f7306` | 110 | refactor: organize data training evaluation and verification commands |
| 3 | `3acf458bd95ba33c6b3be361f893afd357832fcc` | 93 | refactor: group accepted configs and synthetic data by dataset |
| 4 | `ad3272dacc49c233ca0116be4faf9f405cd43e97` | 374 | refactor: organize frozen checkpoints and saved result paths |
| 5 | `dbc79208d73186b29e81a40a53e946cb4d87a1c1` | 18 | docs: simplify thesis navigation and map immutable provenance |
| 6 | `e534763e33c6ad584c91a0b9da4af662e4cc3b8d` | 103 | cleanup: consolidate historical reports and remove verified redundant placeholders |

## Verified final tree

- 713 exact source/destination paths reviewed and staged by allowlists; no git add . / git add -A, force push, reset/rebase, bulk cleanup or merge.
- 337 added Git nodes and285 removed nodes include relocations. The source tree had34,273 files; the resulting tree has34,325. All314 relocation mappings are explicit;32 original source/CLI paths remain as compatibility aliases.
- All33,740 pre-existing result/model/graph Git blobs are identical at their retained or mapped destination paths. All71 final checkpoints, four synthetic graphs, accepted metric/trace records and target identities remain unchanged. Five pre-existing natural-cohort supporting checkpoints remain separately documented legacy evidence, outside the71-final-model count.
- Exact CSV classification:34,292 rows covering every original path plus19 new support entries; KEEP34,239, CONSOLIDATE50, ARCHIVE1, REMOVE_FROM_RELEASE2. The archived conversation summary and two zero-byte placeholders had verified backup copies before removal. No unique scientific evidence was deleted.
- Selected Git bytes matched the reviewed worktree. New binary destinations were limited to the75 exact pre-existing model/graph artifacts. Dataset basename/full-tree checks PASS:0 raw Elliptic CSVs,0 processed Elliptic graph,0 dataset ZIPs,0 COMPLETE packages added. Secret scan:0 hits.
- Clean full Git-tree extraction PASS: all71 model hashes/strict CPU state loads and four graph hashes; artifact discovery75/76 with exactly one intentional external Elliptic graph; saved numerical evidence413 payload files/217 trace cells; canonical and historical model/CLI imports resolve inside the clean extraction. The check explicitly resolves Windows short/long path aliases when confirming the import root.
- Regression76 passed (existing71 plus five path/import tests), two existing Torch warnings,6.44s.33 current CLI and28 legacy aliases pass --help;207 modules compile;43 canonical/targeted modules pass Ruff;79 current documentation links resolve.43 config/target files retain accepted semantics, allowing only current path/class references. Four backbone class ASTs and the complete Elliptic dataset module AST remain equal to source.
- No training, explanation generation, full scientific evaluation or new measurements were performed. Original regression forward checks were retained; the separate model gate performs zero forward/training calls. Clean tests reuse the owner's dependency environment and do not claim a new-machine installation or full result reproduction.

## Protected branches and working trees

- GitHub benchmark main remains `6034c13bfab601bef823c07194c70574cd49ec41`.
- Original release branch remains `2b29e57784217269dcabc5e4d950468a759034d7`.
- Related Elliptic GitHub main remains `9cd7a02339aa6f08427700b87a1a8e49fbbd195b`.
- Original benchmark local HEAD remains `9963bb9f32435a1833d8bafe4e6af7da4db185be`; original baseline local HEAD remains `95710a9852019104f90499cdba628c37f4483a2a`.
- Both original index SHA256 values and user-modified script/README bytes match the initial snapshots. Their tracked/ignored/untracked research inventories were inspected read-only, not copied or cleaned. No upstream RMIT/GraphXAI checkout was changed or pushed.

No new structural blocker remains. Existing scientific limitations are preserved: nine BA500 cohort-conflict cells remain withheld; Elliptic provider data/raw GNN caches and original figure builders remain outside the claimed verification scope. Historical guarded training producers retain their original asset/path constraints; current frozen-model/saved-evidence verification passes without those owner-local dependencies.

[Before/after and backup](RESTRUCTURE_BEFORE_AFTER.md) · [Complete path/classification CSV](RESTRUCTURE_PATH_MAPPING.csv) · [Validation](RESTRUCTURE_VALIDATION.md) · [Current provenance](../provenance/README.md).
