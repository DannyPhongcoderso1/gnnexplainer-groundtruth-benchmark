# Git commits and remote verification — 2026-10-26

Status: **published payload remote equality PASS; scientific reproducibility PARTIAL**.

Branch: https://github.com/DannyPhongcoderso1/gnnexplainer-groundtruth-benchmark/tree/release/thesis-final-20261026

Base/live main unchanged: `6034c13bfab601bef823c07194c70574cd49ec41`.
Published six-group payload HEAD: `5ab5aff57de9a056641223cb5df12e7e0e27bc8c`. `git ls-remote origin refs/heads/release/thesis-final-20261026` returned that exact SHA after the first push.

| Group | Commit | Scope |
|---|---|---|
| 1 | `93e65bcec481905d538b71b6d4b2d058c2b21513` | release: add reviewed core source configs and regression tests |
| 2 | `8fd92c7d7924f14232519fab9630e735ea1b3f1b` | release: preserve BA extension seed42 results stability and runtime |
| 3 | `5d6ef89054b02f299b150c7333378137c96c4f4b` | release: add verified ShapeGGen V1 supplementary projections |
| 4 | `69214a4c65643221252ae7f08af211f475d29472` | release: add Elliptic final200 and FRESH5 saved evidence |
| 5 | `6d98a7e1ace73452bd64d145b75142e447c79ce5` | docs: document release traceability provenance and access limits |
| 6 | `5ab5aff57de9a056641223cb5df12e7e0e27bc8c` | cleanup: archive superseded project plan after verified backup |

Exact staging allowlist: 145 paths; 145 changed paths; 12 individually reviewed ignored paths used `git add -f` with exact path lists. All staged blob bytes matched reviewed working files; `git diff --cached --check` PASS, no new binary/data/archive or blob above25MB. The payload diff/deletion list was reviewed before committing. `git restore --staged --pathspec-from-file` cleared only the isolated release staging between scoped commits; no reset/rebase/merge/force push was used.

Original benchmark HEAD and modified Random script, baseline HEAD and README modification remain unchanged. Baseline live main: `9cd7a02339aa6f08427700b87a1a8e49fbbd195b`; migration pin: `95710a9852019104f90499cdba628c37f4483a2a`. No upstream was changed/pushed, and main was not merged.

Commands executed: `git diff --check`, `git diff --cached --check`, exact `git add`/`git add -f` pathspec files, six scoped `git commit -m` calls, `git push origin release/thesis-final-20261026`, and `git ls-remote` before/after. Payload verifier and original-head/status guards passed before push.

This completed-push report is recorded in a following documentation-only attestation commit. That commit does not change scientific payload hashes. Its final remote equality is recorded in the maintainer's final local report and receipt; resolve current branch HEAD via the branch link. Push success does not establish full reproduction.

Documentation attestation commit `035c4306891ffb14edce1764cf3bed1e5e8e5bb3` was also pushed and live remote equality verified. A subsequent manifest portability check exposed CRLF/LF checkout hash differences in 14 inherited files. This correction preserves every original result Git blob, adds byte-preserving attributes, and records canonical blob hashes. Clean archived-payload verification PASS (149 manifest entries; 150 extracted files). The current correction commit contains only `.gitattributes`, the release manifest, file-manifest CSV, and these two review reports. Final local receipt records the correction's exact remote SHA.
