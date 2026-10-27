# External asset access — 2026-10-27

The catalog preserves all 28 legacy entries and adds individually resolvable identities for all 76 minimum artifacts, including the26 previously omitted entries. Categories:61 artifacts directly in Git,15 Elliptic artifacts requiring external access. None are missing from verified owner storage.

The owner retains `final-thesis-elliptic-artifacts-v1.zip` outside Git at `$THESIS_PRIVATE_ARTIFACT_ROOT/20261027/`. Size:47032364 bytes. SHA256:`1dc7b9285e03a0ca20eb79a7aeb0fe08f29e6447b1dc8977c43084c39aa2389b`. The new local directory has inheritance disabled and allows only the current user, SYSTEM and Administrators. The original COMPLETE packages and all research files remain untouched.

This archive has NOT been uploaded. `download_url` is null. Required unresolved action: the maintainer must establish dataset/checkpoint redistribution permission, select an approved controlled distribution service, grant recipient access, upload this exact version and verify its hash/readback. No service URL or public availability is fabricated. Source dependency access and original historical guard archives remain separately required where the driver authenticates them.

Authorized recipients can restore all 15 exact model/data members with:

```powershell
python scripts/verify_thesis_artifacts.py --restore --bundle D:/approved-access/final-thesis-elliptic-artifacts-v1.zip --destination-root D:/thesis-restored
python scripts/verify_thesis_artifacts.py --private-root D:/thesis-restored --require-complete
```

These example paths do not identify an existing public endpoint. This procedure passed against the actual owner bundle in a temporary destination, resolving76/76 with the Git assets. Re-running restoration reused matching files; differing existing files are rejected.

The original historical PG archives are native Torch serialization ZIPs and remain exact bytes. Fresh checkpoint-to-backbone/seed/member mappings are retained in the catalog. The minimal bundle does not substitute for original COMPLETE/Release002 provenance guard archives; all 28 legacy catalog entries remain available for identifying those originals.

For3192 omitted raw GNN caches, obtain the exact authorized frozen cache root from the maintainer, use original_relative_path in `gnn_cached_artifacts.json`, and verify SHA256. All5268 entries resolved and matched in owner storage. Two omitted-cache restorations (BA1000/Elliptic) were tested without tensor loading. No raw cache transfer or online publication was performed. Published compact per-target/pairwise records suffice for saved aggregate checks; raw-score and model-forward replay remains conditional on asset access and trusted loaders.

Owner-local package path (not a public URL): `D:/CS/KLTN/_thesis_release_backups/20261027/final-thesis-elliptic-artifacts-v1.zip`.
