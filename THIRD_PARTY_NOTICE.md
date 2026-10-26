# Third-party provenance

This release does not assert a new project-wide license. Original benchmark licensing/redistribution rights need maintainer review. Preserve existing license headers and dependency notices.

- Vendored GraphXAI code remains under `src/graphxai/`; original upstream revision and license files are recorded in `docs/provenance/release_manifest.json`. No upstream repository was changed or pushed.
- Elliptic backbone code remains a separate private repository, pinned in the manifest. Its own license/permissions apply.
- PyTorch, PyG and other dependencies retain their respective licenses. Dataset and checkpoint redistribution permissions are not established by installing their code dependencies.

No raw Elliptic dataset, new checkpoint, frozen input ZIP or backup archive was added in this release. Access instructions and checksums are in `docs/provenance/external_assets.json`.
