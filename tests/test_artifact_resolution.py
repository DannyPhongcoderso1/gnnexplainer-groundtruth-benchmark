"""Regression checks for frozen-byte resolution; no Torch import or model loading."""

import hashlib
import zipfile

import pytest

from scripts.verify_thesis_artifacts import (
    discover,
    restore_bundle,
    safe_path,
    verify_catalog,
    write_verified,
)


def entry(data=b"frozen model bytes"):
    return {
        "id": "example.seed42",
        "category": "A",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "published_relative_path": "artifacts/model.pt",
        "original_repository": "benchmark",
        "original_relative_path": "artifacts/model.pt",
        "archive_relative_path": None,
        "restored_relative_path": None,
    }


def test_discovery_rejects_corrupted_bytes(tmp_path):
    path = tmp_path / "artifacts/model.pt"
    path.parent.mkdir()
    path.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="mismatch"):
        discover(entry(), tmp_path)


def test_external_access_is_distinct_from_missing_repository_file(tmp_path):
    local, external = entry(), entry()
    external.update(id="external.seed42", category="B", published_relative_path=None)
    result = verify_catalog({"required_artifacts": [local, external]}, tmp_path)
    assert [x["status"] for x in result] == [
        "MISSING_REPOSITORY_ARTIFACT",
        "EXTERNAL_ACCESS_REQUIRED",
    ]


def test_restore_preserves_existing_different_file(tmp_path):
    src, dst = tmp_path / "source.pt", tmp_path / "existing.pt"
    src.write_bytes(b"frozen model bytes")
    dst.write_bytes(b"user content")
    with pytest.raises(ValueError, match="mismatch"):
        write_verified((src, None), dst, entry()["sha256"], entry()["size_bytes"])
    assert dst.read_bytes() == b"user content"


def test_restore_only_allowlisted_member_and_rejects_traversal(tmp_path):
    data = b"frozen model bytes"
    item = entry(data)
    item.update(
        category="B", published_relative_path=None, restored_relative_path="private/model.pt"
    )
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("private/model.pt", data)
        archive.writestr("../unrelated.pt", b"not allowed")
    catalog = {
        "required_artifacts": [item],
        "external_packages": [
            {
                "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "size_bytes": bundle.stat().st_size,
            }
        ],
    }
    destination = tmp_path / "restored"
    assert restore_bundle(catalog, bundle, destination)[0]["status"] == "RESTORED"
    assert restore_bundle(catalog, bundle, destination)[0]["status"] == "ALREADY_MATCHED"
    assert (destination / "private/model.pt").read_bytes() == data
    assert not (tmp_path / "unrelated.pt").exists()
    for invalid in ["../escape.pt", "/absolute.pt", "C:/escape.pt", "folder\\escape.pt"]:
        with pytest.raises(ValueError):
            safe_path(destination, invalid)


def test_completion_member_hash_is_not_container_hash(tmp_path):
    archive = tmp_path / "completion.zip"
    data = b"frozen model bytes"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("checkpoint.pt", data)
    item = entry(data)
    item.update(
        category="B",
        published_relative_path=None,
        archive_relative_path="completion.zip",
        archive_member="checkpoint.pt",
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    source = discover(item, tmp_path / "clone", source_root=tmp_path)
    assert source == (archive, "checkpoint.pt")
    item["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ZIP hash mismatch"):
        discover(item, tmp_path / "clone", source_root=tmp_path)
