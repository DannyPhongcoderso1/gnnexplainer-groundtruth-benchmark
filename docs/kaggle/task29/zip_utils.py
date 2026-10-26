"""Hash-addressed ZIP creation and extraction without unsafe or overwritten members."""

import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from common import atomic_json, read_json, require, sha

MANIFEST = "ARCHIVE_MANIFEST.json"


def safe_name(name):
    parts = PurePosixPath(name).parts
    require(
        bool(parts)
        and not name.startswith("/")
        and "\\" not in name
        and ":" not in name
        and all(x not in (".", "..") for x in parts),
        f"Unsafe archive name: {name}",
    )
    return name


def verify_zip(path, expected_sha=None, max_bytes=4 * 1024**3, required=()):
    path = Path(path)
    if expected_sha is not None:
        require(sha(path) == expected_sha, f"Archive SHA256 mismatch: {path.name}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)) and MANIFEST in names, "Duplicate/missing manifest")
        require(sum(i.file_size for i in infos) <= max_bytes, "Archive exceeds extraction budget")
        for info in infos:
            safe_name(info.filename)
            require(
                not info.is_dir()
                and not info.flag_bits & 1
                and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG),
                "Directories, symlinks or encrypted archive members rejected",
            )
        require(archive.getinfo(MANIFEST).file_size < 4 * 1024**2, "Oversized archive manifest")
        manifest = json.loads(archive.read(MANIFEST))
        require(
            set(names) == set(manifest["files"]) | {MANIFEST}, "Unexpected/missing ZIP contents"
        )
        require(set(required) <= set(manifest["files"]), "Required ZIP contents missing")
        require(archive.testzip() is None, "ZIP CRC failure")
        for name, row in manifest["files"].items():
            require(archive.getinfo(name).file_size == row["size_bytes"], "ZIP size mismatch")
            with archive.open(name) as stream:
                require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"],
                    f"ZIP member SHA256 mismatch: {name}",
                )
    return manifest


def create_zip(destination, files, metadata, max_bytes=4 * 1024**3):
    destination = Path(destination)
    require(not destination.exists(), f"Refusing completed ZIP overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, source in files.items():
        safe_name(name)
        source = Path(source)
        require(
            name != MANIFEST and source.is_file() and not source.is_symlink(), "Invalid ZIP source"
        )
        rows[name] = {"sha256": sha(source), "size_bytes": source.stat().st_size}
    require(sum(r["size_bytes"] for r in rows.values()) < max_bytes, "ZIP input budget exceeded")
    require(
        shutil.disk_usage(destination.parent).free
        > sum(r["size_bytes"] for r in rows.values()) + 128 * 1024**2,
        "Insufficient free disk for ZIP",
    )
    manifest = {"schema": 1, "metadata": metadata, "files": rows}
    temporary = destination.with_name(destination.name + ".partial")
    require(not temporary.exists(), "Stale partial ZIP requires diagnosis, not overwrite")
    with zipfile.ZipFile(
        temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=4
    ) as archive:
        for name, source in files.items():
            archive.write(source, name)
        archive.writestr(MANIFEST, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    verify_zip(temporary, max_bytes=max_bytes)
    require(
        all(sha(files[name]) == row["sha256"] for name, row in rows.items()),
        "ZIP sources changed during packaging",
    )
    os.replace(temporary, destination)
    digest = sha(destination)
    atomic_json(
        destination.with_suffix(destination.suffix + ".receipt.json"),
        {
            "status": "ZIP_VERIFIED_LOCAL",
            "path": str(destination),
            "sha256": digest,
            "size_bytes": destination.stat().st_size,
            "external_backup_confirmed": False,
        },
        overwrite=False,
    )
    return digest


def reusable_zip(destination, files, metadata, creator=None):
    """Finish an orphan publication or reuse exact immutable contents, never conflicting bytes."""
    destination = Path(destination)
    rows = {n: {"sha256": sha(p), "size_bytes": Path(p).stat().st_size} for n, p in files.items()}
    require(not destination.is_symlink(), "Symlink publication output rejected")
    if destination.exists():
        manifest = verify_zip(destination)
        require(
            manifest["files"] == rows and manifest["metadata"] == metadata,
            "Existing ZIP conflicts with publication inputs",
        )
        digest = sha(destination)
        receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
        receipt = {
            "status": "ZIP_VERIFIED_LOCAL",
            "path": str(destination),
            "sha256": digest,
            "size_bytes": destination.stat().st_size,
            "external_backup_confirmed": False,
        }
        if receipt_path.exists():
            require(read_json(receipt_path) == receipt, "Existing ZIP receipt conflicts")
        else:
            atomic_json(receipt_path, receipt, overwrite=False)
        return digest
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        from recovery import preserve

        preserve(partial, destination.parent / "publication_forensics")
    orphan_receipt = destination.with_suffix(destination.suffix + ".receipt.json")
    if orphan_receipt.exists():
        from recovery import preserve

        preserve(orphan_receipt, destination.parent / "publication_forensics")
    return (creator or create_zip)(destination, files, metadata)


def extract_verified(path, destination, expected_sha=None, max_bytes=4 * 1024**3):
    manifest = verify_zip(path, expected_sha, max_bytes)
    destination = Path(destination).resolve()
    require(
        not destination.is_symlink() and all(not p.is_symlink() for p in destination.parents),
        "Symlink extraction root",
    )
    with zipfile.ZipFile(path) as archive:
        for name, row in manifest["files"].items():
            target = destination / name
            require(
                destination in target.resolve().parents and not target.is_symlink(),
                "Unsafe extraction target",
            )
            require(all(not p.is_symlink() for p in target.parents), "Symlink extraction ancestor")
            if target.exists():
                require(
                    target.is_file() and sha(target) == row["sha256"],
                    f"Refusing different existing content: {target}",
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            require(sha(target) == row["sha256"], "Extracted content mismatch")
    return manifest
