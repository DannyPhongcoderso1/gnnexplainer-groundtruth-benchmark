# ruff: noqa: E402
"""Discover, hash and restore frozen thesis artifacts without loading tensors.

Default verification is read-only. Restoration requires an explicit destination
and never replaces a file with different bytes. No network request is made.
"""

from __future__ import annotations

# Direct execution and module execution resolve the same repository.
import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [str(_root), str(_root / "src")]


import argparse
import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(root, relative):
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or ":" in relative or "\\" in relative:
        raise ValueError(f"Unsafe artifact path: {relative}")
    root = Path(root).resolve()
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes destination")
    return path


def hash_stream(stream):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hash_stream(stream)


def check_bytes(path, expected_hash, expected_size):
    actual = hash_file(path)
    if actual != (expected_hash, expected_size):
        raise ValueError(f"Frozen file hash/size mismatch: {path}")


def source_digest(source):
    path, member = source
    if member is None:
        return hash_file(path)
    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        return hash_stream(stream)


def discover(entry, root, baseline_root=None, source_root=None, private_root=None):
    candidates = []
    if entry["published_relative_path"]:
        candidates.append((safe_path(root, entry["published_relative_path"]), None))
    if entry["original_repository"] == "elliptic" and baseline_root:
        candidates.append((safe_path(baseline_root, entry["original_relative_path"]), None))
    if source_root and entry["original_repository"] == "benchmark":
        if entry["archive_relative_path"]:
            archive = safe_path(source_root, entry["archive_relative_path"])
            if archive.is_file():
                if hash_file(archive)[0] != entry["archive_sha256"]:
                    raise ValueError(f"Original completion ZIP hash mismatch: {archive}")
                candidates.append((archive, entry["archive_member"]))
        else:
            candidates.append((safe_path(source_root, entry["original_relative_path"]), None))
    if private_root and entry["restored_relative_path"]:
        candidates.append((safe_path(private_root, entry["restored_relative_path"]), None))
    for source in candidates:
        if source[0].is_file():
            if source_digest(source) != (entry["sha256"], entry["size_bytes"]):
                raise ValueError(f"Frozen artifact mismatch: {entry['id']}")
            return source
    return None


def write_verified(source, destination, expected_hash, expected_size):
    """Copy a known source exclusively; leave all pre-existing files untouched."""
    destination = Path(destination)
    if destination.exists():
        check_bytes(destination, expected_hash, expected_size)
        return "ALREADY_MATCHED"
    if source_digest(source) != (expected_hash, expected_size):
        raise ValueError("Source does not match frozen identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    path, member = source
    # xb guarantees that even an unexpected existing file cannot be overwritten.
    with destination.open("xb") as output:
        if member is None:
            with Path(path).open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output)
        else:
            with zipfile.ZipFile(path) as archive, archive.open(member) as input_stream:
                shutil.copyfileobj(input_stream, output)
    check_bytes(destination, expected_hash, expected_size)
    return "RESTORED"


def restore_bundle(catalog, bundle, destination):
    package = catalog["external_packages"][0]
    check_bytes(bundle, package["sha256"], package["size_bytes"])
    restored = []
    with zipfile.ZipFile(bundle) as archive:
        for entry in catalog["required_artifacts"]:
            if entry["category"] != "B":
                continue
            member = entry["restored_relative_path"]
            target = safe_path(destination, member)
            if member not in archive.namelist():
                raise ValueError(f"Missing frozen bundle member: {member}")
            result = write_verified(
                (Path(bundle), member), target, entry["sha256"], entry["size_bytes"]
            )
            restored.append({"id": entry["id"], "status": result})
    return restored


def verify_catalog(catalog, root, baseline_root=None, source_root=None, private_root=None):
    results = []
    for entry in catalog["required_artifacts"]:
        found = discover(entry, root, baseline_root, source_root, private_root)
        status = (
            "VERIFIED"
            if found
            else (
                "EXTERNAL_ACCESS_REQUIRED"
                if entry["category"] == "B"
                else "MISSING_REPOSITORY_ARTIFACT"
            )
        )
        results.append({"id": entry["id"], "category": entry["category"], "status": status})
    return results


def verify_masks(records, root, cache_root=None, private_root=None):
    def one(entry):
        paths = []
        if entry["published_relative_path"]:
            paths.append(safe_path(root, entry["published_relative_path"]))
        if cache_root:
            paths.append(safe_path(cache_root, entry["original_relative_path"]))
        if private_root:
            paths.append(safe_path(private_root, entry["original_relative_path"]))
        for path in paths:
            if path.is_file():
                check_bytes(path, entry["sha256"], entry["size_bytes"])
                return {"id": entry["id"], "status": "VERIFIED"}
        return {"id": entry["id"], "status": "EXTERNAL_ACCESS_REQUIRED"}

    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(one, records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--source-root", type=Path, help="Authorized original benchmark; read-only")
    parser.add_argument("--private-root", type=Path, help="Previously restored private package")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--destination-root", type=Path)
    parser.add_argument("--verify-masks", action="store_true")
    parser.add_argument("--restore-masks", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--mask-id", action="append", default=[])
    parser.add_argument("--dataset")
    args = parser.parse_args()
    catalog = load(args.root / "docs/provenance/external_assets.json")
    output = {}
    if args.restore:
        if not args.bundle or not args.destination_root:
            parser.error("--restore requires --bundle and --destination-root")
        output["restored"] = restore_bundle(catalog, args.bundle, args.destination_root)
    elif args.restore_masks:
        if not args.cache_root or not args.destination_root:
            parser.error("--restore-masks requires --cache-root and --destination-root")
        records = load(args.root / "docs/provenance/gnn_cached_artifacts.json")["records"]
        selected = [
            e
            for e in records
            if (not args.mask_id or e["id"] in args.mask_id)
            and (not args.dataset or e["dataset"] == args.dataset)
        ]
        if args.mask_id and {e["id"] for e in selected} != set(args.mask_id):
            parser.error("Unknown mask id, or dataset filter excludes requested mask")
        output["restored_masks"] = [
            {
                "id": e["id"],
                "status": write_verified(
                    (safe_path(args.cache_root, e["original_relative_path"]), None),
                    safe_path(args.destination_root, e["original_relative_path"]),
                    e["sha256"],
                    e["size_bytes"],
                ),
            }
            for e in selected
        ]
    else:
        results = verify_catalog(
            catalog, args.root, args.baseline_root, args.source_root, args.private_root
        )
        output.update(
            {
                "required": len(results),
                "verified": sum(e["status"] == "VERIFIED" for e in results),
                "artifacts": results,
            }
        )
        if args.verify_masks:
            records = load(args.root / "docs/provenance/gnn_cached_artifacts.json")["records"]
            if args.dataset:
                records = [e for e in records if e["dataset"] == args.dataset]
            masks = verify_masks(records, args.root, args.cache_root, args.private_root)
            output["masks"] = {
                "required": len(masks),
                "verified": sum(e["status"] == "VERIFIED" for e in masks),
            }
        if args.require_complete and (
            output["verified"] != output["required"]
            or "masks" in output
            and output["masks"]["verified"] != output["masks"]["required"]
        ):
            print(json.dumps(output, indent=2))
            return 2
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
