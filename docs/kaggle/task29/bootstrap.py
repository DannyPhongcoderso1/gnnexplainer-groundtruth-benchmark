"""Reviewed standard-library notebook bootstrap. No import-time setup or training."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import traceback
import zipfile
from pathlib import Path, PurePosixPath


def extract(path, expected, root):
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise RuntimeError("Trusted upload ZIP SHA256 mismatch: " + Path(path).name)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or "ARCHIVE_MANIFEST.json" not in names:
            raise RuntimeError("Duplicate or missing archive manifest")
        if sum(i.file_size for i in infos) > 2 * 1024**3:
            raise RuntimeError("Input bundle extraction budget exceeded")
        for info in infos:
            name = info.filename
            if (
                info.is_dir()
                or name.startswith("/")
                or "\\" in name
                or ":" in name
                or ".." in PurePosixPath(name).parts
                or info.flag_bits & 1
                or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
            ):
                raise RuntimeError("Unsafe input archive member")
        if archive.getinfo("ARCHIVE_MANIFEST.json").file_size > 4 * 1024**2:
            raise RuntimeError("Input manifest too large")
        manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
        if set(names) != set(manifest["files"]) | {"ARCHIVE_MANIFEST.json"} or archive.testzip():
            raise RuntimeError("Input ZIP member/CRC failure")
        for name, expected_file in manifest["files"].items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if (
                actual != expected_file["sha256"]
                or archive.getinfo(name).file_size != expected_file["size_bytes"]
            ):
                raise RuntimeError("Input member hash/size mismatch")
            destination = Path(root) / name
            if any(p.is_symlink() for p in [destination, *destination.parents]):
                raise RuntimeError("Symlink extraction path")
            if destination.exists():
                with destination.open("rb") as stream:
                    same = hashlib.file_digest(stream, "sha256").hexdigest() == actual
                if not same:
                    raise RuntimeError("Refusing existing different extracted file")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
    return manifest


def extract_directory(directory, trusted_manifest_sha256, root):
    directory = Path(directory)
    manifest_path = directory / "ARCHIVE_MANIFEST.json"
    with manifest_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != trusted_manifest_sha256:
            raise RuntimeError("Trusted direct-input manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if files != set(manifest["files"]) | {"ARCHIVE_MANIFEST.json"}:
        raise RuntimeError("Direct input has unexpected/missing files")
    if sum(row["size_bytes"] for row in manifest["files"].values()) > 2 * 1024**3:
        raise RuntimeError("Direct input exceeds budget")
    for name, row in manifest["files"].items():
        if name.startswith("/") or "\\" in name or ":" in name or ".." in PurePosixPath(name).parts:
            raise RuntimeError("Unsafe direct-input path")
        source = directory / name
        target = Path(root) / name
        if any(p.is_symlink() for p in [source, *source.parents, target, *target.parents]):
            raise RuntimeError("Symlink direct input/output")
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != row["sha256"] or source.stat().st_size != row["size_bytes"]:
            raise RuntimeError("Direct member hash/size mismatch")
        if target.exists():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != actual:
                    raise RuntimeError("Refusing different existing staged file")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return manifest


def start(settings):
    if settings["mode"] not in ("PREFLIGHT_ONLY", "SMOKE_TEST", "FULL_TRAIN"):
        raise RuntimeError("Unknown explicit mode")
    if settings["mode"] == "SMOKE_TEST" and settings["authorize_smoke"] is not True:
        raise RuntimeError("Smoke mode is not authorized")
    if settings["mode"] == "FULL_TRAIN" and (
        settings["production_approved"] is not True or len(settings["approved_smoke_sha256"]) != 64
    ):
        raise RuntimeError(
            "Production disabled: deliberate approval and reviewed smoke hash required"
        )
    root = Path("/kaggle/working")
    root.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < 14 * 1024**3:
        raise RuntimeError("Setup disk reserve below14GiB; stop before installation")
    try:
        for name in ("code", "inputs"):
            location = Path(settings[name + "_zip"])
            if location.is_file():
                extract(location, settings[name + "_sha256"], root / "task29-package")
            else:
                if not location.exists() and (location.parent / "ARCHIVE_MANIFEST.json").is_file():
                    location = location.parent
                extract_directory(
                    location, settings[name + "_manifest_sha256"], root / "task29-package"
                )
        stage = root / "task29-package"
        for prefix, destination in [
            ("benchmark", root / "task29-benchmark"),
            ("execution", root / "task29-execution"),
            ("baseline", root / "elliptic-gnn-xai"),
        ]:
            for source in (stage / prefix).rglob("*"):
                if source.is_file():
                    target = destination / source.relative_to(stage / prefix)
                    if any(p.is_symlink() for p in [target, *target.parents]):
                        raise RuntimeError("Symlink destination")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if source.read_bytes() != target.read_bytes():
                            raise RuntimeError("Refusing extracted source/input overwrite")
                    else:
                        shutil.copyfile(source, target)
        execution = root / "task29-execution"
        env = os.environ.copy()
        env.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            MPLBACKEND="Agg",
            PYTHONDONTWRITEBYTECODE="1",
            ELLIPTIC_BASELINE_REPO=str(root / "elliptic-gnn-xai"),
            PYTHONPATH=os.pathsep.join(
                [str(root / "task29-benchmark"), str(root / "task29-benchmark/src"), str(execution)]
            ),
        )
        env.pop("CUDA_VISIBLE_DEVICES", None)
        subprocess.run(["bash", str(execution / "setup.sh")], check=True, env=env)
        command = [
            str(root / "task29-venv/bin/python"),
            "-u",
            str(execution / "orchestrator.py"),
            "--session",
            settings["session"],
            "--mode",
            settings["mode"],
            "--started-unix",
            str(settings["started_unix"]),
        ]
        if settings["authorize_smoke"]:
            command.append("--authorize-smoke")
        if settings["production_approved"]:
            command.append("--production-approved")
        for key, flag in [
            ("approved_smoke", "--approved-smoke"),
            ("approved_smoke_sha256", "--approved-smoke-sha256"),
            ("session_a_receipt", "--session-a-receipt"),
            ("session_a_receipt_sha256", "--session-a-receipt-sha256"),
        ]:
            if settings.get(key):
                command.extend([flag, str(settings[key])])
        if settings.get("restore_zip"):
            command.extend(
                [
                    "--restore-archive",
                    settings["restore_zip"],
                    "--restore-sha256",
                    settings["restore_sha256"],
                ]
            )
        if settings.get("external_settings"):
            command.extend(["--external-settings", settings["external_settings"]])
        result = subprocess.run(command, check=False, env=env)
        (root / "TASK29_PROCESS_EXIT.json").write_text(
            json.dumps(
                {
                    "session": settings["session"],
                    "mode": settings["mode"],
                    "exit_code": result.returncode,
                    "authority": str(
                        root / "task29/sessions" / settings["session"] / "session_receipt.json"
                    ),
                    "output_retention": "Requires actual platform finalization; failed-version retention unverified",
                },
                indent=2,
            )
        )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        return result.returncode
    except Exception:
        diagnostic = root / "TASK29_BOOTSTRAP_FAILURE.json"
        diagnostic.write_text(
            json.dumps(
                {
                    "status": "BOOTSTRAP_FAILED",
                    "traceback": traceback.format_exc(),
                    "training_start_status": "Consult worker logs; not inferred from bootstrap exception",
                },
                indent=2,
            )
        )
        # No credentials are used by this bootstrap. Preserve a readable downloadable failure bundle.
        with zipfile.ZipFile(
            root / "TASK29_BOOTSTRAP_FAILURE.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            archive.write(diagnostic, diagnostic.name)
        raise
