"""Optional external export; disabled and credential-free unless explicitly authorized."""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import require, sha


def export_kaggle_dataset(file, settings, work):
    if not settings.get("enabled"):
        return {"status": "NOT_ENABLED", "external_backup_confirmed": False}
    require(settings.get("explicit_upload_authorized") is True, "External upload not authorized")
    dataset = settings.get("existing_private_dataset", "")
    require(
        re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", dataset),
        "Explicit dataset owner/slug required",
    )
    executable = shutil.which("kaggle")
    require(executable is not None, "Optional external Kaggle CLI is unavailable")
    file = Path(file)
    with tempfile.TemporaryDirectory(dir=work, prefix="external-export-") as temporary:
        temporary = Path(temporary)
        upload = temporary / "upload"
        download = temporary / "download"
        upload.mkdir()
        download.mkdir()
        shutil.copyfile(file, upload / file.name)
        require(
            settings.get("private_dataset_confirmed") is True,
            "Private destination dataset must be confirmed before export",
        )
        metadata_path = Path(settings["existing_dataset_metadata_path"])
        metadata = json.loads(metadata_path.read_text())
        require(metadata["id"] == dataset, "Existing dataset metadata identity mismatch")
        shutil.copyfile(metadata_path, upload / "dataset-metadata.json")
        # Never relay external command output; it may contain credential-bearing errors.
        timeout = min(120, int(settings.get("request_timeout_seconds", 120)))
        require(1 <= timeout <= 120, "Invalid external request timeout")
        result = subprocess.run(
            [
                executable,
                "datasets",
                "version",
                "-p",
                str(upload),
                "-m",
                f"Task29 {file.name}",
                "-q",
            ],
            capture_output=True,
            timeout=timeout,
        )
        require(result.returncode == 0, "External upload failed; output redacted")
        result = subprocess.run(
            [
                executable,
                "datasets",
                "download",
                dataset,
                "-f",
                file.name,
                "-p",
                str(download),
                "-q",
                "-o",
            ],
            capture_output=True,
            timeout=timeout,
        )
        require(result.returncode == 0, "External read-back unavailable; backup unconfirmed")
        direct = download / file.name
        if not direct.is_file():
            import zipfile

            archives = list(download.glob("*.zip"))
            require(len(archives) == 1, "External read-back file absent")
            with zipfile.ZipFile(archives[0]) as archive:
                require(archive.namelist() == [file.name], "Unexpected external read-back layout")
                with archive.open(file.name) as source, direct.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        require(sha(direct) == sha(file), "External read-back SHA256 mismatch")
    return {
        "status": "EXTERNAL_UPLOAD_READBACK_VERIFIED",
        "external_backup_confirmed": True,
        "dataset": dataset,
        "filename": file.name,
        "sha256": sha(file),
        "version_pin": "Dataset latest read-back verified now; immutable version ID not attested",
    }
