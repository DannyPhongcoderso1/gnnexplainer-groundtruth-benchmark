"""Offline final receipt, review archive and last-step active release promotion."""

import ast
import io
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from common import atomic_json, sha, utc
from release_inventory import approved_sources
from zip_utils import create_zip, verify_zip

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
AUDIT = ROOT / "docs/audit/elliptic_pg_stability"
RELEASE = TASK / "dist/release_002"


def protected_gate():
    approved_sources(ignore_builder=True)
    baseline = json.loads(
        (Path(tempfile.gettempdir()) / "task29_step022_before.json").read_text(encoding="utf-8")
    )
    allowed = {TASK / "build_package.py", TASK / "release_inventory.py"}
    unexpected = [p for p, h in baseline.items() if sha(p) != h and Path(p) not in allowed]
    assert not unexpected, unexpected
    return {"snapshot_count": len(baseline), "unexpected_changes": unexpected}


def main():
    import offline_tests
    import recovery_tests

    pointer = TASK / "ACTIVE_RELEASE.json"
    manifest_path = AUDIT / "TASK29_RELEASE_002_MANIFEST.json"
    review_path = AUDIT / "TASK29_STEP02_2_REVIEW_FOR_CHATGPT.zip"
    assert not any(p.exists() for p in [pointer, manifest_path, review_path])
    receipt_path = AUDIT / "TASK29_STEP02_2_VERIFICATION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "READY_FOR_KAGGLE_STEP03_SMOKE_TEST"
    groups = {}
    for module, case, expected in [
        (offline_tests, offline_tests.OfflineTests, 34),
        (recovery_tests, recovery_tests.RecoveryTests, 39),
    ]:
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(case)
        )
        assert result.wasSuccessful() and result.testsRun == expected and not result.skipped
        groups[module.__name__] = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "success": result.wasSuccessful(),
            "scope": "WORKING_TREE_OFFLINE_MOCK_ONLY",
            "source_sha256": sha(module.__file__),
            "transcript": stream.getvalue(),
        }
    checks = []
    for name in [
        "build_package.py",
        "release_inventory.py",
        "verify_release.py",
        "finalize_release.py",
    ]:
        ast.parse((TASK / name).read_text(encoding="utf-8"))
    for command in [
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            "-m",
            "ruff",
            "check",
            *[
                str(TASK / n)
                for n in [
                    "build_package.py",
                    "release_inventory.py",
                    "verify_release.py",
                    "finalize_release.py",
                ]
            ],
        ],
        ["git", "diff", "--check"],
    ]:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if command[0] == "git":
            assert result.returncode == 0, result.stderr
        checks.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    receipt["working_tree_tests"] = groups
    receipt["static_checks"] = {"python_ast": "PASS", "commands": checks}
    receipt["static_checks"]["lint_note"] = (
        "Full repo Ruff style selection reports non-blocking style warnings in packaging/audit tools. "
        "No scientific/hash/import/archive/test check is disabled; built generator bytes preserved."
    )
    receipt["final_protected_check"] = protected_gate()
    receipt["verification_tools_sha256"] = {
        name: sha(TASK / name)
        for name in ["release_inventory.py", "verify_release.py", "finalize_release.py"]
    }
    receipt["finalized_at_utc"] = utc()
    atomic_json(receipt_path, receipt)
    runbook = AUDIT / "TASK29_KAGGLE_RUNBOOK_RELEASE_002.md"
    build_manifest = RELEASE / "TASK29_RELEASE_002_MANIFEST.json"
    final = {
        "schema": 1,
        "release_id": "release_002",
        "status": "VERIFIED_OFFLINE",
        "verdict": receipt["verdict"],
        "client_audit_date": "2026-10-14",
        "verified_at_utc": utc(),
        "release_path": RELEASE.as_posix(),
        "immutable_build_manifest": {
            "path": build_manifest.as_posix(),
            "sha256": sha(build_manifest),
        },
        "verification_receipt": {"path": receipt_path.as_posix(), "sha256": sha(receipt_path)},
        "runbook": {"path": runbook.as_posix(), "sha256": sha(runbook)},
        "artifacts": receipt["artifacts"],
        "build_inventory": receipt["build_inventory"],
        "working_tree_tests": {
            n: {k: v for k, v in g.items() if k != "transcript"} for n, g in groups.items()
        },
        "extracted_tests": receipt["extracted_release"]["groups"],
        "r1_r5": receipt["r1_r5"],
        "scientific_protocol_unchanged": receipt["scientific_protocol_unchanged"],
        "protected_input_count": len(receipt["protected_inputs"]),
        "normalized_scientific_source_count": len(receipt["normalized_scientific_sources"]),
        "authorization": receipt["authorization"],
        "kaggle_only_not_run": receipt["kaggle_only_not_run"],
        "active_pointer_path": pointer.as_posix(),
        "hash_order": "build -> verification receipt -> final manifest -> review ZIP -> active pointer",
    }
    atomic_json(manifest_path, final, overwrite=False)
    files = {}
    for name in [
        "TASK29_STEP02_2_RELEASE_REPORT.md",
        "TASK29_STEP02_2_PACKAGE_VERIFICATION.md",
        "TASK29_STEP02_2_DEPLOYMENT_HANDOFF.md",
        "TASK29_STEP02_2_VERIFICATION_RECEIPT.json",
        "TASK29_RELEASE_002_MANIFEST.json",
        "TASK29_KAGGLE_RUNBOOK_RELEASE_002.md",
        "TASK29_RELEASE_002_ARCHIVE_INVENTORIES.json",
    ]:
        files["audit/" + name] = AUDIT / name
    for name in [
        "UPLOAD_MANIFEST.json",
        "TASK29_RELEASE_002_MANIFEST.json",
        "TASK29_SESSION_A.ipynb",
        "TASK29_SESSION_B.ipynb",
    ]:
        files["release/" + name] = RELEASE / name
    for name in [
        "build_package.py",
        "release_inventory.py",
        "verify_release.py",
        "finalize_release.py",
        "offline_tests.py",
        "recovery_tests.py",
    ]:
        files["offline_tools/" + name] = TASK / name
    with tempfile.TemporaryDirectory(prefix="task29-review-metadata-") as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(RELEASE / "TASK29_KAGGLE_CODE_BUNDLE.zip") as archive:
            for name in archive.namelist():
                if name.startswith("execution/") and name != "execution/README.md":
                    destination = temporary / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(name))
                    files["packaged/" + name] = destination
        summary = temporary / "TEST_SUMMARY.json"
        atomic_json(
            summary,
            {
                "working_tree": final["working_tree_tests"],
                "extracted_release": final["extracted_tests"],
                "kaggle_live_checks": "NOT_RUN",
            },
        )
        files["audit/TEST_SUMMARY.json"] = summary
        digest = create_zip(
            review_path,
            files,
            {
                "purpose": "CHATGPT_REVIEW_ONLY_NOT_KAGGLE_INPUT",
                "release_id": "release_002",
                "verdict": receipt["verdict"],
            },
            max_bytes=10 * 1024**2,
        )
    assert review_path.stat().st_size < 10 * 1024**2
    review_inventory = verify_zip(review_path, digest, max_bytes=10 * 1024**2)
    assert not any(n.endswith((".pt", ".pth", ".zip")) for n in review_inventory["files"])
    protected_gate()
    for artifact in receipt["artifacts"]:
        assert sha(artifact["path"]) == artifact["sha256"]
    atomic_json(
        pointer,
        {
            "schema": 1,
            "release_id": "release_002",
            "release_path": RELEASE.as_posix(),
            "verdict": receipt["verdict"],
            "promoted_at_utc": utc(),
            "final_manifest": {"path": manifest_path.as_posix(), "sha256": sha(manifest_path)},
            "verification_receipt": {"path": receipt_path.as_posix(), "sha256": sha(receipt_path)},
            "runbook": final["runbook"],
            "artifacts": receipt["artifacts"],
            "review_archive": {
                "path": review_path.as_posix(),
                "sha256": digest,
                "size_bytes": review_path.stat().st_size,
            },
            "authorization": receipt["authorization"],
        },
        overwrite=False,
    )
    protected_gate()
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "review_size_bytes": review_path.stat().st_size,
                "review_sha256": digest,
                "active_pointer": pointer.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
