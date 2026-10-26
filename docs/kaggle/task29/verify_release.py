"""Hermetic, guarded release checks. No install, GPU work, deployment or activation."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "docs/kaggle/task29"
AUDIT = ROOT / "docs/audit/elliptic_pg_stability"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def guards(torch):
    return [
        patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("No model forward")),
        patch.object(
            torch.optim.Optimizer,
            "__init__",
            side_effect=AssertionError("No optimizer construction"),
        ),
        patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer step")),
        patch.object(torch.Tensor, "backward", side_effect=AssertionError("No backward")),
        patch.object(torch.autograd, "backward", side_effect=AssertionError("No backward")),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA experiment")),
    ]


def tests(test_source, execution):
    transcript, groups = io.StringIO(), {}
    # Test code is trusted external audit input, not falsely presented as a shipped runtime dependency.
    for name in ("offline_tests", "recovery_tests"):
        module = types.ModuleType(name)
        module.__file__ = str(execution / (name + ".py"))
        sys.modules[name] = module
        exec(
            compile(
                (test_source / (name + ".py")).read_text(encoding="utf-8"), module.__file__, "exec"
            ),
            module.__dict__,
        )
        cls = getattr(module, "OfflineTests" if name == "offline_tests" else "RecoveryTests")
        with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
            result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(cls)
            )
        groups[name] = {
            "count": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "success": result.wasSuccessful(),
            "scope": "MOCK_TESTED_EXTRACTED_SOURCE",
        }
    return groups, transcript.getvalue()


def child(args):
    import torch

    sandbox, test_source = Path(args.sandbox).resolve(), Path(args.test_source).resolve()
    repo = sandbox / "benchmark"
    execution = repo / "docs/kaggle/task29"
    # Remove repository and editable-project search paths. Only packaged code, trusted audit tools, and installed libraries remain.
    sys.path[:] = [
        str(execution),
        str(repo),
        str(repo / "src"),
        str(repo / "scripts"),
        str(test_source),
    ] + [p for p in sys.path if p and Path(p).resolve() != ROOT and Path(p).resolve() != TASK]
    import importlib.abc
    import importlib.util

    protected_roots = {p.stem for p in execution.glob("*.py")} | {
        "gnnexplainer_benchmark",
        "graphxai",
        "ipdb",
        "scripts",
        "prepare_elliptic_task26",
    }

    class PackagedFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] not in protected_roots:
                return None
            for base in (execution, repo / "src", repo, repo / "scripts"):
                candidate = base / fullname.replace(".", "/")
                if candidate.with_suffix(".py").is_file():
                    return importlib.util.spec_from_file_location(
                        fullname, candidate.with_suffix(".py")
                    )
                if (candidate / "__init__.py").is_file():
                    return importlib.util.spec_from_file_location(
                        fullname, candidate / "__init__.py"
                    )
                if candidate.is_dir():
                    spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
                    spec.submodule_search_locations = [str(candidate)]
                    return spec
            raise ImportError(
                "Missing packaged source; original editable checkout fallback prohibited: "
                + fullname
            )

    sys.meta_path.insert(0, PackagedFinder())
    os.environ["ELLIPTIC_BASELINE_REPO"] = str(sandbox / "baseline")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    guard_list = guards(torch)
    with contextlib.ExitStack() as stack:
        for guard in guard_list:
            stack.enter_context(guard)
        import common
        import preflight

        protocol = common.read_json(execution / "protocol.json")
        preflight.file_gate(protocol, benchmark=repo, baseline=sandbox / "baseline")
        imports = []
        for name in protocol["required_imports"] + [p.stem for p in execution.glob("*.py")]:
            module = importlib.import_module(name)
            file = getattr(module, "__file__", None)
            if name.split(".")[0] in protected_roots and file:
                assert Path(file).resolve().is_relative_to(sandbox), (name, file)
            imports.append(
                {
                    "module": name,
                    "file": file,
                    "source": "PACKAGED"
                    if name.split(".")[0] in protected_roots
                    else "INSTALLED_PINNED_ROOT_OR_STDLIB",
                }
            )
        groups, transcript = tests(test_source, execution)
        assert all(g["success"] for g in groups.values()), transcript
        assert groups["offline_tests"]["count"] == 34 and groups["recovery_tests"]["count"] == 39
        # Execute the actual extracted notebook authentication/bootstrap path, with setup and runtime invocation intercepted.
        launches = []
        for session in ("A", "B"):
            nb = json.loads((Path(args.release) / f"TASK29_SESSION_{session}.ipynb").read_text())
            for direct in (False, True):
                work = sandbox / f"simulated_{session}_{direct}/working"
                work.mkdir(parents=True)
                scope = {}
                exec("".join(nb["cells"][1]["source"]), scope)
                scope["CODE_ZIP"] = (
                    str(sandbox / "direct/code")
                    if direct
                    else str(Path(args.release) / "TASK29_KAGGLE_CODE_BUNDLE.zip")
                )
                scope["INPUTS_ZIP"] = (
                    str(sandbox / "direct/inputs")
                    if direct
                    else str(Path(args.release) / "TASK29_ELLIPTIC_FROZEN_INPUTS.zip")
                )
                real_path = Path
                path_factory = lambda value: (
                    work if str(value) == "/kaggle/working" else real_path(value)
                )
                calls = []

                def intercepted(command, **kwargs):
                    calls.append(
                        {
                            "command": command,
                            "check": kwargs.get("check"),
                            "cuda_order": kwargs["env"].get("CUDA_DEVICE_ORDER"),
                            "baseline": kwargs["env"].get("ELLIPTIC_BASELINE_REPO"),
                        }
                    )
                    if command[0] != "bash":
                        assert command[command.index("--mode") + 1] == "PREFLIGHT_ONLY"
                        assert command[command.index("--session") + 1] == session
                        assert not any(
                            x in command for x in ("--production-approved", "--authorize-smoke")
                        )
                    return types.SimpleNamespace(returncode=0)

                # Only substitute Path after authenticated bootstrap source is defined; this is explicitly a mock of absent Kaggle mounts/resources.
                patched_source = "".join(nb["cells"][2]["source"])
                tree = ast.parse(patched_source)
                launch_stmt = tree.body[-1]
                assert isinstance(launch_stmt, ast.Expr) and isinstance(launch_stmt.value, ast.Call)
                exec(
                    compile(
                        ast.Module(body=tree.body[:-1], type_ignores=[]),
                        "authenticated-notebook",
                        "exec",
                    ),
                    scope,
                )
                scope["scope"]["Path"] = path_factory
                with (
                    patch("subprocess.run", side_effect=intercepted),
                    patch(
                        "shutil.disk_usage", return_value=types.SimpleNamespace(free=30 * 1024**3)
                    ),
                ):
                    exec(
                        compile(
                            ast.Module(body=[launch_stmt], type_ignores=[]),
                            "mock-prefight-launch",
                            "exec",
                        ),
                        scope,
                    )
                assert len(calls) == 2
                # Validate the reconstructed writable layout against every frozen/execution hash.
                old_here = preflight.HERE
                preflight.HERE = work / "task29-execution"
                try:
                    preflight.file_gate(
                        protocol,
                        benchmark=work / "task29-benchmark",
                        baseline=work / "elliptic-gnn-xai",
                    )
                finally:
                    preflight.HERE = old_here
                assert not (work / "task29/runs").exists()
                launches.append(
                    {
                        "session": session,
                        "input_mode": "DIRECT" if direct else "ZIP",
                        "status": "MOCK_PREFLIGHT_LAUNCH_PASS",
                        "calls": calls,
                        "dependency_installation_executed": False,
                        "hardware_preflight_executed": False,
                    }
                )
        provenance = {
            name: getattr(module, "__file__", None)
            for name, module in sys.modules.items()
            if name.split(".")[0] in protected_roots
        }
        for name, file in provenance.items():
            if file:
                assert Path(file).resolve().is_relative_to(sandbox), (name, file)
        result = {
            "status": "EXTRACTED_RELEASE_VERIFIED_OFFLINE",
            "groups": groups,
            "tests_transcript": transcript,
            "imports": imports,
            "module_provenance": provenance,
            "launch_simulations": launches,
            "original_repository_source_imports": 0,
            "models_forwarded": 0,
            "optimizer_steps": 0,
            "cuda_experiments": 0,
            "test_harness_policy": "Approved external test source compiled with virtual test __file__ for fixture path resolution; subject runtime modules are physical extracted files verified by origin/hash.",
            "external_audit_source_sha256": {
                n: sha(test_source / n)
                for n in (
                    "offline_tests.py",
                    "recovery_tests.py",
                    "build_package.py",
                    "release_inventory.py",
                )
            },
        }
        write(args.result, result)


def parent(args):
    sys.path.insert(0, str(TASK))
    from common import identity
    from zip_utils import extract_verified, verify_zip

    release = Path(args.release).resolve()
    upload = json.loads((release / "UPLOAD_MANIFEST.json").read_text())
    build = json.loads(
        (release / ("TASK29_" + release.name.upper() + "_MANIFEST.json")).read_text()
    )
    sandbox = Path(tempfile.mkdtemp(prefix="task29-release-roundtrip-"))
    result_path = sandbox / "child_result.json"
    artifacts, inventories, notebooks = [], {}, []
    try:
        for name, row in upload["files"].items():
            path = release / name
            assert sha(path) == row["sha256"] and path.stat().st_size == row["size_bytes"]
            artifacts.append({"name": name, "path": path.as_posix(), **row})
        for kind, name in [
            ("code", "TASK29_KAGGLE_CODE_BUNDLE.zip"),
            ("inputs", "TASK29_ELLIPTIC_FROZEN_INPUTS.zip"),
        ]:
            path = release / name
            manifest = verify_zip(path, upload["files"][name]["sha256"])
            inventories[kind] = manifest
            extract_verified(path, sandbox, upload["files"][name]["sha256"])
            import zipfile

            with zipfile.ZipFile(path) as archive:
                direct = sandbox / "direct" / kind
                for filename in archive.namelist():
                    target = direct / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(filename))
                digest = hashlib.sha256(archive.read("ARCHIVE_MANIFEST.json")).hexdigest()
                assert digest == build["archive_manifest_sha256"][kind]
        execution = sandbox / "benchmark/docs/kaggle/task29"
        execution.mkdir(parents=True)
        for path in (sandbox / "execution").iterdir():
            shutil.copyfile(path, execution / path.name)
        protocol = json.loads((execution / "protocol.json").read_text())
        source_manifest = json.loads((execution / "execution_manifest.json").read_text())
        assert identity(protocol) == upload["protocol_identity"] == build["protocol_identity"]
        assert set(source_manifest["files"]) == {p.name for p in execution.iterdir()} - {
            "execution_manifest.json"
        }
        assert all(sha(execution / n) == h for n, h in source_manifest["files"].items())
        for path in execution.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"))
        for session in ("A", "B"):
            nb = json.loads((release / f"TASK29_SESSION_{session}.ipynb").read_text())
            assert nb["nbformat"] == 4 and nb["nbformat_minor"] == 5
            assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
            values = {}
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    for stmt in ast.parse("".join(cell["source"])).body:
                        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name):
                                    values[target.id] = stmt.value.value
            assert values["SESSION"] == session and values["MODE"] == "PREFLIGHT_ONLY"
            assert values["AUTHORIZE_SMOKE"] is False and values["PRODUCTION_APPROVED"] is False
            assert all(
                values[n] == ""
                for n in [
                    "APPROVED_SMOKE",
                    "APPROVED_SMOKE_SHA256",
                    "RESTORE_ZIP",
                    "RESTORE_SHA256",
                    "EXTERNAL_SETTINGS",
                ]
            )
            assert (
                values["CODE_SHA256"] == upload["files"]["TASK29_KAGGLE_CODE_BUNDLE.zip"]["sha256"]
            )
            assert (
                values["INPUTS_SHA256"]
                == upload["files"]["TASK29_ELLIPTIC_FROZEN_INPUTS.zip"]["sha256"]
            )
            assert values["CODE_MANIFEST_SHA256"] == build["archive_manifest_sha256"]["code"]
            assert values["INPUTS_MANIFEST_SHA256"] == build["archive_manifest_sha256"]["inputs"]
            notebooks.append({"session": session, "status": "VERIFIED_OFFLINE", "values": values})
        harness = sandbox / "audit_harness"
        harness.mkdir()
        for name in [
            "offline_tests.py",
            "recovery_tests.py",
            "build_package.py",
            "release_inventory.py",
        ]:
            shutil.copyfile(TASK / name, harness / name)
        output = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--child",
                "--sandbox",
                str(sandbox),
                "--test-source",
                str(harness),
                "--release",
                str(release),
                "--result",
                str(result_path),
            ],
            cwd=sandbox,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert output.returncode == 0, output.stdout + output.stderr
        child_result = json.loads(result_path.read_text(encoding="utf-8"))
        before = json.loads(
            (Path.home() / "AppData/Local/Temp/task29_step022_before.json").read_text()
        )
        changed = [p for p, h in before.items() if sha(p) != h]
        allowed = {
            str(TASK / "build_package.py").replace("\\", "/"),
            str(TASK / "release_inventory.py").replace("\\", "/"),
        }
        assert not [p for p in changed if p not in allowed], changed
        pf = json.loads((AUDIT / "preflight_manifest.json").read_text(encoding="utf-8"))
        protected = [
            {"path": p, "sha256": h, "matches": sha(p) == h}
            for p, h in pf["protected_input_sha256"].items()
        ]
        assert all(r["matches"] for r in protected)
        normalized = []
        for row in pf["source_hash_checks"]:
            original = Path(row["path"])
            if row["name"] == "pyg_pgexplainer":
                candidate = original
                scope = "INSTALLED_LIBRARY_SOURCE"
            elif original.is_relative_to(ROOT):
                candidate = sandbox / "benchmark" / original.relative_to(ROOT)
                scope = "PACKAGED_SCIENTIFIC_SOURCE"
            else:
                candidate = (
                    sandbox / "baseline" / original.relative_to(ROOT.parent / "elliptic-gnn-xai")
                )
                scope = "PACKAGED_ARCHITECTURE"
            digest = hashlib.sha256(
                candidate.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            ).hexdigest()
            assert digest == row["expected_historical_normalized_sha256"]
            normalized.append(
                {
                    "name": row["name"],
                    "sha256": digest,
                    "scope": scope,
                    "status": "VERIFIED_OFFLINE",
                }
            )
        assert identity({n: sha(execution / n) for n in build["r1_r5_source_sha256"]}) == identity(
            build["r1_r5_source_sha256"]
        )
        receipt = {
            "schema": 1,
            "release_id": release.name,
            "client_audit_date": "2026-10-14",
            "verdict": "READY_FOR_KAGGLE_STEP03_SMOKE_TEST",
            "release_path": release.as_posix(),
            "artifacts": artifacts,
            "build_inventory": build,
            "archive_inventories": inventories,
            "notebooks": notebooks,
            "extracted_release": child_result,
            "protected_inputs": protected,
            "normalized_scientific_sources": normalized,
            "modified_existing_files": changed,
            "before_snapshot_count": len(before),
            "unexpected_changes": [],
            "old_releases_unchanged": True,
            "scientific_protocol_unchanged": build["scientific_protocol_identity"]
            == build["scientific_protocol_identity_before"],
            "r1_r5": {r: "PACKAGED_AND_MOCK_TESTED" for r in ["R1", "R2", "R3", "R4", "R5"]},
            "authorization": {
                "mode": "PREFLIGHT_ONLY",
                "step03_smoke_authorized": False,
                "full_train_authorized": False,
            },
            "kaggle_only_not_run": [
                "Linux exact runtime install",
                "T4x2 UUID/concurrent CUDA updates",
                "CUDA numerical/resume parity",
                "Linux process termination",
                "failed-version output retention/external backup",
                "full30 time/memory/disk overhead",
            ],
            "temporary_roundtrip_root": str(sandbox),
            "temporary_cleanup": "After writing this receipt",
        }
        write(AUDIT / "TASK29_STEP02_2_VERIFICATION_RECEIPT.json", receipt)
        write(AUDIT / "TASK29_RELEASE_002_ARCHIVE_INVENTORIES.json", inventories)
        print(
            json.dumps(
                {
                    "verdict": receipt["verdict"],
                    "extracted_tests": child_result["groups"],
                    "imports": len(child_result["imports"]),
                    "launch_simulations": len(child_result["launch_simulations"]),
                    "artifacts": artifacts,
                },
                indent=2,
            )
        )
        assert sandbox.resolve().parent == Path(
            tempfile.gettempdir()
        ).resolve() and sandbox.name.startswith("task29-release-roundtrip-"), (
            "Unsafe temporary cleanup path"
        )
        shutil.rmtree(sandbox)
    except Exception:
        write(
            AUDIT / "TASK29_STEP02_2_VERIFICATION_RECEIPT.json",
            {"verdict": "BLOCKED_NEEDS_FIX", "preserved_roundtrip_root": str(sandbox)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--release", required=True)
    parser.add_argument("--sandbox")
    parser.add_argument("--test-source")
    parser.add_argument("--result")
    args = parser.parse_args()
    child(args) if args.child else parent(args)
