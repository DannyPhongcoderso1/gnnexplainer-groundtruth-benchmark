"""Offline-only meaningful control-plane tests; synthetic states and mocks, no models."""

import ast
import copy
import importlib
import json
import math
import random
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import common
from aggregate_stability import calculate
from build_package import notebook
from common import (
    GateError,
    atomic_json,
    child_environment,
    matrix,
    read_json,
    time_allows_seed,
    training_config,
    validate_mode,
    validate_restore_receipt,
    validate_session_a,
    validate_session_config,
)
from export_utils import seed_package, session_package, verify_completed_package
from orchestrator import classify_failure, command_for, simulate
from progress import Observer
from resources import hardware_gate
from zip_utils import MANIFEST, create_zip, extract_verified, verify_zip

HERE = Path(__file__).resolve().parent


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="task29-mock-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, name, text="dummy artifact; not a learned checkpoint"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_01_source_syntax_and_imports_without_execution(self):
        with patch(
            "subprocess.Popen", side_effect=AssertionError("Unexpected subprocess on import")
        ):
            for path in HERE.glob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"))
                if path.name != "offline_tests.py":
                    importlib.import_module(path.stem)

    def test_02_session_matrices_fixed_gpu_and_disjoint(self):
        a, b = matrix("A"), matrix("B")
        self.assertEqual(len(a), 6)
        self.assertEqual(len(b), 4)
        self.assertTrue({r["run_key"] for r in a}.isdisjoint(r["run_key"] for r in b))
        for row in a + b:
            self.assertEqual(row["physical_gpu"], 0 if row["backbone"] == "graphsage" else 1)
        with self.assertRaises(GateError):
            matrix("SESSION_C")

    def test_03_authorization_fail_closed(self):
        validate_mode("PREFLIGHT_ONLY")
        with self.assertRaises(GateError):
            validate_mode("SMOKE_TEST")
        with self.assertRaises(GateError):
            validate_mode("FULL_TRAIN", production_approved=True)
        receipt = self.root / "smoke.json"
        atomic_json(receipt, {"status": "SMOKE_PASS_REVIEW_REQUIRED"})
        validate_mode(
            "FULL_TRAIN",
            production_approved=True,
            approved_smoke_sha256=common.sha(receipt),
            smoke_receipt=receipt,
        )
        with self.assertRaises(GateError):
            validate_mode(
                "FULL_TRAIN",
                production_approved=True,
                approved_smoke_sha256="0" * 64,
                smoke_receipt=receipt,
            )

    def test_04_gpu_binding_command_and_numerical_env_unchanged(self):
        for backbone, gpu in [("graphsage", "0"), ("gcn", "1")]:
            env = child_environment(backbone)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], gpu)
            command = command_for(backbone, 1, "A", "config.json", "approval.json")
            self.assertEqual(command[command.index("--operation") + 1], "train")
            self.assertNotIn("--allow-debug-limits", command)
            self.assertNotIn("--reset-incomplete", command)
            for key in ("OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "MKL_NUM_THREADS"):
                self.assertEqual(env.get(key), common.os.environ.get(key))

    def test_05_two_workers_simulation(self):
        durations = {"graphsage": [9212] * 5, "gcn": [8998] * 5}
        a = simulate("A", durations)
        b = simulate("B", durations)
        self.assertTrue(all(r["status"] == "COMPLETE" for r in a + b))
        self.assertEqual(max(r["finish_seconds"] for r in a), 27636)
        self.assertEqual(max(r["finish_seconds"] for r in b), 18424)

    def test_06_failure_stops_new_seeds_unaffected_current_finishes(self):
        result = simulate(
            "A",
            {"graphsage": [10] * 5, "gcn": [11] * 5},
            fail=("graphsage", 0),
            budget=100,
            margin=0,
            factor=1,
        )
        self.assertEqual(result[0]["status"], "FAILED")
        self.assertEqual(result[1]["status"], "NOT_STARTED")
        gcn = [r for r in result if r["backbone"] == "gcn"]
        self.assertEqual(gcn[0]["status"], "COMPLETE")
        self.assertEqual(gcn[1]["status"], "NOT_STARTED")

    def test_07_time_budget_prevents_next_full_seed(self):
        self.assertFalse(time_allows_seed(3000, 9000, 3600, 1.25))
        result = simulate("A", {"graphsage": [9212] * 5, "gcn": [8998] * 5}, budget=18000)
        self.assertTrue(any(r["status"] == "NOT_STARTED" for r in result))

    def test_08_observer_call_order_result_and_rng_identity(self):
        events, calls = [], []
        observer = Observer("A", "graphsage", 1, events.append, targets=100, interval=50)

        def original(instance, epoch, *args, **kwargs):
            calls.append((instance, epoch, args, kwargs))
            return 0.25

        before = random.getstate()
        wrapped = observer.wrap(original)
        for index in range(100):
            self.assertEqual(wrapped("instance", 11, "model", index=index), 0.25)
        self.assertEqual(len(calls), 100)
        self.assertEqual(random.getstate(), before)
        self.assertEqual([e["target_completed"] for e in events], [50, 100])
        self.assertEqual(events[-1]["optimizer_updates"], 1200)
        source = ast.parse((HERE / "progress.py").read_text())
        names = [ast.unparse(n.func) for n in ast.walk(source) if isinstance(n, ast.Call)]
        self.assertFalse(any("torch" in n or "random" in n or "synchronize" in n for n in names))

    def test_09_nonfinite_or_failed_update_not_acknowledged(self):
        events = []
        observer = Observer("A", "gcn", 0, events.append)
        with self.assertRaises(RuntimeError):
            observer.wrap(lambda *a, **k: float("nan"))(None, 0)
        with self.assertRaises(ValueError):
            observer.wrap(lambda *a, **k: (_ for _ in ()).throw(ValueError("mock update failed")))(
                None, 0
            )
        self.assertEqual(events, [])

    def test_10_atomic_json_and_namespaces(self):
        path = self.root / "progress.json"
        atomic_json(path, {"state": (1, 2, [3])})
        self.assertEqual(read_json(path), {"state": [1, 2, [3]]})
        with self.assertRaises(GateError):
            atomic_json(path, {"different": True}, overwrite=False)
        protocol = {"historical_training_config": {}, "execution_identity": "frozen"}
        directories = {
            str(training_config(protocol, r["backbone"], r["seed"], work=self.root)[0])
            for r in matrix("A") + matrix("B")
        }
        self.assertEqual(len(directories), 10)

    def test_11_zip_roundtrip_and_no_overwrite(self):
        files = {"model/checkpoint.pt": self.file("dummy.pt")}
        archive = self.root / "seed.zip"
        digest = create_zip(archive, files, {"seed": 0})
        self.assertEqual(verify_zip(archive, digest)["metadata"]["seed"], 0)
        extract_verified(archive, self.root / "extract", digest)
        self.assertEqual(
            (self.root / "extract/model/checkpoint.pt").read_text(),
            files["model/checkpoint.pt"].read_text(),
        )
        with self.assertRaises(GateError):
            create_zip(archive, files, {})

    def test_12_archive_hash_tamper_and_budget_rejected(self):
        archive = self.root / "seed.zip"
        create_zip(archive, {"x": self.file("x")}, {})
        with self.assertRaises(GateError):
            verify_zip(archive, "0" * 64)
        with self.assertRaises(GateError):
            verify_zip(archive, max_bytes=1)

    def test_13_unsafe_duplicate_symlink_archives_rejected(self):
        for name, mode in [("../evil", stat.S_IFREG), ("safe", stat.S_IFLNK)]:
            archive = self.root / (str(mode) + ".zip")
            with zipfile.ZipFile(archive, "w") as z:
                info = zipfile.ZipInfo(name)
                info.external_attr = mode << 16
                z.writestr(info, "bad")
                z.writestr(
                    MANIFEST, json.dumps({"files": {name: {"sha256": "0" * 64, "size_bytes": 3}}})
                )
            with self.assertRaises(GateError):
                verify_zip(archive)
        archive = self.root / "duplicate.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("x", "one")
            z.writestr("x", "two")
            z.writestr(MANIFEST, "{}")
        with self.assertRaises(GateError):
            verify_zip(archive)

    def test_14_extraction_cannot_replace_existing_different_file(self):
        archive = self.root / "seed.zip"
        create_zip(archive, {"x": self.file("source")}, {})
        self.file("extract/x", "user content")
        with self.assertRaises(GateError):
            extract_verified(archive, self.root / "extract")
        self.assertEqual((self.root / "extract/x").read_text(), "user content")

    def test_15_complete_seed_zip_and_marker_commit(self):
        directory = self.root / "gcn_seed0"
        paths = common.state_paths(directory, "gcn", 0)
        for path in (
            paths["final"],
            paths["manifest"],
            paths["run"] / "checkpoints/epoch_030.pt",
            paths["run"] / "COMPLETE.json",
            directory / "training.json",
            directory / "runtime.json",
            directory / "durable_progress.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("dummy fixture only")
        verification = {
            "status": "FINAL_VERIFIED",
            "checkpoint_sha256": common.sha(paths["final"]),
            "training_runtime_seconds": 10,
        }
        result = seed_package(
            directory,
            "gcn",
            0,
            verification,
            self.root / "session",
            self.file("protocol.json", "{}"),
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(verify_completed_package(directory)["zip_sha256"], result["zip_sha256"])
        self.assertNotIn("dataset", "\n".join(verify_zip(result["zip_path"])["files"]))
        reused = seed_package(
            directory,
            "gcn",
            0,
            verification,
            self.root / "session",
            self.root / "protocol.json",
        )
        self.assertEqual(reused["zip_sha256"], result["zip_sha256"])

    def test_16_partial_master_contains_recovery_and_retains_seed_zip(self):
        session = self.root / "session"
        session.mkdir()
        dummy = self.file("checkpoint.pt")
        seed_zip = session / "TASK29_GCN_SEED0_COMPLETE.zip"
        create_zip(seed_zip, {"checkpoint.pt": dummy}, {"seed": 0})
        receipt = {"session": "A", "status": "PARTIAL", "protocol_identity": "p"}
        result = session_package(
            session,
            receipt,
            {"gcn/seed_1": {"latest_epoch_state.pt": dummy}},
            self.file("IMPORT.md"),
        )
        contents = verify_zip(result["path"])["files"]
        self.assertIn("recoverable/gcn/seed_1/latest_epoch_state.pt", contents)
        self.assertIn(seed_zip.name, contents)
        self.assertTrue(seed_zip.exists())

    def test_17_session_b_requires_verified_complete_a(self):
        contract = {"driver": "mock"}
        rows = [
            {
                "status": "COMPLETE",
                "zip_sha256": "1" * 64,
                "backbone": row["backbone"],
                "seed": row["seed"],
                "zip_path": f"TASK29_{row['backbone'].upper()}_SEED{row['seed']}_COMPLETE.zip",
            }
            for row in matrix("A")
        ]
        receipt = {
            "session": "A",
            "status": "COMPLETE",
            "matrix": matrix("A"),
            "protocol_identity": "p",
            "runtime_contract": contract,
            "completed_keys": [r["run_key"] for r in matrix("A")],
            "results": rows,
        }
        validate_session_a(receipt, "p", contract)
        for key, value in [
            ("status", "PARTIAL"),
            ("runtime_contract", {"driver": "other"}),
            ("session", "B"),
        ]:
            wrong = {**receipt, key: value}
            with self.assertRaises(GateError):
                validate_session_a(wrong, "p", contract)

    def test_18_resume_is_same_session_same_protocol(self):
        receipt = {
            "session": "A",
            "matrix": matrix("A"),
            "protocol_identity": "p",
            "runtime_contract": {},
        }
        validate_restore_receipt(receipt, "A", "p", {})
        with self.assertRaises(GateError):
            validate_restore_receipt(receipt, "B", "p", {})
        with self.assertRaises(GateError):
            validate_restore_receipt(receipt, "A", "new", {})

    def test_19_realistic_ram_disk_gpu_gates(self):
        data = {
            "gpus": [
                {
                    "index": str(i),
                    "name": "Tesla T4",
                    "uuid": f"GPU-{i}",
                    "driver_version": "mock",
                    "memory.total": "15360",
                }
                for i in range(2)
            ],
            "memory": {"ram_limit_bytes": 30 * common.GIB, "ram_available_bytes": 20 * common.GIB},
            "disk_free_bytes": 50 * common.GIB,
        }
        hardware_gate(data, 12 * common.GIB)
        for key, value in [("gpus", data["gpus"][:1]), ("disk_free_bytes", 1)]:
            with self.assertRaises(GateError):
                hardware_gate({**data, key: value}, 12 * common.GIB)

    def test_20_failure_classification(self):
        self.assertEqual(classify_failure("CUDA out of memory", 1), "RUNTIME_ENVIRONMENT")
        self.assertEqual(classify_failure("hash mismatch", 1), "SCIENTIFIC_INTEGRITY")
        self.assertEqual(classify_failure("", 86), "RECOVERABLE_INTERRUPTION")

    def test_21_notebook_json_defaults_and_locked_session(self):
        for session in ("A", "B"):
            nb = notebook(session, "1" * 64, "2" * 64)
            self.assertEqual(nb["nbformat"], 4)
            self.assertIn('MODE = "PREFLIGHT_ONLY"', "".join(nb["cells"][1]["source"]))
            self.assertIn("PRODUCTION_APPROVED = False", "".join(nb["cells"][1]["source"]))
            self.assertIn(f"assert SESSION == {session!r}", "".join(nb["cells"][2]["source"]))

    def test_22_mock_jaccard_is_target_first_sample_sd(self):
        runs = []
        for seed in range(5):
            records = []
            for node in [123, 456]:
                selected = 0 if node == 456 or seed < 3 else 1
                records.append(
                    {
                        "node_id": node,
                        "candidate_global_edge_indices": [0, 1, 2, 3],
                        "selected_global_edge_indices": [selected],
                        "raw_logits": [10 if i == selected else 0 for i in range(4)],
                        "k": 1,
                    }
                )
            runs.append({"seed": seed, "records": records})
        pairs, targets, summary = calculate(runs, [123, 456])
        self.assertEqual(len(pairs), 20)
        self.assertAlmostEqual(targets[0]["mean_pairwise_jaccard"], 0.4)
        self.assertAlmostEqual(summary["mean"], 0.7)
        self.assertAlmostEqual(summary["sample_sd"], math.sqrt(0.18))
        wrong = copy.deepcopy(runs)
        wrong[-1]["seed"] = 42
        with self.assertRaises(GateError):
            calculate(wrong, [123, 456])

    def test_23_mock_observer_resume_epoch_index(self):
        events = []
        observer = Observer("B", "gcn", 4, events.append, targets=100, interval=50)
        wrapped = observer.wrap(lambda *a, **k: 1.0)
        for _ in range(100):
            wrapped(None, 8)
        self.assertEqual(events[-1]["optimizer_updates"], 900)
        self.assertEqual(events[-1]["target_remaining"], 0)

    def test_24_external_persistence_no_calls_without_authorization(self):
        from persistence import export_kaggle_dataset

        with patch("subprocess.run", side_effect=AssertionError("Network operation prohibited")):
            result = export_kaggle_dataset(self.file("zip"), {"enabled": False}, self.root)
            self.assertFalse(result["external_backup_confirmed"])
            with self.assertRaises(GateError):
                export_kaggle_dataset(self.root / "zip", {"enabled": True}, self.root)

    def test_25_synthetic_adam_schema_without_optimizer_or_forward(self):
        import torch
        from state_verify import check_tensors

        group = {
            "lr": 0.003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": False,
            "params": [0, 1, 2, 3],
        }
        shapes = [(64, 270), (64,), (1, 64), (1,)]
        keys = ["mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"]
        with (
            patch.object(
                torch.nn.Module, "__call__", side_effect=AssertionError("Model forward prohibited")
            ),
            patch.object(
                torch.optim.Optimizer,
                "__init__",
                side_effect=AssertionError("Optimizer construction prohibited"),
            ),
        ):
            state = {k: torch.zeros(s) for k, s in zip(keys, shapes, strict=True)}
            optimizer = {
                "param_groups": [{**group, "betas": (0.9, 0.999)}],
                "state": {
                    i: {
                        "step": torch.tensor(60000.0),
                        "exp_avg": torch.zeros(s),
                        "exp_avg_sq": torch.zeros(s),
                    }
                    for i, s in enumerate(shapes)
                },
            }
            payload = {"algorithm_state_dict": state, "optimizer_state_dict": optimizer}
            check_tensors(payload, "gcn", 60000, group, torch)
            optimizer["state"][0]["step"] = torch.tensor(1.0)
            with self.assertRaises(GateError):
                check_tensors(payload, "gcn", 60000, group, torch)

    def test_26_a_receipt_cannot_duplicate_or_substitute_seed_rows(self):
        rows = [
            {
                "status": "COMPLETE",
                "zip_sha256": "1" * 64,
                "backbone": row["backbone"],
                "seed": row["seed"],
                "zip_path": f"TASK29_{row['backbone'].upper()}_SEED{row['seed']}_COMPLETE.zip",
            }
            for row in matrix("A")
        ]
        receipt = {
            "session": "A",
            "status": "COMPLETE",
            "matrix": matrix("A"),
            "protocol_identity": "p",
            "runtime_contract": {},
            "completed_keys": [r["run_key"] for r in matrix("A")],
            "results": rows,
        }
        receipt["results"][-1] = dict(receipt["results"][0])
        with self.assertRaises(GateError):
            validate_session_a(receipt, "p", {})

    def test_27_synthetic_epoch_state_progress_and_rng_readback(self):
        import numpy as np
        import torch
        from state_verify import verify_epoch

        group = {
            "lr": 0.003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": False,
            "params": [0, 1, 2, 3],
        }
        config = self.root / "training.json"
        atomic_json(config, {"fixture": True})
        shapes = [(64, 309), (64,), (1, 64), (1,)]
        state = {
            "epoch_completed": 1,
            "algorithm_current_epoch": 0,
            "backbone": "GraphSAGE",
            "seed": 0,
            "algorithm_state_dict": dict(
                zip(
                    ["mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"],
                    [torch.zeros(s) for s in shapes],
                    strict=True,
                )
            ),
            "optimizer_state_dict": {
                "param_groups": [group],
                "state": {
                    i: {
                        "step": torch.tensor(2000.0),
                        "exp_avg": torch.zeros(s),
                        "exp_avg_sq": torch.zeros(s),
                    }
                    for i, s in enumerate(shapes)
                },
            },
            "epoch_rows": [
                {
                    "epoch": 1,
                    "seed": 0,
                    "backbone": "GraphSAGE",
                    "num_train_instances": 2000,
                    "mean_loss": 0.1,
                    "median_loss": 0.1,
                    "min_loss": 0.1,
                    "max_loss": 0.1,
                    "runtime_seconds": 1.0,
                }
            ],
            "hashes": {"task_config": common.sha(config), "dataset": "fixture_hash"},
            "rng_state_cpu": torch.zeros(5056, dtype=torch.uint8),
            "rng_state_cuda": [torch.zeros(16, dtype=torch.uint8)],
            "rng_state_numpy": ("MT19937", np.zeros(624, dtype=np.uint32), 0, 0, 0.0),
            "hard_compatibility": {"fixture": True},
            "config_identity": "fixture",
            "runtime_ledger": {},
        }
        path = self.root / "epoch_001.pt"
        torch.save(state, path)
        receipt = verify_epoch(
            path,
            "graphsage",
            0,
            config,
            {"optimizer_group": group, "runner_input_hashes": {"dataset": "fixture_hash"}},
        )
        self.assertEqual(receipt["durable_updates"], 2000)
        del state["rng_state_cuda"]
        torch.save(state, path)
        with self.assertRaises(GateError):
            verify_epoch(
                path,
                "graphsage",
                0,
                config,
                {"optimizer_group": group, "runner_input_hashes": {"dataset": "fixture_hash"}},
            )

    def test_28_session_configuration_rejects_seed_redistribution(self):
        cfg = {
            "schema": 1,
            "session": "A",
            "matrix": matrix("A"),
            "session_limit_seconds": 43200,
            "finalization_margin_seconds": 3600,
            "slowdown_factor": 1.25,
            "resource_interval_seconds": 30,
            "target_heartbeat_interval": 50,
            "heartbeat_seconds": 30,
            "external_persistence": {"enabled": False},
        }
        validate_session_config(cfg)
        cfg["matrix"][0]["physical_gpu"] = 1
        with self.assertRaises(GateError):
            validate_session_config(cfg)

    def test_29_failed_packaging_does_not_publish_complete_marker(self):
        directory = self.root / "incomplete"
        with patch("export_utils.create_zip", side_effect=OSError("mock disk failure")):
            with self.assertRaises(GateError):
                seed_package(
                    directory,
                    "gcn",
                    0,
                    {"status": "NOT_VERIFIED"},
                    self.root,
                    self.file("protocol"),
                )
        self.assertFalse((directory / "TASK29_COMPLETE.json").exists())

    def test_30_syntax_bootstrap_no_provision_for_unapproved_mode(self):
        import bootstrap

        with patch("subprocess.run", side_effect=AssertionError("Provisioning prohibited")):
            with self.assertRaises(RuntimeError):
                bootstrap.start(
                    {
                        "mode": "FULL_TRAIN",
                        "production_approved": False,
                        "approved_smoke_sha256": "",
                    }
                )

    def test_31_direct_auto_extracted_mount_trust_and_tamper(self):
        import bootstrap

        archive = self.root / "input.zip"
        create_zip(
            archive, {"execution/code.py": self.file("code.py", "# reviewed dummy code")}, {}
        )
        direct = self.root / "direct"
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(direct)
        manifest_sha = common.sha(direct / MANIFEST)
        bootstrap.extract_directory(direct, manifest_sha, self.root / "staged")
        self.assertTrue((self.root / "staged/execution/code.py").is_file())
        (direct / "execution/code.py").write_text("tampered")
        with self.assertRaises(RuntimeError):
            bootstrap.extract_directory(direct, manifest_sha, self.root / "tampered-stage")

    def test_32_exact_serialization_catches_finite_tensor_corruption(self):
        import numpy as np
        import torch
        from state_verify import exact_serialized_state

        expected = {
            "model": torch.zeros(4),
            "numpy_rng": np.zeros(624, dtype=np.uint32),
            "settings": (False, None, 2000),
        }
        checkpoint = self.root / "mock.pt"
        torch.save(expected, checkpoint)
        exact_serialized_state(checkpoint, expected)
        changed = {**expected, "model": torch.ones(4)}
        torch.save(changed, checkpoint)
        with self.assertRaises(GateError):
            exact_serialized_state(checkpoint, expected)

    def test_33_restore_partial_archive_to_exact_epoch_namespace(self):
        from restore import restore

        protocol = read_json(HERE / "protocol.json")
        work = self.root / "mock-work"
        directory, config = training_config(protocol, "gcn", 1, work=work)
        target = common.state_paths(directory, "gcn", 1)["run"] / "checkpoints/epoch_007.pt"
        state = self.file("mock_state.pt")
        config_path = self.root / "config.json"
        atomic_json(config_path, config)
        durable_path = self.root / "durable.json"
        atomic_json(
            durable_path, {"path": str(target), "sha256": common.sha(state), "epoch_completed": 7}
        )
        runtime_path = self.root / "runtime.json"
        atomic_json(runtime_path, {"fixture": True})
        receipt_path = self.root / "session_receipt.json"
        contract = {"fixture": "no GPU"}
        atomic_json(
            receipt_path,
            {
                "session": "A",
                "status": "PARTIAL",
                "matrix": matrix("A"),
                "protocol_identity": common.identity(protocol),
                "runtime_contract": contract,
                "results": [],
            },
        )
        archive = self.root / "partial.zip"
        digest = create_zip(
            archive,
            {
                "session_receipt.json": receipt_path,
                "recoverable/gcn/seed_1/latest_epoch_state.pt": state,
                "recoverable/gcn/seed_1/training.json": config_path,
                "recoverable/gcn/seed_1/durable_progress.json": durable_path,
                "recoverable/gcn/seed_1/runtime.json": runtime_path,
            },
            {"fixture": True},
        )
        restore(archive, digest, "A", contract, work=work)
        self.assertEqual(common.sha(target), common.sha(state))
        self.assertFalse((directory / "TASK29_COMPLETE.json").exists())
        with self.assertRaises(GateError):
            restore(archive, digest, "B", contract, work=self.root / "wrong-work")

    def test_34_failed_zip_creation_keeps_seed_incomplete(self):
        directory = self.root / "gcn_seed2"
        paths = common.state_paths(directory, "gcn", 2)
        for path in (
            paths["final"],
            paths["manifest"],
            paths["run"] / "checkpoints/epoch_030.pt",
            paths["run"] / "COMPLETE.json",
            directory / "training.json",
            directory / "runtime.json",
            directory / "durable_progress.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("dummy fixture only")
        verification = {
            "status": "FINAL_VERIFIED",
            "checkpoint_sha256": common.sha(paths["final"]),
            "training_runtime_seconds": 10,
        }
        with patch("export_utils.create_zip", side_effect=OSError("mock disk failure")):
            with self.assertRaises(OSError):
                seed_package(
                    directory, "gcn", 2, verification, self.root / "session", self.file("protocol")
                )
        self.assertFalse((directory / "TASK29_COMPLETE.json").exists())
        self.assertTrue(paths["final"].exists())


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(OfflineTests)
    )
    receipt = {
        "scope": "OFFLINE_MOCK_ONLY",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
        "real_training_runs": 0,
        "model_forwards": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "kaggle_calls": 0,
        "real_final200_extractions": 0,
        "real_pg_stability_computations": 0,
    }
    atomic_json(HERE / "offline_test_results.json", receipt)
    raise SystemExit(0 if result.wasSuccessful() else 1)
