"""Step02.1 fault injection, synthetic CPU state and control-plane tests only."""

import ast
import copy
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import export_utils
import orchestrator
import recovery
import zip_utils
from common import GateError, atomic_json, identity, matrix, read_json, sha, state_paths
import offline_tests
from zip_utils import create_zip, verify_zip


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="task29-fix-mock-")
        self.root = Path(self.tmp.name)
        import torch

        self.guards = [
            patch.object(
                torch.nn.Module, "__call__", side_effect=AssertionError("No model forward")
            ),
            patch.object(
                torch.optim.Optimizer,
                "__init__",
                side_effect=AssertionError("No optimizer construction"),
            ),
            patch.object(
                torch.optim.Adam, "step", side_effect=AssertionError("No optimizer steps")
            ),
            patch.object(torch.Tensor, "backward", side_effect=AssertionError("No backward")),
            patch.object(torch.autograd, "backward", side_effect=AssertionError("No backward")),
            patch.object(
                torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA experiment")
            ),
        ]
        for guard in self.guards:
            guard.start()
            self.addCleanup(guard.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def text(self, path, text="dummy bytes, not a learned checkpoint"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def resume_fixture(self):
        self.directory = self.root / "gcn/seed_0"
        self.config = self.text(self.directory / "training.json", "{}")
        self.protocol = {"fixture": True}
        self.runtime = {"runtime_contract": {"protocol_identity": identity(self.protocol)}}
        self.states = state_paths(self.directory, "gcn", 0)["run"] / "checkpoints"

    def verified(self, epoch):
        path = self.text(self.states / f"epoch_{epoch:03d}.pt")
        row = {
            "status": "EPOCH_VERIFIED",
            "path": str(path),
            "sha256": sha(path),
            "epoch_completed": epoch,
            "durable_updates": epoch * 2000,
            "config_identity": "expected",
            "backbone": "gcn",
            "seed": 0,
            "runtime_contract": self.runtime["runtime_contract"],
        }
        atomic_json(self.directory / f"verified_epoch_{epoch:03d}.json", row)
        return path, row

    def checked(self, path, *args):
        row = read_json(self.directory / f"verified_epoch_{int(path.stem.split('_')[-1]):03d}.json")
        return {k: v for k, v in row.items() if k != "runtime_contract"}

    def gate(self, **kw):
        def check(path):
            require = common.require
            require(
                read_json(
                    self.directory / f"verified_epoch_{int(path.stem.split('_')[-1]):03d}.json"
                )["config_identity"]
                == "expected",
                "Wrong config identity",
            )

        with patch("recovery.verify_epoch", side_effect=self.checked):
            return recovery.resume_gate(
                self.directory, "gcn", 0, self.config, self.protocol, self.runtime, check, **kw
            )

    def test_r1_01_verified11_unverified12_selector_sees11(self):
        self.resume_fixture()
        self.verified(11)
        newer = self.text(self.states / "epoch_012.pt")
        digest = sha(newer)
        result = self.gate()
        self.assertEqual(result["epoch_completed"], 11)
        self.assertFalse(newer.exists())
        self.assertTrue(
            any(
                sha(p) == digest
                for p in (self.directory / "recovery_quarantine").rglob("*")
                if p.is_file()
            )
        )
        source = ast.parse(
            (common.HERE.parents[2] / "scripts/run_elliptic_pgexplainer_training.py").read_text()
        )
        fn = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name == "latest_epoch_state"
        )
        ns = {
            "Path": Path,
            "Any": object,
            "torch": types.SimpleNamespace(
                load=lambda p, **kw: {
                    "config_identity": "expected",
                    "epoch_completed": int(p.stem.split("_")[-1]),
                }
            ),
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "unchanged-selector", "exec"), ns)
        self.assertEqual(ns["latest_epoch_state"](self.states.parent, "expected")[0], 11)
        self.assertEqual(self.gate()["epoch_completed"], 11)

    def test_r1_02_corrupt12_is_preserved_without_load(self):
        self.resume_fixture()
        self.verified(11)
        self.text(self.states / "epoch_012.pt", "corrupt serialization")
        with patch("recovery.verify_epoch", side_effect=self.checked) as verifier:
            result = recovery.resume_gate(
                self.directory, "gcn", 0, self.config, self.protocol, self.runtime, lambda p: None
            )
            self.assertEqual(result["epoch_completed"], 11)
            self.assertTrue(
                all(call.args[0].name == "epoch_011.pt" for call in verifier.call_args_list)
            )

    def test_r1_03_verified12_is_selected(self):
        self.resume_fixture()
        self.verified(11)
        self.verified(12)
        self.assertEqual(self.gate()["epoch_completed"], 12)

    def test_r1_04_wrong_sha_blocks(self):
        self.resume_fixture()
        path, row = self.verified(12)
        path.write_text("modified")
        with self.assertRaises(GateError):
            self.gate()
        self.assertEqual(read_json(self.directory / "recovery_admission.json")["status"], "BLOCKED")

    def test_r1_05_wrong_identity_seed_backbone_runtime_blocks(self):
        for key, value in [
            ("config_identity", "wrong"),
            ("seed", 3),
            ("backbone", "graphsage"),
            ("runtime_contract", {}),
        ]:
            with self.subTest(key=key):
                self.directory = self.root / key / "seed_0"
                self.config = self.text(self.directory / "training.json", "{}")
                self.protocol = {"fixture": True}
                self.runtime = {"runtime_contract": {"protocol_identity": identity(self.protocol)}}
                self.states = state_paths(self.directory, "gcn", 0)["run"] / "checkpoints"
                _, row = self.verified(11)
                row[key] = value
                atomic_json(self.directory / "verified_epoch_011.json", row)
                with self.assertRaises(GateError):
                    self.gate()

    def test_r1_06_no_durable_explicit_policy(self):
        self.resume_fixture()
        self.text(self.states / "epoch_001.pt")
        with self.assertRaises(GateError):
            self.gate()
        with self.assertRaises(GateError):
            self.gate()
        result = self.gate(allow_restart=True)
        self.assertEqual(result["status"], "RESTART_EXPLICITLY_APPROVED")
        self.assertEqual(result["durable_updates"], 0)
        self.assertTrue(result["rejected"])

    def test_r1_07_corrupt_receipt_and_attempt_logs_block(self):
        self.resume_fixture()
        self.verified(11)
        (self.directory / "verified_epoch_011.json").write_text("{bad")
        with self.assertRaises(GateError):
            self.gate()
        self.directory = self.root / "no-epoch"
        self.config = self.text(self.directory / "training.json", "{}")
        self.states = state_paths(self.directory, "gcn", 0)["run"] / "checkpoints"
        self.text(
            self.directory / "stdout.log", "TASK29_EVENT previous attempt worker-start marker"
        )
        with self.assertRaises(GateError):
            self.gate()

    def test_r1_08_real_synthetic_schema_rejection_no_models(self):
        # Existing synthetic epoch test exercises real CPU deserialization, Adam/RNG schema and wrong update count.
        case = offline_tests.OfflineTests("test_27_synthetic_epoch_state_progress_and_rng_readback")
        result = unittest.TestResult()
        case.run(result)
        self.assertEqual(result.failures + result.errors, [])
        self.resume_fixture()
        self.verified(11)
        with patch(
            "recovery.verify_epoch", side_effect=GateError("synthetic schema/update mismatch")
        ):
            with self.assertRaises(GateError):
                recovery.resume_gate(
                    self.directory,
                    "gcn",
                    0,
                    self.config,
                    self.protocol,
                    self.runtime,
                    lambda p: None,
                )

    def seed_fixture(self, seed=0):
        directory = self.root / f"seed_{seed}"
        paths = state_paths(directory, "gcn", seed)
        for path in [
            paths["final"],
            paths["manifest"],
            paths["run"] / "checkpoints/epoch_030.pt",
            paths["run"] / "COMPLETE.json",
            directory / "training.json",
            directory / "runtime.json",
            directory / "durable_progress.json",
        ]:
            self.text(path)
        protocol = self.text(self.root / "protocol.json", "{}")
        verification = {
            "status": "FINAL_VERIFIED",
            "checkpoint_sha256": sha(paths["final"]),
            "training_runtime_seconds": 10,
            "epoch_state": {"sha256": sha(paths["run"] / "checkpoints/epoch_030.pt")},
        }
        return directory, paths, verification, protocol

    def publish(self, fixture):
        directory, _, verification, protocol = fixture
        return export_utils.seed_package(
            directory, "gcn", 0, verification, self.root / "session", protocol
        )

    def fault_retry(self, target, callback):
        fixture = self.seed_fixture()
        before = sha(fixture[1]["final"])
        with patch(target, side_effect=callback):
            with self.assertRaises(OSError):
                self.publish(fixture)
        self.assertFalse((fixture[0] / "TASK29_COMPLETE.json").exists())
        result = self.publish(fixture)
        self.assertEqual(sha(fixture[1]["final"]), before)
        verify_zip(result["zip_path"], result["zip_sha256"])
        self.assertEqual(
            read_json(fixture[0] / "publication_transaction.json")["status"], "COMPLETE"
        )
        self.assertEqual(self.publish(fixture)["zip_sha256"], result["zip_sha256"])

    def test_r2_01_metadata_creation_retry(self):
        original = export_utils.atomic_json

        def fault(path, *args, **kw):
            if Path(path).name == "completion_manifest.json":
                raise OSError("metadata injection")
            return original(path, *args, **kw)

        self.fault_retry("export_utils.atomic_json", fault)

    def test_r2_02_zip_staging_retry(self):
        self.fault_retry(
            "export_utils.create_zip",
            lambda *a, **k: (_ for _ in ()).throw(OSError("staging injection")),
        )

    def test_r2_03_zip_write_preserves_partial_retry(self):
        def fault(path, *a, **k):
            self.text(Path(path).with_name(Path(path).name + ".partial"), "partial writer bytes")
            raise OSError("write injection")

        self.fault_retry("export_utils.create_zip", fault)
        self.assertTrue(list((self.root / "session/publication_forensics").glob("*")))

    def test_r2_04_zip_verification_retry(self):
        self.fault_retry(
            "zip_utils.verify_zip",
            lambda *a, **k: (_ for _ in ()).throw(OSError("verification injection")),
        )

    def test_r2_05_receipt_creation_orphan_zip_retry(self):
        original = zip_utils.atomic_json

        def fault(path, *args, **kw):
            if str(path).endswith(".zip.receipt.json"):
                raise OSError("receipt injection")
            return original(path, *args, **kw)

        self.fault_retry("zip_utils.atomic_json", fault)

    def test_r2_06_marker_commit_retry(self):
        original = export_utils.atomic_json

        def fault(path, *args, **kw):
            if Path(path).name == "TASK29_COMPLETE.json":
                raise OSError("marker injection")
            return original(path, *args, **kw)

        self.fault_retry("export_utils.atomic_json", fault)

    def test_r2_07_conflict_never_overwrites_complete(self):
        fixture = self.seed_fixture()
        result = self.publish(fixture)
        before = sha(result["zip_path"])
        changed = copy.deepcopy(fixture[2])
        changed["training_runtime_seconds"] = 999
        with self.assertRaises(GateError):
            self.publish((fixture[0], fixture[1], changed, fixture[3]))
        self.assertEqual(sha(result["zip_path"]), before)
        self.assertEqual(read_json(fixture[0] / "TASK29_COMPLETE.json")["zip_sha256"], before)
        self.assertEqual(
            read_json(fixture[0] / "publication_transaction.json")["status"], "FAILED_INTEGRITY"
        )

    def test_r2_08_corrupt_existing_zip_rejected(self):
        fixture = self.seed_fixture()
        archive = self.text(self.root / "session/TASK29_GCN_SEED0_COMPLETE.zip", "corrupt ZIP")
        with self.assertRaises(Exception):
            self.publish(fixture)
        self.assertEqual(archive.read_text(), "corrupt ZIP")
        self.assertFalse((fixture[0] / "TASK29_COMPLETE.json").exists())
        self.assertEqual(
            read_json(fixture[0] / "publication_transaction.json")["status"], "FAILED_INTEGRITY"
        )

    def master(self, receipt):
        return export_utils.session_package(
            self.root / "session", receipt, {}, self.text(self.root / "guide.md", "guide")
        )

    def test_r3_01_exact_repeat_preserves_canonical(self):
        receipt = {"session": "A", "status": "PARTIAL", "protocol_identity": "fixture"}
        first = self.master(receipt)
        second = self.master(receipt)
        self.assertEqual(first, second)
        self.assertEqual(sha(first["path"]), first["sha256"])

    def test_r3_02_improved_partial_and_complete_attempts(self):
        receipt = {"session": "A", "status": "PARTIAL", "protocol_identity": "fixture"}
        first = self.master(receipt)
        receipt["progress"] = "improved"
        second = self.master(receipt)
        self.assertNotEqual(first["path"], second["path"])
        self.assertEqual(sha(first["path"]), first["sha256"])
        rows = []
        for row in matrix("A"):
            archive = (
                self.root
                / "session"
                / f"TASK29_{row['backbone'].upper()}_SEED{row['seed']}_COMPLETE.zip"
            )
            digest = create_zip(
                archive,
                {"checkpoint.pt": self.text(self.root / row["run_key"] / "dummy")},
                {"backbone": row["backbone"], "seed": row["seed"]},
            )
            rows.append(
                {**row, "status": "COMPLETE", "zip_path": str(archive), "zip_sha256": digest}
            )
        receipt.update(
            status="COMPLETE", results=rows, completed_keys=[r["run_key"] for r in matrix("A")]
        )
        third = self.master(receipt)
        self.assertNotEqual(second["path"], third["path"])
        self.assertEqual(
            read_json(self.root / "session/current_master.json")["path"], third["path"]
        )
        self.assertEqual(sha(first["path"]), first["sha256"])
        self.assertEqual(sha(second["path"]), second["sha256"])
        for row in rows:
            verify_zip(row["zip_path"], row["zip_sha256"])

    def test_r3_03_wrong_session_rejected(self):
        self.master({"session": "A", "status": "PARTIAL", "protocol_identity": "fixture"})
        with self.assertRaises(GateError):
            self.master({"session": "B", "status": "PARTIAL", "protocol_identity": "fixture"})

    def test_r3_04_nested_corruption_and_false_complete_rejected(self):
        self.text(self.root / "session/TASK29_GCN_SEED0_COMPLETE.zip", "corrupt")
        with self.assertRaises(Exception):
            self.master({"session": "A", "status": "PARTIAL", "protocol_identity": "fixture"})
        with self.assertRaises(GateError):
            self.master(
                {
                    "session": "A",
                    "status": "COMPLETE",
                    "protocol_identity": "fixture",
                    "results": [],
                    "completed_keys": [],
                }
            )

    def controller(self):
        with patch("orchestrator.WORK", self.root):
            return orchestrator.Controller(
                read_json(common.HERE / "SESSION_A.json"),
                {"runtime_contract": {}},
                self.root / "approval",
                time.time(),
            )

    def test_r4_01_complete_terminal_snapshot(self):
        c = self.controller()
        c.results = [{**r, "training_runtime_seconds": 1} for r in matrix("A")]
        for r in c.status.values():
            r.update(status="COMPLETE", last_committed_epoch=30)
        with patch("orchestrator.session_package", return_value={"status": "mock verified"}):
            receipt = c.finish()
        view = read_json(c.directory / "progress_status.json")
        self.assertEqual(view["status"], "COMPLETE")
        self.assertEqual(len(view["completed_keys"]), 6)
        self.assertEqual(read_json(view["receipt_path"]), receipt)

    def test_r4_02_failure_late_events_cannot_regress(self):
        c = self.controller()
        key = "gcn/seed_0"
        c.status[key].update(status="TRAINING", last_observed_target=850, last_committed_epoch=11)
        c.failure("gcn", 0, "WORKER_RUNTIME_FAILURE", "mock crash")
        c.event({"event": "WORKER_STARTED", "backbone": "gcn", "seed": 0, "status": "TRAINING"})
        with patch("orchestrator.session_package", return_value={}):
            c.finish()
        view = read_json(c.directory / "progress_status.json")
        self.assertEqual(view["runs"][key]["status"], "FAILED")
        self.assertEqual(view["runs"][key]["last_committed_epoch"], 11)
        self.assertEqual(view["runs"][key]["last_observed_target"], 850)
        self.assertIn(key, view["failed_keys"])

    def test_r4_03_timeout_with_completed_seed(self):
        c = self.controller()
        c.results = [{"backbone": "gcn", "seed": 0, "training_runtime_seconds": 1}]
        c.status["gcn/seed_0"].update(status="COMPLETE", last_committed_epoch=30)
        c.terminal_reason = "time budget exhausted"
        with patch("orchestrator.session_package", return_value={}):
            c.finish()
        view = read_json(c.directory / "progress_status.json")
        self.assertEqual(view["status"], "PARTIAL")
        self.assertEqual(view["runs"]["gcn/seed_1"]["status"], "PARTIAL")
        self.assertEqual(view["completed_keys"], ["gcn/seed_0"])

    def test_r4_04_preflight_blocked_snapshot(self):
        with patch("orchestrator.WORK", self.root):
            orchestrator.blocked_status("A", "mock preflight failure")
        view = read_json(self.root / "sessions/A/progress_status.json")
        self.assertEqual(view["status"], "BLOCKED")
        self.assertTrue(all(r["status"] == "BLOCKED" for r in view["runs"].values()))
        self.assertTrue(Path(view["receipt_path"]).exists())

    def test_r4_05_packaging_failure_still_finalizes_receipt(self):
        c = self.controller()
        retained = self.text(c.directory / "retained_seed.zip")
        with patch("orchestrator.session_package", side_effect=OSError("mock disk fail")):
            receipt = c.finish()
        self.assertEqual(receipt["status"], "PARTIAL_OUTPUT_PACKAGING")
        self.assertEqual(
            read_json(c.directory / "progress_status.json")["status"], receipt["status"]
        )
        self.assertTrue(retained.exists())
        self.assertTrue((c.directory / "execution_summary.json").exists())

    def test_r5_01_exit_mapping_truthful_modes(self):
        self.assertEqual(orchestrator.exit_code("COMPLETE"), 0)
        for status in ["PARTIAL", "PARTIAL_OUTPUT_PACKAGING", "FAILED", "BLOCKED"]:
            self.assertNotEqual(orchestrator.exit_code(status), 0)
        self.assertEqual(orchestrator.exit_code("PREFLIGHT_ONLY"), 0)
        self.assertEqual(orchestrator.exit_code("SMOKE_ONLY"), 0)

    def test_r5_02_main_returns_partial_only_after_finalization(self):
        # Exercise real main CLI and locking with mocked preflight/worker work, without spawning processes.
        protocol = read_json(common.HERE / "protocol.json")
        contract = {"fixture": True}
        smoke = self.root / "smoke.json"
        atomic_json(
            smoke,
            {
                "status": "SMOKE_PASS_REVIEW_REQUIRED",
                "runtime_contract": contract,
                "protocol_identity": identity(protocol),
            },
        )
        argv = [
            "orchestrator",
            "--session",
            "A",
            "--mode",
            "FULL_TRAIN",
            "--production-approved",
            "--approved-smoke",
            str(smoke),
            "--approved-smoke-sha256",
            sha(smoke),
            "--started-unix",
            str(time.time()),
        ]
        fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)

        def run(c):
            return c.finish()

        with (
            patch.dict("sys.modules", {"fcntl": fake_fcntl}),
            patch("sys.argv", argv),
            patch("orchestrator.WORK", self.root),
            patch(
                "orchestrator.preflight_run",
                return_value={
                    "runtime_contract": contract,
                    "protocol_identity": identity(protocol),
                },
            ),
            patch.object(orchestrator.Controller, "run", run),
        ):
            code = orchestrator.main()
        self.assertEqual(code, 20)
        receipt = read_json(self.root / "sessions/A/session_receipt.json")
        self.assertEqual(receipt["status"], "PARTIAL")
        current = read_json(self.root / "sessions/A/current_master.json")
        verify_zip(current["path"], current["sha256"])

    def test_r5_03_control_plane_exception_attempts_finalization(self):
        protocol = read_json(common.HERE / "protocol.json")
        contract = {}
        smoke = self.root / "smoke.json"
        atomic_json(
            smoke,
            {
                "status": "SMOKE_PASS_REVIEW_REQUIRED",
                "runtime_contract": contract,
                "protocol_identity": identity(protocol),
            },
        )
        argv = [
            "x",
            "--session",
            "A",
            "--mode",
            "FULL_TRAIN",
            "--production-approved",
            "--approved-smoke",
            str(smoke),
            "--approved-smoke-sha256",
            sha(smoke),
            "--started-unix",
            str(time.time()),
        ]
        f = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)
        with (
            patch.dict("sys.modules", {"fcntl": f}),
            patch("sys.argv", argv),
            patch("orchestrator.WORK", self.root),
            patch(
                "orchestrator.preflight_run",
                return_value={
                    "runtime_contract": contract,
                    "protocol_identity": identity(protocol),
                },
            ),
            patch.object(
                orchestrator.Controller,
                "run",
                side_effect=RuntimeError("mock control plane failure"),
            ),
        ):
            with self.assertRaises(RuntimeError):
                orchestrator.main()
        self.assertTrue((self.root / "sessions/A/session_receipt.json").exists())
        view = read_json(self.root / "sessions/A/progress_status.json")
        self.assertNotEqual(view["status"], "COMPLETE")

    def test_r1_09_completed_seed_worker_never_launches_process(self):
        c = self.controller()
        protocol = c.protocol
        directory, cfg = common.training_config(protocol, "gcn", 0, work=self.root)
        atomic_json(directory / "runtime.json", {"runtime_contract": {}})
        result = {
            "status": "COMPLETE",
            "backbone": "gcn",
            "seed": 0,
            "checkpoint_path": "verified fixture",
            "training_runtime_seconds": 1,
        }
        c.stop_new.set()
        with (
            patch(
                "orchestrator.training_config",
                side_effect=lambda p, b, s: common.training_config(p, b, s, work=self.root),
            ),
            patch(
                "orchestrator.verify_completed_package",
                side_effect=lambda d: result if d == directory else None,
            ),
            patch("orchestrator.subprocess.Popen", side_effect=AssertionError("must not retrain")),
        ):
            c.worker("gcn")
        self.assertEqual(c.results, [result])
        self.assertEqual(c.status["gcn/seed_0"]["status"], "COMPLETE")

    def test_r1_10_gate_with_real_cpu_checkpoint_and_tamper(self):
        import torch
        import numpy as np
        from state_verify import verify_epoch

        self.resume_fixture()
        group = read_json(common.HERE / "protocol.json")["optimizer_group"]
        self.protocol = {"optimizer_group": group, "runner_input_hashes": {"dataset": "fixture"}}
        self.runtime = {
            "runtime_contract": {"protocol_identity": identity(self.protocol)},
            "hard_compatibility": {"fixture": True},
        }
        shapes = [(64, 270), (64,), (1, 64), (1,)]
        payload = {
            "epoch_completed": 11,
            "algorithm_current_epoch": 10,
            "backbone": "GCN",
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
                        "step": torch.tensor(22000.0),
                        "exp_avg": torch.zeros(s),
                        "exp_avg_sq": torch.zeros(s),
                    }
                    for i, s in enumerate(shapes)
                },
            },
            "epoch_rows": [
                {
                    "epoch": e,
                    "seed": 0,
                    "backbone": "GCN",
                    "num_train_instances": 2000,
                    "mean_loss": 0.1,
                    "median_loss": 0.1,
                    "min_loss": 0.1,
                    "max_loss": 0.1,
                    "runtime_seconds": 1.0,
                }
                for e in range(1, 12)
            ],
            "hashes": {"task_config": sha(self.config), "dataset": "fixture"},
            "rng_state_cpu": torch.zeros(5056, dtype=torch.uint8),
            "rng_state_cuda": [torch.zeros(16, dtype=torch.uint8)],
            "rng_state_numpy": ("MT19937", np.zeros(624, dtype=np.uint32), 0, 0, 0.0),
            "hard_compatibility": {"fixture": True},
            "config_identity": "expected",
            "runtime_ledger": {},
        }
        path = self.states / "epoch_011.pt"
        path.parent.mkdir(parents=True)
        torch.save(payload, path)
        row = verify_epoch(path, "gcn", 0, self.config, self.protocol, self.runtime)
        atomic_json(
            self.directory / "verified_epoch_011.json",
            {**row, "runtime_contract": self.runtime["runtime_contract"]},
        )
        result = recovery.resume_gate(
            self.directory, "gcn", 0, self.config, self.protocol, self.runtime, lambda p: None
        )
        self.assertEqual(result["durable_updates"], 22000)
        # Even a fresh matching checksum cannot admit a checkpoint with invalid optimizer accounting.
        payload["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(21999.0)
        torch.save(payload, path)
        for name in ["verified_epoch_011.json", "durable_progress.json"]:
            atomic_json(
                self.directory / name,
                {**row, "sha256": sha(path), "runtime_contract": self.runtime["runtime_contract"]},
            )
        with self.assertRaises(GateError):
            recovery.resume_gate(
                self.directory, "gcn", 0, self.config, self.protocol, self.runtime, lambda p: None
            )

    def test_r2_09_publication_retries_without_worker_retraining(self):
        c = self.controller()
        protocol = c.protocol
        directory, cfg = common.training_config(protocol, "gcn", 0, work=self.root)
        paths = state_paths(directory, "gcn", 0)
        self.text(paths["run"] / "COMPLETE.json", "{}")
        atomic_json(directory / "runtime.json", {"runtime_contract": {}})
        result = {
            "status": "COMPLETE",
            "backbone": "gcn",
            "seed": 0,
            "checkpoint_path": "fixture",
            "training_runtime_seconds": 1,
        }
        c.stop_new.set()
        with (
            patch(
                "orchestrator.training_config",
                side_effect=lambda p, b, s: common.training_config(p, b, s, work=self.root),
            ),
            patch("orchestrator.verify_completed_package", return_value=None),
            patch("orchestrator.file_gate"),
            patch("state_verify.verify_final", return_value={"status": "FINAL_VERIFIED"}),
            patch("orchestrator.seed_package", return_value=result),
            patch(
                "orchestrator.subprocess.Popen",
                side_effect=AssertionError("publication must not retrain"),
            ),
        ):
            c.worker("gcn")
        self.assertEqual(c.results, [result])

    def test_r5_04_main_preflight_failure_returns_blocked(self):
        argv = [
            "x",
            "--session",
            "A",
            "--mode",
            "PREFLIGHT_ONLY",
            "--started-unix",
            str(time.time()),
        ]
        f = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)
        with (
            patch.dict("sys.modules", {"fcntl": f}),
            patch("sys.argv", argv),
            patch("orchestrator.WORK", self.root),
            patch("orchestrator.preflight_run", side_effect=GateError("mock wrong driver")),
        ):
            self.assertEqual(orchestrator.main(), 23)
        self.assertEqual(
            read_json(self.root / "sessions/A/progress_status.json")["status"], "BLOCKED"
        )

    def test_r5_05_successful_preflight_is_not_production_complete(self):
        argv = [
            "x",
            "--session",
            "A",
            "--mode",
            "PREFLIGHT_ONLY",
            "--started-unix",
            str(time.time()),
        ]
        f = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)
        with (
            patch.dict("sys.modules", {"fcntl": f}),
            patch("sys.argv", argv),
            patch("orchestrator.WORK", self.root),
            patch("orchestrator.preflight_run", return_value={"status": "PREFLIGHT_PASS"}),
        ):
            self.assertEqual(orchestrator.main(), 0)
        self.assertEqual(
            read_json(self.root / "sessions/A/mode_result.json"),
            {"status": "PREFLIGHT_ONLY", "production_complete": False},
        )

    def test_r3_05_master_index_commit_failure_retry(self):
        receipt = {"session": "A", "status": "PARTIAL", "protocol_identity": "fixture"}
        original = export_utils.atomic_json

        def fault(path, *a, **kw):
            if Path(path).name == "current_master.json":
                raise OSError("index commit injection")
            return original(path, *a, **kw)

        with patch("export_utils.atomic_json", side_effect=fault):
            with self.assertRaises(OSError):
                self.master(receipt)
        old = self.root / "session/TASK29_ELLIPTIC_STABILITY_SESSION_A_ALL.zip"
        before = sha(old)
        result = self.master(receipt)
        verify_zip(result["path"], result["sha256"])
        self.assertEqual(sha(old), before)

    def test_r2_10_final_checksum_binding_after_archive_write(self):
        fixture = self.seed_fixture()
        original = export_utils.create_zip

        def mutation(path, files, metadata):
            Path(files["checkpoint.pt"]).write_text("changed after prepublication check")
            return original(path, files, metadata)

        with patch("export_utils.create_zip", side_effect=mutation):
            with self.assertRaises(GateError):
                self.publish(fixture)
        self.assertFalse((fixture[0] / "TASK29_COMPLETE.json").exists())

    def test_restore_01_exact_partial_restore_is_idempotent(self):
        case = offline_tests.OfflineTests(
            "test_33_restore_partial_archive_to_exact_epoch_namespace"
        )
        import restore

        original = restore.restore

        def twice(*args, **kw):
            first = original(*args, **kw)
            self.assertEqual(original(*args, **kw), first)
            return first

        with patch("restore.restore", side_effect=twice):
            result = unittest.TestResult()
            case.run(result)
        self.assertEqual(result.failures + result.errors, [])

    def test_r1_11_fresh_warnings_idempotent_then_started_run_blocks(self):
        self.resume_fixture()
        self.text(
            self.directory / "stderr.log", "benign import warning before first scientific execution"
        )
        self.text(self.directory / "stdout.log", "library startup message")
        first = self.gate()
        self.assertEqual(first["status"], "FRESH_START")
        self.assertEqual(self.gate()["status"], "FRESH_START")
        recovery.begin_scientific_execution(self.directory, first)
        with self.assertRaises(GateError):
            self.gate()
        with self.assertRaises(GateError):
            self.gate()
        self.assertTrue(
            read_json(self.directory / "recovery_admission.json")["previous_execution_started"]
        )

    def test_r1_12_quarantine_evidence_in_master_is_not_recoverable(self):
        c = self.controller()
        directory = c.directory.parent.parent / "runs/gcn/seed_0"
        evidence = self.text(
            directory / "recovery_quarantine/checkpoints/rejected.pt.digest",
            "unverified rejected bytes",
        )
        atomic_json(
            directory / "recovery_admission.json", {"status": "BLOCKED", "epoch_completed": 0}
        )
        result = export_utils.session_package(
            c.directory,
            {"session": "A", "status": "PARTIAL", "protocol_identity": "fixture"},
            {},
            self.text(self.root / "guide.md"),
            c.forensics(),
        )
        manifest = verify_zip(result["path"], result["sha256"])
        name = "forensics/gcn/seed_0/recovery_quarantine/checkpoints/rejected.pt.digest"
        self.assertEqual(manifest["files"][name]["sha256"], sha(evidence))
        self.assertFalse(any(n.startswith("recoverable/") for n in manifest["files"]))

    def test_r4_06_resumed_worker_failure_keeps_verified_epoch(self):
        c = self.controller()
        c.event(
            {
                "event": "WORKER_STARTED",
                "backbone": "gcn",
                "seed": 0,
                "last_committed_epoch": 11,
                "durable_updates": 22000,
                "verified_checkpoint_path": "verified_epoch_011.pt",
            }
        )
        c.failure("gcn", 0, "WORKER_RUNTIME_FAILURE", "interrupted during replay epoch12")
        with patch("orchestrator.session_package", return_value={}):
            c.finish()
        view = read_json(c.directory / "progress_status.json")["runs"]["gcn/seed_0"]
        self.assertEqual(view["last_committed_epoch"], 11)
        self.assertEqual(view["durable_updates"], 22000)
        self.assertEqual(view["verified_checkpoint_path"], "verified_epoch_011.pt")
        self.assertIsNone(view["last_observed_target"])


if __name__ == "__main__":
    unittest.main()
