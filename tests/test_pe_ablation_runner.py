import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


PY38_PLUS = sys.version_info >= (3, 8)
PY38_SKIP_REASON = (
    "PE ablation runner tests require Python >= 3.8; system Python 3.6 "
    "is skipped explicitly so it does not import Python>=3.7-only training code."
)


if not PY38_PLUS:
    class PeAblationRunnerPythonVersionSkipTest(unittest.TestCase):
        @unittest.skip(PY38_SKIP_REASON)
        def test_python38_required(self):
            pass


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_pe_ablation.py"
TEST_LAUNCHER_SECRET = "literal-test-secret-for-launcher-state"


def load_module():
    spec = importlib.util.spec_from_file_location("run_pe_ablation", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_bundle(module):
    payload = base64.b64decode(module.build_ablation_bundle_b64())
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        out = {}
        for member in tf.getmembers():
            if member.isfile():
                with tf.extractfile(member) as fh:
                    out[member.name] = fh.read()
    return out


@unittest.skipUnless(PY38_PLUS, PY38_SKIP_REASON)
class PeAblationRunnerCommandTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.cmd = self.module.build_ablation_cmd()

    def test_http_boot_command_arms_self_termination_before_payload(self):
        boot_cmd = self.module.build_boot_command("printf 'literal {brace} ${PGOLF_BUNDLE_B64}\\n'")
        self.assertIn("[pgolf-selfterm] Self-termination armed", boot_cmd)
        self.assertIn("podTerminate", boot_cmd)
        self.assertIn("pgolf_extract_stdout.txt", boot_cmd)
        self.assertIn("bundle extraction failed", boot_cmd)
        self.assertIn("pgolf_exit_code.txt", boot_cmd)
        self.assertIn("pgolf_stdout.txt", boot_cmd)
        self.assertIn("status.txt", boot_cmd)
        self.assertIn("exit \"${extract_ec}\"", boot_cmd)
        self.assertIn("literal {brace}", boot_cmd)
        self.assertLess(
            boot_cmd.index("[pgolf-selfterm] Self-termination armed"),
            boot_cmd.index("PGOLF_BUNDLE_B64"),
        )

    def test_full_ablation_boot_command_contains_selfterm_marker(self):
        boot_cmd = self.module.build_boot_command(self.cmd)
        self.assertIn("[pgolf-selfterm] Self-termination armed", boot_cmd)
        self.assertIn("PGOLF_HARD_DEADLINE_SEC", boot_cmd)
        self.assertNotIn("__PGOLF_SELFTERM_PREAMBLE__", boot_cmd)
        self.assertNotIn("__PGOLF_USER_CMD__", boot_cmd)
        self.assertLess(
            boot_cmd.index("[pgolf-selfterm] Self-termination armed"),
            boot_cmd.index("PGOLF_BUNDLE_B64"),
        )

    def test_http_boot_starts_server_before_user_command(self):
        boot_cmd = self.module.build_boot_command("printf 'literal {brace} ${PGOLF_BUNDLE_B64}\\n'")
        self.assertIn("PGOLF_HTTP_PID", boot_cmd)
        self.assertIn("http_server.log", boot_cmd)
        self.assertIn("RUNNING", boot_cmd)
        self.assertLess(
            boot_cmd.index("python3 -m http.server 30000"),
            boot_cmd.index("literal {brace}"),
        )

    def test_http_boot_timeout_writes_retrievable_artifacts(self):
        boot_cmd = self.module.build_boot_command(self.cmd)
        self.assertIn("PGOLF_MAX_MINUTES", boot_cmd)
        self.assertIn("pgolf_cmd_pid", boot_cmd)
        self.assertIn("kill -TERM", boot_cmd)
        self.assertIn("kill -KILL", boot_cmd)
        self.assertIn("TIMEOUT", boot_cmd)
        self.assertIn("/root/rehearsal_out/pgolf_exit_code.txt", boot_cmd)
        self.assertIn("/root/rehearsal_out/overall_exit_code.txt", boot_cmd)
        self.assertIn("/root/rehearsal_out/status.txt", boot_cmd)
        self.assertIn("ec=124", boot_cmd)
        self.assertIn("exit ${ec}", boot_cmd)

    def test_pe_ablation_launcher_passes_pgolf_max_minutes_to_pod_env(self):
        captured = {}

        def fake_create_pod(**kwargs):
            captured.update(kwargs)
            return {"id": "pod-test", "costPerHr": "2.99"}

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "run_pe_ablation.py",
                "--smoke",
                "--max-minutes",
                "3",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(self.module, "balance", return_value=(100.0, None)), \
                    mock.patch.object(self.module, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(self.module, "wait_runtime", return_value={"uptimeInSeconds": 1}), \
                    mock.patch.object(self.module, "wait_http_proxy", return_value=None), \
                    mock.patch.object(self.module, "download_file", return_value=None), \
                    mock.patch.object(self.module, "terminate_and_wait", return_value=None):
                self.module.main()

        self.assertEqual(captured["extra_env"]["PGOLF_MAX_MINUTES"], "3")
        self.assertIn("PGOLF_BUNDLE_B64", captured["extra_env"])

    def test_pe_ablation_launcher_passes_custom_runtime_timeout_to_wait_runtime(self):
        captured = {}

        def fake_create_pod(**kwargs):
            return {"id": "pod-runtime-timeout", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            captured["pod_id"] = pod_id
            captured["timeout"] = timeout
            return {"uptimeInSeconds": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "run_pe_ablation.py",
                "--smoke",
                "--max-minutes",
                "3",
                "--runtime-timeout-sec",
                "17",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(self.module, "balance", return_value=(100.0, None)), \
                    mock.patch.object(self.module, "build_ablation_bundle_b64", return_value="small-bundle"), \
                    mock.patch.object(self.module, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(self.module, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(self.module, "wait_http_proxy", return_value="DONE"), \
                    mock.patch.object(self.module, "download_file", return_value=None), \
                    mock.patch.object(self.module, "terminate_and_wait", return_value=None):
                self.module.main()

            state = json.loads((Path(tmpdir) / "launcher_state.json").read_text(encoding="utf-8"))

        self.assertEqual(captured, {"pod_id": "pod-runtime-timeout", "timeout": 17})
        self.assertEqual(state["runtime_timeout_sec"], 17)

    def test_pe_ablation_attempts_startup_readiness_before_terminal_wait(self):
        wait_calls = []
        downloads = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-startup-readiness", "costPerHr": "2.99"}

        def fake_wait_http_proxy(pod_id, port, **kwargs):
            wait_calls.append((pod_id, port, kwargs))
            return "RUNNING" if len(wait_calls) == 1 else "DONE"

        def fake_download_file(pod_id, port, name, out_dir, optional=False, **kwargs):
            downloads.append((pod_id, port, name, optional, kwargs))
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "run_pe_ablation.py",
                "--smoke",
                "--max-minutes",
                "3",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(self.module, "balance", return_value=(100.0, None)), \
                    mock.patch.object(self.module, "build_ablation_bundle_b64", return_value="small-bundle"), \
                    mock.patch.object(self.module, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(self.module, "wait_runtime", return_value={"uptimeInSeconds": 1}), \
                    mock.patch.object(self.module, "wait_http_proxy", side_effect=fake_wait_http_proxy), \
                    mock.patch.object(self.module, "download_file", side_effect=fake_download_file), \
                    mock.patch.object(self.module, "terminate_and_wait", return_value=None):
                self.module.main()

        self.assertEqual(len(wait_calls), 2)
        self.assertTrue(wait_calls[0][2].get("startup_readiness"))
        self.assertFalse(wait_calls[1][2].get("startup_readiness", False))
        self.assertEqual(downloads[0][2], "status.txt")

    def test_pe_ablation_writes_launcher_state_before_wait_runtime_and_redacts(self):
        captured = {}
        raw_bundle = "bundle-containing-" + TEST_LAUNCHER_SECRET

        def fake_create_pod(**kwargs):
            captured.update(kwargs)
            return {"id": "pod-state", "costPerHr": "2.99"}

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "launcher_state.json"

            def fake_wait_runtime(pod_id, timeout=None):
                self.assertEqual(pod_id, "pod-state")
                self.assertTrue(state_path.exists(), "launcher_state.json must exist before wait_runtime")
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state["phase"], "pod_created")
                self.assertEqual(state["pod_id"], "pod-state")
                self.assertEqual(state["pod_name"], "pgolf-pe-ablation-smoke")
                self.assertEqual(state["gpus"], 1)
                self.assertEqual(state["max_minutes"], 3)
                self.assertEqual(state["results_dir"], tmpdir)
                self.assertEqual(state["hard_deadline_sec"], 3 * 60 + self.module.RETRIEVAL_BUFFER_SECONDS)
                self.assertEqual(state["bundle_b64_length"], len(raw_bundle))
                self.assertEqual(state["bundle_b64_sha256"], hashlib.sha256(raw_bundle.encode("utf-8")).hexdigest())
                self.assertRegex(state["command_sha256"], r"^[0-9a-f]{64}$")
                self.assertFalse(state["cleanup_attempted"])
                self.assertEqual(state["cleanup_status"], "not_started")
                return {"uptimeInSeconds": 1}

            argv = [
                "run_pe_ablation.py",
                "--smoke",
                "--max-minutes",
                "3",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(os.environ, {"RUNPOD_API_KEY": TEST_LAUNCHER_SECRET}, clear=False), \
                    mock.patch.object(self.module, "balance", return_value=(100.0, None)), \
                    mock.patch.object(self.module, "build_ablation_bundle_b64", return_value=raw_bundle), \
                    mock.patch.object(self.module, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(self.module, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(self.module, "wait_http_proxy", return_value=None), \
                    mock.patch.object(self.module, "download_file", return_value=None), \
                    mock.patch.object(self.module, "terminate_and_wait", return_value=None):
                self.module.main()

            raw_state = state_path.read_text(encoding="utf-8")
            self.assertNotIn(TEST_LAUNCHER_SECRET, raw_state)
            self.assertNotIn(raw_bundle, raw_state)
            self.assertNotIn("RUNPOD_API_KEY", raw_state)
            self.assertNotIn("PGOLF_BUNDLE_B64", raw_state)
            state = json.loads(raw_state)
            self.assertEqual(state["phase"], "cleanup_completed")
            self.assertTrue(state["cleanup_attempted"])
            self.assertEqual(state["cleanup_status"], "succeeded")
            self.assertEqual(captured["extra_env"]["PGOLF_BUNDLE_B64"], raw_bundle)

    def test_pe_ablation_cleanup_recorded_on_keyboard_interrupt(self):
        terminations = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-keyboard", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "run_pe_ablation.py",
                "--smoke",
                "--max-minutes",
                "3",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(self.module, "balance", return_value=(100.0, None)), \
                    mock.patch.object(self.module, "build_ablation_bundle_b64", return_value="small-bundle"), \
                    mock.patch.object(self.module, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(self.module, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(self.module, "terminate_and_wait", side_effect=lambda pod_id: terminations.append(pod_id)):
                with self.assertRaises(KeyboardInterrupt):
                    self.module.main()

            state = json.loads((Path(tmpdir) / "launcher_state.json").read_text(encoding="utf-8"))
            self.assertEqual(terminations, ["pod-keyboard"])
            self.assertEqual(state["last_exception_type"], "KeyboardInterrupt")
            self.assertTrue(state["cleanup_attempted"])
            self.assertEqual(state["cleanup_status"], "succeeded")

    def test_pe_ablation_terminates_when_cleanup_state_write_fails_after_pod_created(self):
        import runpod_http_rehearsal

        terminations = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-cleanup-state-write-fails", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            raise RuntimeError("runtime exploded before cleanup")

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "run_pe_ablation.py",
                "--smoke",
                "--max-minutes",
                "3",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(self.module, "balance", return_value=(100.0, None)), \
                    mock.patch.object(self.module, "build_ablation_bundle_b64", return_value="small-bundle"), \
                    mock.patch.object(self.module, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(self.module, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(runpod_http_rehearsal, "write_launcher_state", side_effect=RuntimeError("state write failed")), \
                    mock.patch.object(self.module, "terminate_and_wait", side_effect=lambda pod_id: terminations.append(pod_id)):
                with self.assertRaisesRegex(RuntimeError, "runtime exploded before cleanup"):
                    self.module.main()

        self.assertEqual(terminations, ["pod-cleanup-state-write-fails"])

    def test_pe_ablation_terminates_on_runtime_exception(self):
        terminations = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-runtime-error", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            raise RuntimeError("runtime exploded")

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "run_pe_ablation.py",
                "--smoke",
                "--max-minutes",
                "3",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(self.module, "balance", return_value=(100.0, None)), \
                    mock.patch.object(self.module, "build_ablation_bundle_b64", return_value="small-bundle"), \
                    mock.patch.object(self.module, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(self.module, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(self.module, "terminate_and_wait", side_effect=lambda pod_id: terminations.append(pod_id)):
                with self.assertRaisesRegex(RuntimeError, "runtime exploded"):
                    self.module.main()

            state = json.loads((Path(tmpdir) / "launcher_state.json").read_text(encoding="utf-8"))
            self.assertEqual(terminations, ["pod-runtime-error"])
            self.assertEqual(state["last_exception_type"], "RuntimeError")
            self.assertTrue(state["cleanup_attempted"])
            self.assertEqual(state["cleanup_status"], "succeeded")

    def test_pe_ablation_cleanup_recorded_on_system_exit_without_secret_message(self):
        terminations = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-system-exit", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            raise SystemExit("intentional exit " + TEST_LAUNCHER_SECRET)

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "run_pe_ablation.py",
                "--smoke",
                "--max-minutes",
                "3",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(self.module, "balance", return_value=(100.0, None)), \
                    mock.patch.object(self.module, "build_ablation_bundle_b64", return_value="small-bundle"), \
                    mock.patch.object(self.module, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(self.module, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(self.module, "terminate_and_wait", side_effect=lambda pod_id: terminations.append(pod_id)):
                with self.assertRaises(SystemExit):
                    self.module.main()

            raw_state = (Path(tmpdir) / "launcher_state.json").read_text(encoding="utf-8")
            state = json.loads(raw_state)
            self.assertEqual(terminations, ["pod-system-exit"])
            self.assertEqual(state["last_exception_type"], "SystemExit")
            self.assertRegex(state["last_exception_message_sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn(TEST_LAUNCHER_SECRET, raw_state)
            self.assertTrue(state["cleanup_attempted"])
            self.assertEqual(state["cleanup_status"], "succeeded")

    def test_http_rehearsal_launcher_passes_pgolf_max_minutes_to_pod_env(self):
        import runpod_http_rehearsal

        captured = {}

        def fake_create_pod(**kwargs):
            captured.update(kwargs)
            return {"id": "pod-test", "costPerHr": "2.99"}

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "runpod_http_rehearsal.py",
                "--max-minutes",
                "2",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runpod_http_rehearsal, "balance", return_value=(100.0, None)), \
                    mock.patch.object(runpod_http_rehearsal, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(runpod_http_rehearsal, "wait_runtime", return_value={"uptimeInSeconds": 1}), \
                    mock.patch.object(runpod_http_rehearsal, "wait_http_proxy", return_value=None), \
                    mock.patch.object(runpod_http_rehearsal, "download_file", return_value=None), \
                    mock.patch.object(runpod_http_rehearsal, "terminate_and_wait", return_value=None):
                runpod_http_rehearsal.main()

        self.assertEqual(captured["extra_env"]["PGOLF_MAX_MINUTES"], "2")
        self.assertIn("PGOLF_BUNDLE_B64", captured["extra_env"])

    def test_http_rehearsal_launcher_passes_custom_runtime_timeout_to_wait_runtime(self):
        import runpod_http_rehearsal

        captured = {}

        def fake_create_pod(**kwargs):
            return {"id": "pod-http-runtime-timeout", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            captured["pod_id"] = pod_id
            captured["timeout"] = timeout
            return {"uptimeInSeconds": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "runpod_http_rehearsal.py",
                "--max-minutes",
                "2",
                "--runtime-timeout-sec",
                "23",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runpod_http_rehearsal, "balance", return_value=(100.0, None)), \
                    mock.patch.object(runpod_http_rehearsal, "build_bundle_b64", return_value="small-http-bundle"), \
                    mock.patch.object(runpod_http_rehearsal, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(runpod_http_rehearsal, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(runpod_http_rehearsal, "wait_http_proxy", return_value="DONE"), \
                    mock.patch.object(runpod_http_rehearsal, "download_file", return_value=None), \
                    mock.patch.object(runpod_http_rehearsal, "terminate_and_wait", return_value=None):
                runpod_http_rehearsal.main()

            state = json.loads((Path(tmpdir) / "launcher_state.json").read_text(encoding="utf-8"))

        self.assertEqual(captured, {"pod_id": "pod-http-runtime-timeout", "timeout": 23})
        self.assertEqual(state["runtime_timeout_sec"], 23)

    def test_http_rehearsal_attempts_startup_readiness_before_terminal_wait(self):
        import runpod_http_rehearsal

        wait_calls = []
        downloads = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-http-startup-readiness", "costPerHr": "2.99"}

        def fake_wait_http_proxy(pod_id, port, **kwargs):
            wait_calls.append((pod_id, port, kwargs))
            return "RUNNING" if len(wait_calls) == 1 else "DONE"

        def fake_download_file(pod_id, port, name, out_dir, optional=False, **kwargs):
            downloads.append((pod_id, port, name, optional, kwargs))
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "runpod_http_rehearsal.py",
                "--max-minutes",
                "2",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runpod_http_rehearsal, "balance", return_value=(100.0, None)), \
                    mock.patch.object(runpod_http_rehearsal, "build_bundle_b64", return_value="small-http-bundle"), \
                    mock.patch.object(runpod_http_rehearsal, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(runpod_http_rehearsal, "wait_runtime", return_value={"uptimeInSeconds": 1}), \
                    mock.patch.object(runpod_http_rehearsal, "wait_http_proxy", side_effect=fake_wait_http_proxy), \
                    mock.patch.object(runpod_http_rehearsal, "download_file", side_effect=fake_download_file), \
                    mock.patch.object(runpod_http_rehearsal, "terminate_and_wait", return_value=None):
                runpod_http_rehearsal.main()

        self.assertEqual(len(wait_calls), 2)
        self.assertTrue(wait_calls[0][2].get("startup_readiness"))
        self.assertFalse(wait_calls[1][2].get("startup_readiness", False))
        self.assertEqual(downloads[0][2], "status.txt")

    def test_http_rehearsal_writes_launcher_state_before_wait_runtime_and_redacts(self):
        import runpod_http_rehearsal

        captured = {}
        raw_bundle = "http-bundle-containing-" + TEST_LAUNCHER_SECRET
        raw_cmd = "printf '%s\\n' " + TEST_LAUNCHER_SECRET

        def fake_create_pod(**kwargs):
            captured.update(kwargs)
            return {"id": "pod-http-state", "costPerHr": "2.99"}

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "launcher_state.json"

            def fake_wait_runtime(pod_id, timeout=None):
                self.assertEqual(pod_id, "pod-http-state")
                self.assertTrue(state_path.exists(), "launcher_state.json must exist before wait_runtime")
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state["phase"], "pod_created")
                self.assertEqual(state["pod_id"], "pod-http-state")
                self.assertEqual(state["pod_name"], "pgolf-http-1gpu")
                self.assertEqual(state["gpus"], 1)
                self.assertEqual(state["max_minutes"], 2)
                self.assertEqual(state["results_dir"], tmpdir)
                self.assertEqual(state["hard_deadline_sec"], 2 * 60 + 120)
                self.assertEqual(state["bundle_b64_length"], len(raw_bundle))
                self.assertEqual(state["bundle_b64_sha256"], hashlib.sha256(raw_bundle.encode("utf-8")).hexdigest())
                self.assertEqual(state["command_length"], len(raw_cmd))
                self.assertEqual(state["command_sha256"], hashlib.sha256(raw_cmd.encode("utf-8")).hexdigest())
                return {"uptimeInSeconds": 1}

            argv = [
                "runpod_http_rehearsal.py",
                "--max-minutes",
                "2",
                "--results-dir",
                tmpdir,
                "--cmd",
                raw_cmd,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.dict(os.environ, {"RUNPOD_API_KEY": TEST_LAUNCHER_SECRET}, clear=False), \
                    mock.patch.object(runpod_http_rehearsal, "balance", return_value=(100.0, None)), \
                    mock.patch.object(runpod_http_rehearsal, "build_bundle_b64", return_value=raw_bundle), \
                    mock.patch.object(runpod_http_rehearsal, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(runpod_http_rehearsal, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(runpod_http_rehearsal, "wait_http_proxy", return_value=None), \
                    mock.patch.object(runpod_http_rehearsal, "download_file", return_value=None), \
                    mock.patch.object(runpod_http_rehearsal, "terminate_and_wait", return_value=None):
                runpod_http_rehearsal.main()

            raw_state = state_path.read_text(encoding="utf-8")
            state = json.loads(raw_state)
            self.assertNotIn(TEST_LAUNCHER_SECRET, raw_state)
            self.assertNotIn(raw_bundle, raw_state)
            self.assertNotIn(raw_cmd, raw_state)
            self.assertNotIn("RUNPOD_API_KEY", raw_state)
            self.assertNotIn("PGOLF_BUNDLE_B64", raw_state)
            self.assertEqual(state["phase"], "cleanup_completed")
            self.assertEqual(state["cleanup_status"], "succeeded")
            self.assertEqual(captured["extra_env"]["PGOLF_BUNDLE_B64"], raw_bundle)

    def test_http_rehearsal_cleanup_recorded_on_system_exit(self):
        import runpod_http_rehearsal

        terminations = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-http-system-exit", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            raise SystemExit("http exit " + TEST_LAUNCHER_SECRET)

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "runpod_http_rehearsal.py",
                "--max-minutes",
                "2",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runpod_http_rehearsal, "balance", return_value=(100.0, None)), \
                    mock.patch.object(runpod_http_rehearsal, "build_bundle_b64", return_value="small-http-bundle"), \
                    mock.patch.object(runpod_http_rehearsal, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(runpod_http_rehearsal, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(runpod_http_rehearsal, "terminate_and_wait", side_effect=lambda pod_id: terminations.append(pod_id)):
                with self.assertRaises(SystemExit):
                    runpod_http_rehearsal.main()

            raw_state = (Path(tmpdir) / "launcher_state.json").read_text(encoding="utf-8")
            state = json.loads(raw_state)
            self.assertEqual(terminations, ["pod-http-system-exit"])
            self.assertEqual(state["last_exception_type"], "SystemExit")
            self.assertNotIn(TEST_LAUNCHER_SECRET, raw_state)
            self.assertTrue(state["cleanup_attempted"])
            self.assertEqual(state["cleanup_status"], "succeeded")

    def test_http_rehearsal_cleanup_recorded_on_keyboard_interrupt(self):
        import runpod_http_rehearsal

        terminations = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-http-keyboard", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "runpod_http_rehearsal.py",
                "--max-minutes",
                "2",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runpod_http_rehearsal, "balance", return_value=(100.0, None)), \
                    mock.patch.object(runpod_http_rehearsal, "build_bundle_b64", return_value="small-http-bundle"), \
                    mock.patch.object(runpod_http_rehearsal, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(runpod_http_rehearsal, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(runpod_http_rehearsal, "terminate_and_wait", side_effect=lambda pod_id: terminations.append(pod_id)):
                with self.assertRaises(KeyboardInterrupt):
                    runpod_http_rehearsal.main()

            state = json.loads((Path(tmpdir) / "launcher_state.json").read_text(encoding="utf-8"))
            self.assertEqual(terminations, ["pod-http-keyboard"])
            self.assertEqual(state["last_exception_type"], "KeyboardInterrupt")
            self.assertTrue(state["cleanup_attempted"])
            self.assertEqual(state["cleanup_status"], "succeeded")

    def test_http_rehearsal_terminates_when_cleanup_state_write_fails_after_pod_created(self):
        import runpod_http_rehearsal

        terminations = []
        original_write_launcher_state = runpod_http_rehearsal.write_launcher_state
        write_calls = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-http-cleanup-state-write-fails", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            raise RuntimeError("http runtime exploded before cleanup")

        def flaky_write_launcher_state(results_dir, state):
            write_calls.append(state.get("phase"))
            if len(write_calls) == 1:
                return original_write_launcher_state(results_dir, state)
            raise RuntimeError("state write failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "runpod_http_rehearsal.py",
                "--max-minutes",
                "2",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runpod_http_rehearsal, "balance", return_value=(100.0, None)), \
                    mock.patch.object(runpod_http_rehearsal, "build_bundle_b64", return_value="small-http-bundle"), \
                    mock.patch.object(runpod_http_rehearsal, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(runpod_http_rehearsal, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(runpod_http_rehearsal, "write_launcher_state", side_effect=flaky_write_launcher_state), \
                    mock.patch.object(runpod_http_rehearsal, "terminate_and_wait", side_effect=lambda pod_id: terminations.append(pod_id)):
                with self.assertRaisesRegex(RuntimeError, "http runtime exploded before cleanup"):
                    runpod_http_rehearsal.main()

        self.assertEqual(terminations, ["pod-http-cleanup-state-write-fails"])
        self.assertIn("cleanup_started", write_calls)

    def test_http_rehearsal_terminates_on_runtime_exception(self):
        import runpod_http_rehearsal

        terminations = []

        def fake_create_pod(**kwargs):
            return {"id": "pod-http-runtime-error", "costPerHr": "2.99"}

        def fake_wait_runtime(pod_id, timeout=None):
            raise RuntimeError("http runtime exploded")

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "runpod_http_rehearsal.py",
                "--max-minutes",
                "2",
                "--results-dir",
                tmpdir,
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(runpod_http_rehearsal, "balance", return_value=(100.0, None)), \
                    mock.patch.object(runpod_http_rehearsal, "build_bundle_b64", return_value="small-http-bundle"), \
                    mock.patch.object(runpod_http_rehearsal, "create_pod", side_effect=fake_create_pod), \
                    mock.patch.object(runpod_http_rehearsal, "wait_runtime", side_effect=fake_wait_runtime), \
                    mock.patch.object(runpod_http_rehearsal, "terminate_and_wait", side_effect=lambda pod_id: terminations.append(pod_id)):
                with self.assertRaisesRegex(RuntimeError, "http runtime exploded"):
                    runpod_http_rehearsal.main()

            state = json.loads((Path(tmpdir) / "launcher_state.json").read_text(encoding="utf-8"))
            self.assertEqual(terminations, ["pod-http-runtime-error"])
            self.assertEqual(state["last_exception_type"], "RuntimeError")
            self.assertTrue(state["cleanup_attempted"])
            self.assertEqual(state["cleanup_status"], "succeeded")

    def test_http_proxy_wait_accepts_timeout_status(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"TIMEOUT\n"
        with mock.patch("runpod_http_rehearsal.urllib.request.urlopen", return_value=response) as urlopen:
            status = self.module.wait_http_proxy("pod-timeout", 30000, timeout=1)
        self.assertEqual(status, "TIMEOUT")
        self.assertTrue(urlopen.called)

    def test_http_proxy_startup_readiness_accepts_running_status(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"RUNNING\n"
        with mock.patch("runpod_http_rehearsal.urllib.request.urlopen", return_value=response) as urlopen:
            status = self.module.wait_http_proxy(
                "pod-running",
                30000,
                timeout=1,
                startup_readiness=True,
            )
        self.assertEqual(status, "RUNNING")
        self.assertTrue(urlopen.called)

    def test_http_proxy_terminal_wait_does_not_accept_running_status(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"RUNNING\n"
        with mock.patch("runpod_http_rehearsal.urllib.request.urlopen", return_value=response) as urlopen, \
                mock.patch("runpod_http_rehearsal.time.time", side_effect=[100.0, 100.1, 102.0]), \
                mock.patch("runpod_http_rehearsal.time.sleep", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                self.module.wait_http_proxy("pod-running", 30000, timeout=1)
        self.assertTrue(urlopen.called)

    def test_downloads_sp8192_once_before_runs(self):
        self.assertEqual(self.cmd.count("cached_challenge_fineweb.py --variant sp8192"), 1)
        self.assertLess(
            self.cmd.index("cached_challenge_fineweb.py --variant sp8192"),
            self.cmd.index("/root/runs/R0_fixed5_Gram"),
        )

    def test_runs_requested_variants_in_clean_dirs(self):
        for run_id in ("R0_fixed5_Gram", "R1_fixed4_Gram", "R2_PE4_Gram"):
            self.assertIn(f"/root/runs/{run_id}", self.cmd)
            self.assertIn(f"/root/rehearsal_out/{run_id}", self.cmd)

    def test_clean_dirs_share_downloaded_data(self):
        self.assertIn("/root/rehearsal_src/datasets", self.cmd)
        self.assertIn("/root/rehearsal_src/tokenizers", self.cmd)
        self.assertIn("ln -sfn", self.cmd)

    def test_no_result_variant_scripts_used_on_pod(self):
        self.assertNotIn("pe_ablation_R0_fixed5.py", self.cmd)
        self.assertNotIn("pe_ablation_R1_fixed4.py", self.cmd)
        self.assertNotIn("pe_ablation_R2_pe4.py", self.cmd)
        self.assertNotIn("results/pe_ablation_", self.cmd)
        self.assertNotIn("cp pe_ablation_", self.cmd)

    def test_run_failures_are_recorded_without_stopping_sequence(self):
        self.assertIn("run_status.csv", self.cmd)
        self.assertIn("exit_code.txt", self.cmd)
        self.assertIn("overall_exit_code", self.cmd)
        self.assertNotIn("set -euo pipefail", self.cmd)

    def test_shared_setup_failures_stop_before_per_run_collection(self):
        setup_section = self.cmd[:self.cmd.index("run_one R0_fixed5_Gram 5 fixed")]
        self.assertIn("pip install", setup_section)
        self.assertIn("cached_challenge_fineweb.py --variant sp8192", setup_section)
        self.assertIn("setup_ec=$?", setup_section)
        self.assertIn("exit \"${setup_ec}\"", setup_section)

    def test_csv_header_has_exact_requested_fields(self):
        expected = (
            "run_id,seed,ns_steps,coeff_type,gram_ns,steps_completed,"
            "train_seconds,optimizer_seconds,optimizer_pct,sliding_bpb,ttt_bpb,artifact_bytes"
        )
        self.assertEqual(self.module.CSV_HEADER, expected)
        self.assertIn(f"printf '{expected}\\n'", self.cmd)

    def test_fallback_csv_row_marks_missing_metrics_explicitly(self):
        row = self.module.fallback_csv_row("R0_fixed5_Gram", 42, 5, "fixed", True, 123)
        fields = next(csv.reader([row]))
        self.assertEqual(fields, [
            "R0_fixed5_Gram", "42", "5", "fixed", "yes",
            "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "123",
        ])
        self.assertIn("MISSING", self.cmd)

    def test_parse_pe_ablation_csv_requires_exact_field_count_and_uses_last(self):
        valid1 = "R0_fixed5_Gram,42,5,fixed,yes,10,11.5,1.5,13.0,1.23456789,N/A,123"
        invalid = "R1_fixed4_Gram,42,4,fixed,yes,,,,,,123"
        valid2 = "R2_PE4_Gram,42,4,pe,yes,12,12.5,1.7,13.6,1.11111111,1.00000000,456"
        log_text = "\n".join([
            "prefix PE_ABLATION_CSV: " + valid1,
            "prefix PE_ABLATION_CSV: " + invalid,
            "prefix PE_ABLATION_CSV: " + valid2,
        ])
        self.assertEqual(self.module.extract_pe_ablation_csv_rows(log_text), [valid1, valid2])

    def test_runner_expects_real_ablation_artifact_name_only(self):
        for artifacts in self.module.RUN_ARTIFACTS.values():
            self.assertIn("final_model.int6.ptz", artifacts)
            self.assertNotIn("final_model.int8.ptz", artifacts)
        self.assertNotIn("final_model.int8.ptz", self.cmd)
        self.assertIn("final_model.int6.ptz", self.cmd)
        self.assertIn("model_artifact_bytes=$(wc -c < \"${out_dir}/final_model.int6.ptz\")", self.cmd)

    def test_run_meta_and_runtime_env_are_exhaustive_and_consistent(self):
        expected_actual_overrides = (
            "RUN_ID",
            "SEED",
            "MAX_WALLCLOCK_SECONDS",
            "VOCAB_SIZE",
            "DATA_DIR",
            "QK_GAIN_INIT",
            "TTT_ENABLED",
            "TTT_LR",
            "TTT_EPOCHS",
            "SLIDING_WINDOW_ENABLED",
            "EVAL_STRIDE",
            "MIN_LR",
            "GPTQ_RESERVE_SECONDS",
            "MUON_BACKEND_STEPS",
            "MUON_NS_COEFF_TYPE",
            "MUON_NS_GRAM",
            "MATRIX_BITS",
            "EMBED_BITS",
            "COMPRESSOR",
        )
        for key in expected_actual_overrides:
            self.assertIn(f"{key}=", self.cmd, key)
            self.assertIn(f"export {key}=", self.cmd, key)
        self.assertNotIn("export DATA_PATH=", self.cmd)
        self.assertNotIn("export TOKENIZER_PATH=", self.cmd)
        self.assertIn("DATA_PATH_INFO=", self.cmd)
        self.assertIn("TOKENIZER_PATH_INFO=", self.cmd)
        self.assertIn("informational_only_template_derives_paths_from_DATA_DIR_and_VOCAB_SIZE", self.cmd)

    def test_artifact_cap_violation_is_visible_in_status_and_stdout(self):
        self.assertIn("run_id,exit_code,start_utc,end_utc,artifact_bytes,log_bytes,status,note", self.cmd)
        self.assertIn("ARTIFACT_CAP_BYTES=16000000", self.cmd)
        self.assertIn("CAP_VIOLATION", self.cmd)
        self.assertIn("artifact_bytes>=16000000", self.cmd)
        self.assertIn("not usable as record evidence", self.cmd)

    def test_ablation_command_honors_gpu_count_without_changing_default(self):
        self.assertIn("torchrun --standalone --nproc_per_node=8 train_gpt.py", self.cmd)
        one_gpu_cmd = self.module.build_ablation_cmd(gpus=1)
        self.assertIn("torchrun --standalone --nproc_per_node=1 train_gpt.py", one_gpu_cmd)

    def test_smoke_command_is_exact_bundle_compile_decode_probe(self):
        smoke_cmd = self.module.build_smoke_cmd()
        self.assertIn("SMOKE_MODE=compile_decode_no_training", smoke_cmd)
        self.assertIn("smoke_exact_payload", smoke_cmd)
        self.assertIn("_PAYLOAD", smoke_cmd)
        self.assertIn("compile(source", smoke_cmd)
        self.assertIn("run_status.csv", smoke_cmd)
        self.assertIn("retrieval smoke only; no training or validation data download", smoke_cmd)
        self.assertNotIn("torchrun", smoke_cmd)
        self.assertNotIn("cached_challenge_fineweb.py --variant sp8192", smoke_cmd)

    def test_default_production_budget_is_120_minutes(self):
        self.assertEqual(self.module.DEFAULT_MAX_MINUTES, 120)

    def test_command_sets_ablation_env_and_sp8192(self):
        self.assertIn("export VOCAB_SIZE=8192", self.cmd)
        self.assertIn("export MUON_NS_GRAM=1", self.cmd)
        self.assertIn("run_one R0_fixed5_Gram 5 fixed", self.cmd)
        self.assertIn("run_one R1_fixed4_Gram 4 fixed", self.cmd)
        self.assertIn("run_one R2_PE4_Gram 4 pe", self.cmd)
        self.assertIn("TOKENIZER_PATH_INFO=${run_dir}/tokenizers/fineweb_8192_bpe.model", self.cmd)

    def test_stale_shell_runner_is_disabled(self):
        shell = (REPO_ROOT / "scripts" / "run_pe_ablation.sh").read_text(encoding="utf-8")
        self.assertIn("deprecated", shell.lower())
        self.assertIn("run_pe_ablation.py", shell)
        self.assertNotIn("pe_ablation_R0_fixed5.py", shell)


@unittest.skipUnless(PY38_PLUS, PY38_SKIP_REASON)
class PeAblationBundleTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_ablation_bundle_uses_only_allowlisted_files(self):
        files = decode_bundle(self.module)
        names = sorted(files)
        self.assertEqual(
            names,
            sorted([
                "cached_challenge_fineweb.py",
                "requirements.txt",
                "tokenizer_specs.json",
                "train_gpt.py",
            ]),
        )
        self.assertFalse(any(name.startswith("results/") or "pe_ablation_" in name for name in names))

    def test_generator_uses_embedded_reviewed_template_not_results_path(self):
        template_path = self.module.TEMPLATE_TRAIN_SCRIPT
        rel_template = template_path.relative_to(REPO_ROOT).as_posix()
        self.assertEqual(rel_template, "scripts/pe_ablation_train_template.py")
        self.assertTrue(template_path.exists())
        self.assertFalse(hasattr(self.module, "REFERENCE_TRAIN_SCRIPT"))
        provenance = self.module.ablation_source_provenance()
        self.assertEqual(provenance["template_path"], str(template_path))
        self.assertRegex(provenance["template_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("source_candidate_path", provenance)
        self.assertNotIn("source_candidate_sha256", provenance)
        self.assertEqual(
            provenance["embedded_source_sha256"],
            "fbb784fdc87115d3cd1878c515a628ecfdfb2d75f3dfa9598a5f210925ac7cea",
        )
        self.assertEqual(provenance["embedded_source_bytes"], 50232)

    def test_template_source_does_not_read_untracked_results_candidate(self):
        original_read_text = Path.read_text

        def guarded_read_text(path, *args, **kwargs):
            self.assertFalse(
                path.as_posix().endswith("results/1809_pr_decoded.py"),
                "template source must not depend on untracked results/1809_pr_decoded.py at runtime",
            )
            return original_read_text(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", guarded_read_text):
            provenance = self.module.ablation_source_provenance()
            source = self.module.build_ablation_train_source()
        self.assertIn("flash_attn_interface", source)
        self.assertEqual(provenance["embedded_source_bytes"], 50232)
        self.assertNotIn("results/1809_pr_decoded.py", str(provenance))

    def test_source_provenance_reports_hashes_line_count_and_entries(self):
        provenance = self.module.ablation_source_provenance()
        self.assertEqual(provenance["template_path"], str(self.module.TEMPLATE_TRAIN_SCRIPT))
        self.assertRegex(provenance["template_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(provenance["embedded_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(provenance["embedded_source_bytes"], 50232)
        self.assertRegex(provenance["generated_payload_sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(provenance["generated_line_count"], 0)
        self.assertEqual(provenance["bundle_entries"], list(self.module.BUNDLE_ALLOWED_ARCNAMES))

    def test_bundled_train_script_is_1809_style_not_root_baseline(self):
        files = decode_bundle(self.module)
        wrapper = files["train_gpt.py"].decode("utf-8")
        source = self.module.decode_ablation_train_wrapper(wrapper)
        root_source = (REPO_ROOT / "train_gpt.py").read_text(encoding="utf-8")
        self.assertNotEqual(source, root_source)
        self.assertNotEqual(wrapper, source)
        self.assertNotIn("Default Simple Baseline run", source)
        self.assertIn("flash_attn_interface", source)
        self.assertIn("vocab_size=int(os.environ.get('VOCAB_SIZE',8192))", source)
        self.assertIn("num_layers=int(os.environ.get('NUM_LAYERS',11))", source)
        self.assertIn("mlp_mult=float(os.environ.get('MLP_MULT',4.))", source)
        self.assertIn("num_loops=int(os.environ.get('NUM_LOOPS',2))", source)
        self.assertIn("parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',7))", source)
        self.assertIn("quantized_model_path='final_model.int6.ptz'", source)
        self.assertIn("gptq_mixed_quantize", source)
        self.assertIn("eval_val_sliding", source)
        self.assertIn("eval_val_ttt", source)
        self.assertIn("MUON_NS_COEFF_TYPE", source)
        self.assertIn("MUON_NS_GRAM", source)
        self.assertIn("_POLAR_NS_COEFFS[-steps:]", source)
        self.assertIn("def _ns_gram", source)
        self.assertIn("PE_ABLATION_CSV_HEADER", source)
        self.assertNotIn("final_model.int8.ptz", source)
        compile(source, "bundled_train_gpt.py", "exec")

    def test_generated_source_accounts_code_bytes_from_runtime_file_size(self):
        files = decode_bundle(self.module)
        wrapper = files["train_gpt.py"].decode("utf-8")
        source = self.module.decode_ablation_train_wrapper(wrapper)
        self.assertIn("os.path.getsize(__file__)", source)
        self.assertNotIn("code_bytes=len(code.encode", source)
        self.assertNotIn("Path(__file__).read_text", source)

    def test_bundled_wrapper_compiles_and_is_small(self):
        files = decode_bundle(self.module)
        wrapper = files["train_gpt.py"].decode("utf-8")
        wrapper_bytes = len(files["train_gpt.py"])
        self.assertLess(wrapper_bytes, self.module.CONSERVATIVE_WRAPPER_BYTE_CAP)
        self.assertIn("exec(compile(_SOURCE,__file__,'exec'),globals())", wrapper)
        compile(wrapper, "train_gpt.py", "exec")

    def test_wrapper_decoder_round_trips_to_full_generated_source(self):
        source = self.module.build_ablation_train_source()
        wrapper = self.module.build_ablation_train_wrapper(source)
        decoded = self.module.decode_ablation_train_wrapper(wrapper)
        self.assertEqual(decoded, source)
        self.assertIn("def _ns_gram", decoded)
        self.assertIn("PE_ABLATION_CSV_HEADER", decoded)
        self.assertIn("gptq_mixed_quantize", decoded)

    def test_dry_run_audit_strings_show_allowlisted_generated_script(self):
        audit = self.module.dry_run_audit_text()
        self.assertIn("archive=train_gpt.py", audit)
        self.assertIn("payload_description=#1809-style GPTQ/int6/Brotli SP8192 ablation payload", audit)
        self.assertIn("exact_upstream_1809=false", audit)
        self.assertIn("template_path=", audit)
        self.assertIn("template_sha256=", audit)
        self.assertIn("embedded_source_sha256=fbb784fdc87115d3cd1878c515a628ecfdfb2d75f3dfa9598a5f210925ac7cea", audit)
        self.assertIn("embedded_source_bytes=50232", audit)
        self.assertNotIn("source_candidate_path=", audit)
        self.assertNotIn("source_candidate_sha256=", audit)
        self.assertNotIn("results/1809_pr_decoded.py", audit)
        self.assertIn("generated_payload_sha256=", audit)
        self.assertIn("generated_line_count=", audit)
        self.assertIn("wrapper_sha256=", audit)
        self.assertIn("wrapper_bytes=", audit)
        self.assertIn("artifact_byte_accounting=wrapper_bytes+final_model.int6.ptz_bytes", audit)
        self.assertIn("historical_model_bytes=", audit)
        self.assertIn("estimated_cap_headroom_min=", audit)
        self.assertIn("estimated_cap_headroom_max=", audit)
        self.assertIn("bundle_entries=train_gpt.py,cached_challenge_fineweb.py,tokenizer_specs.json,requirements.txt", audit)
        self.assertIn("artifact=final_model.int6.ptz", audit)
        self.assertIn(self.module.CSV_HEADER, audit)
        self.assertIn("DATA_PATH_INFO=informational_only", audit)
        self.assertIn("TOKENIZER_PATH_INFO=informational_only", audit)
        self.assertNotIn("records/track_non_record_16mb/2026-04-26_SP8192_PolarExpress_Ablation/train_gpt.py", audit)

    def test_provenance_reports_wrapper_hash_size_and_headroom(self):
        provenance = self.module.ablation_source_provenance()
        self.assertRegex(provenance["wrapper_sha256"], r"^[0-9a-f]{64}$")
        self.assertLess(provenance["wrapper_bytes"], self.module.CONSERVATIVE_WRAPPER_BYTE_CAP)
        self.assertEqual(
            provenance["artifact_byte_accounting"],
            "wrapper_bytes+final_model.int6.ptz_bytes",
        )
        self.assertEqual(
            provenance["historical_model_bytes"],
            list(self.module.HISTORICAL_MODEL_BYTE_REFERENCES),
        )
        expected_min = self.module.ARTIFACT_CAP_BYTES - provenance["wrapper_bytes"] - max(self.module.HISTORICAL_MODEL_BYTE_REFERENCES)
        expected_max = self.module.ARTIFACT_CAP_BYTES - provenance["wrapper_bytes"] - min(self.module.HISTORICAL_MODEL_BYTE_REFERENCES)
        self.assertEqual(provenance["estimated_cap_headroom_min"], expected_min)
        self.assertEqual(provenance["estimated_cap_headroom_max"], expected_max)

    def test_retrieval_buffer_is_at_least_five_minutes(self):
        self.assertGreaterEqual(self.module.RETRIEVAL_BUFFER_SECONDS, 300)


@unittest.skipUnless(PY38_PLUS, PY38_SKIP_REASON)
class PeAblationTrainScriptHooksTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.source = self.module.build_ablation_train_source()

    def test_train_script_has_env_controlled_ns_hooks(self):
        self.assertIn("MUON_NS_COEFF_TYPE", self.source)
        self.assertIn("MUON_NS_GRAM", self.source)
        self.assertIn("_POLAR_NS_COEFFS", self.source)
        self.assertIn("def _ns_gram", self.source)


def load_train_gpt_with_env(**env):
    runner = load_module()
    source = runner.build_ablation_train_source()
    old_env = {key: os.environ.get(key) for key in env}
    old_modules = {key: sys.modules.get(key) for key in (
        "flash_attn_interface",
        "sentencepiece",
        "torch",
        "torch.distributed",
        "torch.nn",
        "torch.nn.functional",
        "torch.nn.parallel",
    )}
    missing = object()
    old_modules = {key: (value if value is not None else missing) for key, value in old_modules.items()}

    class FakeNoGrad:
        def __call__(self, fn=None):
            return fn if fn is not None else self
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTensor:
        def __init__(self, *shape):
            self.shape = tuple(shape)
        def size(self, dim):
            return self.shape[dim]

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.int16 = "int16"
    fake_torch.bool = "bool"
    fake_torch.int8 = "int8"
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.empty = lambda *shape: FakeTensor(*shape)
    fake_torch.compile = lambda fn, *args, **kwargs: fn

    fake_optim = types.SimpleNamespace(Optimizer=type("Optimizer", (), {"__init__": lambda self, params, defaults: None}))
    fake_torch.optim = fake_optim

    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = type("Module", (), {})
    fake_nn.Linear = type("Linear", (fake_nn.Module,), {})
    fake_nn.Parameter = lambda value: value
    fake_nn.Embedding = type("Embedding", (fake_nn.Module,), {})
    fake_nn.ModuleList = list
    fake_functional = types.ModuleType("torch.nn.functional")
    fake_parallel = types.ModuleType("torch.nn.parallel")
    fake_parallel.DistributedDataParallel = type("DistributedDataParallel", (), {})
    fake_dist = types.ModuleType("torch.distributed")
    fake_spm = types.ModuleType("sentencepiece")
    fake_spm.SentencePieceProcessor = type("SentencePieceProcessor", (), {})
    fake_flash = types.ModuleType("flash_attn_interface")
    fake_flash.flash_attn_func = lambda *args, **kwargs: None

    sys.modules.update({
        "flash_attn_interface": fake_flash,
        "sentencepiece": fake_spm,
        "torch": fake_torch,
        "torch.distributed": fake_dist,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_functional,
        "torch.nn.parallel": fake_parallel,
    })
    os.environ.update({key: str(value) for key, value in env.items()})
    try:
        module_name = "train_gpt_under_test"
        sys.modules.pop(module_name, None)
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        exec(compile(source, "generated_train_gpt.py", "exec"), module.__dict__)
        return module
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for key, value in old_modules.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@unittest.skipUnless(PY38_PLUS, PY38_SKIP_REASON)
class PeAblationTrainTelemetryTest(unittest.TestCase):
    def test_selected_ns_coeffs_pe_uses_last_four(self):
        module = load_train_gpt_with_env(MUON_NS_COEFF_TYPE="pe", MUON_NS_GRAM="1")
        self.assertEqual(module._selected_ns_coeffs(4), module._POLAR_NS_COEFFS[-4:])

    def test_selected_ns_coeffs_fixed_repeats_requested_steps(self):
        module = load_train_gpt_with_env(MUON_NS_COEFF_TYPE="fixed", MUON_NS_GRAM="1")
        self.assertEqual(module._selected_ns_coeffs(4), (module._FIXED_NS_COEFFS,) * 4)

    def test_gram_dispatch_threshold_and_env_plumbing(self):
        module = load_train_gpt_with_env(MUON_NS_COEFF_TYPE="fixed", MUON_NS_GRAM="1")
        calls = []

        def fake_gram(G, steps, eps):
            calls.append(("gram", tuple(G.shape), steps))
            return "gram"

        def fake_standard(G, steps, eps):
            calls.append(("standard", tuple(G.shape), steps))
            return "standard"

        module._ns_gram = fake_gram
        module._ns_standard = fake_standard
        self.assertEqual(module.zeropower_via_newtonschulz5(module.torch.empty(2, 4), steps=4), "gram")
        self.assertEqual(module.zeropower_via_newtonschulz5(module.torch.empty(3, 4), steps=4), "standard")
        self.assertEqual(calls, [("gram", (2, 4), 4), ("standard", (3, 4), 4)])

    def test_pe_ablation_csv_formatter_order_and_prefix(self):
        module = load_train_gpt_with_env(MUON_NS_COEFF_TYPE="pe", MUON_NS_GRAM="1")
        row = module.format_pe_ablation_csv(
            run_id="R2_PE4_Gram",
            seed=42,
            ns_steps=4,
            coeff_type="pe",
            gram_ns=True,
            steps_completed=12,
            train_seconds=123.4567,
            optimizer_seconds=7.8912,
            optimizer_pct=6.3901,
            sliding_bpb=1.234567891,
            ttt_bpb=None,
            artifact_bytes=15_999_999,
        )
        fields = next(csv.reader([row]))
        self.assertEqual(fields, [
            "R2_PE4_Gram", "42", "4", "pe", "yes", "12",
            "123.457", "7.891", "6.390", "1.23456789", "N/A", "15999999",
        ])
        self.assertEqual(
            module.PE_ABLATION_CSV_HEADER,
            "run_id,seed,ns_steps,coeff_type,gram_ns,steps_completed,"
            "train_seconds,optimizer_seconds,optimizer_pct,sliding_bpb,ttt_bpb,artifact_bytes",
        )


if __name__ == "__main__":
    unittest.main()
