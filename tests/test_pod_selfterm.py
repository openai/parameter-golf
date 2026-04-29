"""Tests for pod-side self-termination logic.

Validates that the self-termination shell preamble and environment
variable helpers produce correct, safe payloads without launching
any real RunPod pod.
"""
import os
import re
import unittest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SELFTERM_PATH = REPO_ROOT / "scripts" / "pod_selfterm.py"
SAFE_PATH = REPO_ROOT / "scripts" / "runpod_safe.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Tests for scripts/pod_selfterm.py
# ===========================================================================
class TestSelfTermConstants(unittest.TestCase):
    """Hard-deadline constant and retrieval buffer are explicit."""

    def setUp(self):
        self.mod = _load("pod_selfterm", SELFTERM_PATH)

    def test_hard_deadline_is_720_seconds(self):
        self.assertEqual(self.mod.POD_HARD_DEADLINE_SECONDS, 720)

    def test_retrieval_buffer_documented(self):
        self.assertGreaterEqual(self.mod.RETRIEVAL_BUFFER_SECONDS, 60,
                                "At least 60 s retrieval buffer expected")


class TestSelfTermEnvDict(unittest.TestCase):
    """selfterm_env_dict returns the right env-var mapping."""

    def setUp(self):
        self.mod = _load("pod_selfterm", SELFTERM_PATH)

    def test_contains_deadline_key(self):
        d = self.mod.selfterm_env_dict("fake-key")
        self.assertIn("PGOLF_HARD_DEADLINE_SEC", d)
        self.assertEqual(d["PGOLF_HARD_DEADLINE_SEC"], "720")

    def test_contains_api_key(self):
        d = self.mod.selfterm_env_dict("test-api-key-123")
        self.assertEqual(d["RUNPOD_API_KEY"], "test-api-key-123")

    def test_custom_deadline(self):
        d = self.mod.selfterm_env_dict("k", deadline_sec=600)
        self.assertEqual(d["PGOLF_HARD_DEADLINE_SEC"], "600")

    def test_returns_plain_dict(self):
        d = self.mod.selfterm_env_dict("k")
        self.assertIsInstance(d, dict)


class TestSelfTermBashPreamble(unittest.TestCase):
    """The generated bash preamble must contain critical safety elements."""

    def setUp(self):
        self.mod = _load("pod_selfterm", SELFTERM_PATH)
        self.snippet = self.mod.selfterm_bash_preamble()

    def test_is_nonempty_string(self):
        self.assertIsInstance(self.snippet, str)
        self.assertGreater(len(self.snippet), 50)

    def test_reads_deadline_env_var(self):
        self.assertIn("PGOLF_HARD_DEADLINE_SEC", self.snippet)

    def test_default_720_fallback(self):
        # The snippet should fall back to 720 if env var is unset
        self.assertIn("720", self.snippet)

    def test_uses_runpod_pod_id(self):
        self.assertIn("RUNPOD_POD_ID", self.snippet)

    def test_uses_runpod_api_key(self):
        self.assertIn("RUNPOD_API_KEY", self.snippet)

    def test_calls_terminate_mutation(self):
        self.assertIn("podTerminate", self.snippet)

    def test_calls_graphql_endpoint(self):
        self.assertIn("api.runpod.io/graphql", self.snippet)

    def test_runs_in_background(self):
        # Must end the function call with & to background it
        self.assertIn("&", self.snippet)

    def test_contains_sleep(self):
        self.assertIn("sleep", self.snippet)

    def test_has_kill_1_fallback(self):
        # Fallback: kill PID 1 to stop the container
        self.assertIn("kill 1", self.snippet)

    def test_logs_arming_message(self):
        self.assertRegex(self.snippet, re.compile(r"self.termination armed", re.IGNORECASE))


class TestSelfTermBashPreambleShellSafety(unittest.TestCase):
    """The preamble must be safe to embed at the top of any bash script."""

    def setUp(self):
        self.mod = _load("pod_selfterm", SELFTERM_PATH)
        self.snippet = self.mod.selfterm_bash_preamble()

    def test_no_set_e_interaction(self):
        # The preamble should not call 'set -e' because the outer
        # wrapper may set it later; it must not break under set -e.
        # Conversely, any internal failures should be trapped.
        # Just confirm the snippet doesn't set -e itself.
        lines = self.snippet.strip().splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped == "set -e":
                self.fail("Preamble must not set -e; it would break background subshells")

    def test_subshell_or_function_isolation(self):
        # The self-term logic should be isolated in a subshell ( ... ) &
        # or a function so it doesn't pollute the main script namespace.
        self.assertTrue(
            "(" in self.snippet or "_pgolf_selfterm" in self.snippet,
            "Self-term logic should be isolated in subshell or function"
        )


# ===========================================================================
# Tests for runpod_safe.py integration
# ===========================================================================
class TestRunpodSafeIntegration(unittest.TestCase):
    """Verify runpod_safe.py's create_pod and launch_job wire self-term."""

    def setUp(self):
        # We can't import runpod_safe directly (it imports websocket etc.)
        # Instead, read the source and check for integration markers.
        self.source = SAFE_PATH.read_text()

    def test_create_pod_passes_api_key_env(self):
        self.assertIn("RUNPOD_API_KEY", self.source)

    def test_create_pod_passes_deadline_env(self):
        self.assertIn("PGOLF_HARD_DEADLINE_SEC", self.source)

    def test_job_wrapper_includes_selfterm(self):
        # The launch_job function should inject the selfterm preamble
        self.assertIn("selfterm_bash_preamble", self.source)

    def test_imports_pod_selfterm(self):
        self.assertIn("pod_selfterm", self.source)


class TestRunpodSafeMinuteValidation(unittest.TestCase):
    """runpod_safe enforces a retrieval buffer inside the hard deadline."""

    def setUp(self):
        self.mod = _load("runpod_safe_validation", SAFE_PATH)

    def test_accepts_10_minutes(self):
        self.assertIsNone(self.mod._validate_max_minutes(10))

    def test_rejects_11_minutes(self):
        with self.assertRaises(ValueError):
            self.mod._validate_max_minutes(11)


# ===========================================================================
# Tests for runpod_launch.py integration
# ===========================================================================
class TestRunpodLaunchIntegration(unittest.TestCase):
    """Verify runpod_launch.py passes self-term env vars."""

    def setUp(self):
        launch_path = REPO_ROOT / "scripts" / "runpod_launch.py"
        self.source = launch_path.read_text()

    def test_passes_deadline_env(self):
        self.assertIn("PGOLF_HARD_DEADLINE_SEC", self.source)

    def test_passes_api_key_env(self):
        # Should pass RUNPOD_API_KEY to the pod env
        # (may reference it through pod_selfterm or directly)
        self.assertIn("RUNPOD_API_KEY", self.source)

    def test_imports_pod_selfterm(self):
        self.assertIn("pod_selfterm", self.source)


if __name__ == "__main__":
    unittest.main()
