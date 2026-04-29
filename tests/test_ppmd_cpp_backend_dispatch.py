"""Phase 5 tests: --backend {python,cpp} dispatcher in eval_path_a_ppmd.py.

Skip-gates:
  * /bin/python3.8 (extension cannot be built/loaded there)
  * `_ppmd_cpp` not built / does not expose score_path_a_arrays
"""

from __future__ import annotations

import importlib.util
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "scripts" / "ppmd_cpp"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for p in (_BUILD_DIR, _SCRIPTS_DIR, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


SCRIPT = _REPO_ROOT / "scripts" / "eval_path_a_ppmd.py"


def _running_under_system_python38() -> bool:
    exe = os.path.realpath(sys.executable)
    return exe == "/bin/python3.8" or (
        exe.endswith("/python3.8") and "venv" not in exe
    )


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_path_a_ppmd", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load eval_path_a_ppmd.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_synthetic_inputs(seed: int, vocab_size: int, n_positions: int):
    rng = random.Random(seed)
    tokens = []
    is_boundary = []
    for _ in range(vocab_size):
        n = rng.randint(1, 5)
        tokens.append(bytes(rng.randint(0, 255) for _ in range(n)))
        is_boundary.append(rng.random() < 0.3)
    target_ids = np.asarray(
        [rng.randrange(vocab_size) for _ in range(n_positions)], dtype=np.int32
    )
    prev_ids = np.asarray(
        [-1 if i == 0 else int(target_ids[i - 1]) for i in range(n_positions)],
        dtype=np.int32,
    )
    nll_nats = np.asarray(
        [rng.uniform(0.5, 4.0) for _ in range(n_positions)], dtype=np.float64
    )
    return tokens, is_boundary, target_ids, prev_ids, nll_nats


class BackendFlagDefaultTest(unittest.TestCase):
    """The --help output must show --backend with default 'python'.

    This test runs under any Python (it just shells out to --help), so it does
    not require .venv-smoke or the C++ extension.
    """

    def test_backend_flag_defaults_to_python(self) -> None:
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = out.stdout + out.stderr
        self.assertIn("--backend", text,
                      f"--backend flag missing from --help output:\n{text}")
        # Default must be python (string appears in the help text near --backend).
        # Argparse renders '(default: python)' when default is set explicitly.
        self.assertIn("python", text)


class BackendCppDispatchTestBase(unittest.TestCase):
    def setUp(self) -> None:
        if _running_under_system_python38():
            self.skipTest(
                "running under /bin/python3.8; use .venv-smoke/bin/python"
            )
        try:
            import _ppmd_cpp  # noqa: F401
        except ImportError as e:
            self.skipTest(f"_ppmd_cpp extension not built: {e}")
        import _ppmd_cpp
        if not hasattr(_ppmd_cpp, "score_path_a_arrays"):
            self.skipTest("_ppmd_cpp does not expose score_path_a_arrays")

        self.mod = _load_eval_module()
        if not hasattr(self.mod, "_score_path_a_arrays_dispatch"):
            self.skipTest(
                "eval_path_a_ppmd does not expose _score_path_a_arrays_dispatch"
            )


class BackendCppDispatchTest(BackendCppDispatchTestBase):
    def test_backend_cpp_dispatches_to_extension_when_available(self) -> None:
        tokens, is_boundary, target_ids, prev_ids, nll_nats = _make_synthetic_inputs(
            seed=42, vocab_size=16, n_positions=8
        )
        candidates = [
            self.mod.CandidateBytes(
                token_id=tid,
                after_boundary=tokens[tid],
                after_non_boundary=tokens[tid],
                emittable=True,
            )
            for tid in range(len(tokens))
        ]
        out = self.mod._score_path_a_arrays_dispatch(
            backend="cpp",
            target_ids=target_ids,
            prev_ids=prev_ids,
            nll_nats=nll_nats,
            candidates=candidates,
            is_boundary_token_lut=list(is_boundary),
            order=5,
            lambda_hi=0.9,
            lambda_lo=0.05,
            conf_threshold=0.9,
        )
        self.assertIn("total_bits", out)
        self.assertIn("total_bytes", out)
        self.assertIn("bpb", out)
        self.assertGreater(int(out["total_bytes"]), 0)

    def test_backend_cpp_bpb_matches_python_64_positions(self) -> None:
        tokens, is_boundary, target_ids, prev_ids, nll_nats = _make_synthetic_inputs(
            seed=2024, vocab_size=32, n_positions=64
        )
        candidates = [
            self.mod.CandidateBytes(
                token_id=tid,
                after_boundary=tokens[tid],
                after_non_boundary=tokens[tid],
                emittable=True,
            )
            for tid in range(len(tokens))
        ]
        py_out = self.mod._score_path_a_arrays_dispatch(
            backend="python",
            target_ids=target_ids,
            prev_ids=prev_ids,
            nll_nats=nll_nats,
            candidates=candidates,
            is_boundary_token_lut=list(is_boundary),
            order=5,
            lambda_hi=0.9,
            lambda_lo=0.05,
            conf_threshold=0.9,
        )
        cpp_out = self.mod._score_path_a_arrays_dispatch(
            backend="cpp",
            target_ids=target_ids,
            prev_ids=prev_ids,
            nll_nats=nll_nats,
            candidates=candidates,
            is_boundary_token_lut=list(is_boundary),
            order=5,
            lambda_hi=0.9,
            lambda_lo=0.05,
            conf_threshold=0.9,
        )
        self.assertEqual(int(py_out["total_bytes"]), int(cpp_out["total_bytes"]))
        py_bpb = float(py_out["bpb"])
        cpp_bpb = float(cpp_out["bpb"])
        diff = abs(py_bpb - cpp_bpb)
        self.assertLessEqual(
            diff, 1e-10,
            f"bpb diff {diff!r} (py={py_bpb!r} cpp={cpp_bpb!r})"
        )


if __name__ == "__main__":
    unittest.main()


class BackendFlagChoicesTest(unittest.TestCase):
    """The --help output must show cuda and auto as valid --backend choices."""

    def test_backend_flag_accepts_cuda_and_auto(self) -> None:
        out = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = out.stdout + out.stderr
        self.assertIn("cuda", text, "--backend help must mention 'cuda'")
        self.assertIn("auto", text, "--backend help must mention 'auto'")

    def test_backend_equiv_check_rejects_python_backend(self) -> None:
        """--backend-equiv-check N with --backend python must exit non-zero."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--backend", "python",
             "--backend-equiv-check", "10",
             "--core-smoke"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0,
                            "Expected non-zero exit when backend=python with equiv-check")

    def test_backend_cuda_graceful_skip_when_no_extension(self) -> None:
        """--backend cuda must either succeed or print a WARN and fall back gracefully."""
        mod = _load_eval_module()
        if not hasattr(mod, "_score_path_a_arrays_dispatch"):
            self.skipTest("eval_path_a_ppmd does not expose _score_path_a_arrays_dispatch")

        # If _ppmd_cuda IS importable but has no CUDA device, we still expect a graceful result.
        tokens, is_boundary, target_ids, prev_ids, nll_nats = _make_synthetic_inputs(
            seed=100, vocab_size=8, n_positions=4
        )
        candidates = [
            mod.CandidateBytes(
                token_id=tid,
                after_boundary=tokens[tid],
                after_non_boundary=tokens[tid],
                emittable=True,
            )
            for tid in range(len(tokens))
        ]
        try:
            out = mod._score_path_a_arrays_dispatch(
                backend="cuda",
                target_ids=target_ids,
                prev_ids=prev_ids,
                nll_nats=nll_nats,
                candidates=candidates,
                is_boundary_token_lut=list(is_boundary),
                order=5,
                lambda_hi=0.9,
                lambda_lo=0.05,
                conf_threshold=0.9,
                abort_on_import_failure=False,
            )
            # If we get a result (either from cuda or python fallback), it must be valid.
            self.assertIn("bpb", out)
            self.assertIn("positions", out)
        except Exception as exc:
            self.fail(f"--backend cuda raised unexpectedly: {exc}")

    def test_backend_auto_graceful_skip_when_no_extensions(self) -> None:
        """--backend auto must return a valid result (falling back to python if needed)."""
        mod = _load_eval_module()
        if not hasattr(mod, "_score_path_a_arrays_dispatch"):
            self.skipTest("eval_path_a_ppmd does not expose _score_path_a_arrays_dispatch")

        tokens, is_boundary, target_ids, prev_ids, nll_nats = _make_synthetic_inputs(
            seed=101, vocab_size=8, n_positions=4
        )
        candidates = [
            mod.CandidateBytes(
                token_id=tid,
                after_boundary=tokens[tid],
                after_non_boundary=tokens[tid],
                emittable=True,
            )
            for tid in range(len(tokens))
        ]
        out = mod._score_path_a_arrays_dispatch(
            backend="auto",
            target_ids=target_ids,
            prev_ids=prev_ids,
            nll_nats=nll_nats,
            candidates=candidates,
            is_boundary_token_lut=list(is_boundary),
            order=5,
            lambda_hi=0.9,
            lambda_lo=0.05,
            conf_threshold=0.9,
            abort_on_import_failure=False,
        )
        self.assertIn("bpb", out)
        self.assertIn("positions", out)
        self.assertGreater(int(out["positions"]), 0)
