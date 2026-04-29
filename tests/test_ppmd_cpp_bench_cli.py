"""Phase 4 tests: bench CLI for the Path A PPM-D C++ backend.

Skip-gates (mirroring Phase 1+2+3 tests):
  * /bin/python3.8 (no Python.h / no built extension)
  * `_ppmd_cpp` not built

The first test (--help parsing) does NOT require the extension and runs anywhere.
The second/third tests (synthetic run) require `_ppmd_cpp` and skip cleanly
if it is missing. None of these tests submit SLURM jobs or require a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCH_PY = _REPO_ROOT / "scripts" / "ppmd_cpp" / "bench_cpu.py"
_BUILD_DIR = _REPO_ROOT / "scripts" / "ppmd_cpp"


def _venv_python() -> str:
    """Return the .venv-smoke python if present, else current sys.executable."""
    venv = _REPO_ROOT / ".venv-smoke" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _extension_available() -> bool:
    py = _venv_python()
    proc = subprocess.run(
        [py, "-c", "import sys; sys.path.insert(0, %r); import _ppmd_cpp" % str(_BUILD_DIR)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


class TestBenchCpuCliParses(unittest.TestCase):
    def test_bench_cpu_cli_parses(self) -> None:
        self.assertTrue(_BENCH_PY.exists(), f"missing {_BENCH_PY}")
        py = _venv_python()
        proc = subprocess.run(
            [py, str(_BENCH_PY), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0,
                         f"--help exit={proc.returncode}\nstderr={proc.stderr}")
        out = proc.stdout
        for flag in (
            "--mode",
            "--positions",
            "--vocab",
            "--avg-bytes-per-token",
            "--threads",
            "--seed",
            "--results-dir",
            "--results-name",
            "--no-write",
            "--prefix-slice-path",
        ):
            self.assertIn(flag, out, f"missing flag {flag} in --help output")


class TestBenchCpuSynthetic(unittest.TestCase):
    def setUp(self) -> None:
        if not _extension_available():
            self.skipTest("_ppmd_cpp extension not built (.venv-smoke)")

    def _run_bench(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        py = _venv_python()
        env = os.environ.copy()
        # Ensure the bench script can locate the built extension.
        env["PYTHONPATH"] = str(_BUILD_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [py, str(_BENCH_PY), *args],
            capture_output=True,
            text=True,
            cwd=str(cwd or _REPO_ROOT),
            env=env,
        )

    def test_bench_cpu_no_write_skips_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proc = self._run_bench(
                "--mode", "synthetic",
                "--positions", "16",
                "--vocab", "64",
                "--seed", "0",
                "--no-write",
                "--results-dir", td,
            )
            self.assertEqual(proc.returncode, 0,
                             f"exit={proc.returncode}\nstderr={proc.stderr}")
            # Stdout should contain a single JSON object.
            stripped = proc.stdout.strip().splitlines()
            self.assertTrue(stripped, "no stdout from bench")
            payload = json.loads(stripped[-1])
            for key in (
                "mode", "positions", "vocab", "threads",
                "wallclock_seconds", "total_bits", "total_bytes", "bpb",
                "probes_per_second_estimate", "projected_full_eval_seconds",
            ):
                self.assertIn(key, payload, f"missing key {key} in {payload}")
            self.assertEqual(payload["mode"], "synthetic")
            self.assertEqual(payload["positions"], 16)
            self.assertEqual(payload["vocab"], 64)
            self.assertGreater(payload["wallclock_seconds"], 0.0)
            self.assertGreater(payload["probes_per_second_estimate"], 0.0)
            self.assertGreater(payload["projected_full_eval_seconds"], 0.0)
            # --no-write: no result files should be written.
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_bench_cpu_synthetic_writes_file_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            name = "unit_smoke"
            proc = self._run_bench(
                "--mode", "synthetic",
                "--positions", "16",
                "--vocab", "64",
                "--seed", "0",
                "--results-dir", td,
                "--results-name", name,
            )
            self.assertEqual(proc.returncode, 0,
                             f"exit={proc.returncode}\nstderr={proc.stderr}")
            out_path = Path(td) / f"{name}.json"
            self.assertTrue(out_path.exists(), f"expected {out_path}")
            disk = json.loads(out_path.read_text())
            stdout_payload = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(disk, stdout_payload,
                             "results file content should match stdout JSON")


if __name__ == "__main__":
    unittest.main()
