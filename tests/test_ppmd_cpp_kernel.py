"""Phase 2 equivalence tests: C++ PPM-D byte-prob kernel vs Python reference.

These tests are skipped cleanly when:
  * Running under /bin/python3.8 (no Python.h / no built extension), or
  * The `_ppmd_cpp` extension is not built, or
  * The extension is built but does not yet expose the `PPMDState` class
    (e.g. between Phase 1 and Phase 2 builds).

The Python reference impl in scripts/eval_path_a_ppmd.py is the conformance
contract. We assert byte-prob agreement to >= 1e-15 over a fuzz set, and digest
equality after a 1000-byte feed.
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "scripts" / "ppmd_cpp"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for p in (_BUILD_DIR, _SCRIPTS_DIR, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _running_under_system_python38() -> bool:
    exe = os.path.realpath(sys.executable)
    return exe == "/bin/python3.8" or (
        exe.endswith("/python3.8") and "venv" not in exe
    )


class PpmdCppKernelTestBase(unittest.TestCase):
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

        if not hasattr(_ppmd_cpp, "PPMDState"):
            self.skipTest(
                "_ppmd_cpp does not yet expose PPMDState (Phase 2 not built)"
            )
        # Python reference must import cleanly.
        try:
            from scripts.eval_path_a_ppmd import PPMDState as PyPPMDState  # noqa: F401
        except Exception as e:
            self.skipTest(f"cannot import Python reference PPMDState: {e}")


class TestPPMDCppKernel(PpmdCppKernelTestBase):
    def test_ppmd_cpp_byte_prob_matches_python_reference(self) -> None:
        """200 contexts of length 0..200, each compared across all 256 bytes."""

        import _ppmd_cpp
        from scripts.eval_path_a_ppmd import PPMDState as PyPPMDState

        rng = random.Random(0xC0FFEE)
        max_diff = 0.0
        max_self_diff = 0.0
        for ctx_idx in range(200):
            ctx_len = ctx_idx  # 0..199
            seed_bytes = bytes(rng.randint(0, 255) for _ in range(ctx_len))
            py_state = PyPPMDState(order=5)
            cpp_state = _ppmd_cpp.PPMDState(5)
            py_state.update_bytes(seed_bytes)
            cpp_state.update_bytes(seed_bytes)

            py_probs = py_state.byte_probs()
            cpp_probs = cpp_state.byte_probs()
            self.assertEqual(len(cpp_probs), 256)
            for b in range(256):
                py_p = float(py_probs[b])
                cpp_p_full = float(cpp_probs[b])
                cpp_p_single = float(cpp_state.byte_prob(b))
                d_py = abs(py_p - cpp_p_full)
                d_self = abs(cpp_p_full - cpp_p_single)
                if d_py > max_diff:
                    max_diff = d_py
                if d_self > max_self_diff:
                    max_self_diff = d_self
                self.assertLess(
                    d_py,
                    1e-15,
                    f"ctx_idx={ctx_idx} byte={b}: py={py_p!r} cpp_full={cpp_p_full!r} diff={d_py!r}",
                )
                self.assertLess(
                    d_self,
                    1e-15,
                    f"ctx_idx={ctx_idx} byte={b}: cpp_full={cpp_p_full!r} cpp_single={cpp_p_single!r} diff={d_self!r}",
                )
        # Stash for reporting
        self.__class__._max_py_diff = max_diff
        self.__class__._max_self_diff = max_self_diff

    def test_ppmd_cpp_byte_probs_sums_to_one(self) -> None:
        import _ppmd_cpp

        rng = random.Random(42)
        cpp_state = _ppmd_cpp.PPMDState(5)
        cpp_state.update_bytes(bytes(rng.randint(0, 255) for _ in range(1000)))
        probs = cpp_state.byte_probs()
        s = float(sum(probs))
        self.assertAlmostEqual(s, 1.0, places=12)

    def test_ppmd_cpp_state_digest_matches_python(self) -> None:
        import _ppmd_cpp
        from scripts.eval_path_a_ppmd import PPMDState as PyPPMDState

        rng = random.Random(123)
        data = bytes(rng.randint(0, 255) for _ in range(1000))
        py_state = PyPPMDState(order=5)
        cpp_state = _ppmd_cpp.PPMDState(5)
        py_state.update_bytes(data)
        cpp_state.update_bytes(data)
        self.assertEqual(py_state.state_digest(), cpp_state.state_digest())

    def test_ppmd_cpp_virtual_fork_and_update_does_not_mutate_base(self) -> None:
        import _ppmd_cpp

        rng = random.Random(7)
        cpp_state = _ppmd_cpp.PPMDState(5)
        cpp_state.update_bytes(bytes(rng.randint(0, 255) for _ in range(500)))
        base_digest = cpp_state.state_digest()

        virtual = cpp_state.clone_virtual()
        for _ in range(50):
            b = rng.randint(0, 255)
            virtual = virtual.fork_and_update(b)
            # Probe the virtual to make sure its lookups don't mutate base.
            _ = virtual.byte_probs()
            _ = virtual.byte_prob(b)

        self.assertEqual(cpp_state.state_digest(), base_digest)

    def test_ppmd_cpp_score_first_digest_invariance(self) -> None:
        import _ppmd_cpp

        rng = random.Random(99)
        cpp_state = _ppmd_cpp.PPMDState(5)
        cpp_state.update_bytes(bytes(rng.randint(0, 255) for _ in range(500)))
        before = cpp_state.state_digest()
        for _ in range(100):
            cpp_state.byte_probs()
            for b in range(256):
                cpp_state.byte_prob(b)
        after = cpp_state.state_digest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
