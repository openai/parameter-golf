"""Phase 3 CUDA kernel equivalence tests.

Verifies that _ppmd_cuda.cuda.byte_probs_batched produces results
bit-for-bit identical (within IEEE 754 double precision, ≤ 1e-15) to the
CPU reference path via _ppmd_cpp.

Skips cleanly when:
  - _ppmd_cuda cannot be imported (no CUDA build), or
  - cuda.device_count() == 0 (no GPU on this host).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE.parent / "scripts" / "ppmd_cpp"
for p in (BUILD_DIR,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Skip module if _ppmd_cuda is not built.
_ppmd_cuda = pytest.importorskip("_ppmd_cuda")

# Skip module if no CUDA device is available.
if _ppmd_cuda.cuda.device_count() == 0:
    pytest.skip("no CUDA device available", allow_module_level=True)

import _ppmd_cpp  # noqa: E402  (must import after path setup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state_with_windows(n_windows: int, seed: int = 42):
    """Return (PPMDState, list[bytes]) with diverse short windows."""
    rng = random.Random(seed)
    state = _ppmd_cpp.PPMDState(order=5)
    payload = bytes(rng.randrange(256) for _ in range(20_000))
    state.update_bytes(payload)

    windows = []
    for _ in range(n_windows):
        wlen = rng.choice([0, 1, 2, 3, 4, 5, 6])
        w = bytes(rng.randrange(256) for _ in range(wlen))
        windows.append(w)
    return state, windows


def _cpu_reference(state, windows) -> np.ndarray:
    """Compute byte_probs via the CPU reference path (VirtualPPMDState)."""
    cpu = np.zeros((len(windows), 256), dtype=np.float64)
    for i, w in enumerate(windows):
        v = state.clone_virtual()
        for b in w:
            v = v.fork_and_update(b)
        cpu[i, :] = list(v.byte_probs())
    return cpu


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestByteProbsKernelEquivalence:

    def test_byte_probs_match_cpp_reference(self):
        """CUDA and CPU byte_probs agree to ≤ 1e-15 across 200 windows."""
        state, windows = _build_state_with_windows(200, seed=42)
        cpu_probs  = _cpu_reference(state, windows)
        cuda_probs = np.asarray(_ppmd_cuda.cuda.byte_probs_batched(state, windows))

        assert cuda_probs.shape == (200, 256), (
            f"unexpected shape {cuda_probs.shape}"
        )
        diff = np.abs(cuda_probs - cpu_probs)
        max_diff = float(diff.max())
        assert max_diff <= 1e-15, (
            f"max abs diff = {max_diff} exceeds 1e-15\n"
            f"worst window idx = {int(diff.max(axis=1).argmax())}"
        )

    def test_byte_probs_sum_to_one(self):
        """Each CUDA output row sums to 1.0 within 1e-12."""
        state, windows = _build_state_with_windows(200, seed=7)
        cuda_probs = np.asarray(_ppmd_cuda.cuda.byte_probs_batched(state, windows))
        sums       = cuda_probs.sum(axis=1)
        max_err    = float(np.abs(sums - 1.0).max())
        assert max_err <= 1e-12, f"max |sum-1| = {max_err}"

    def test_empty_window_matches_base_byte_probs(self):
        """A single empty window must match state.byte_probs() (the base)."""
        state, _ = _build_state_with_windows(1, seed=99)
        cpu_base  = np.asarray(list(state.byte_probs()))
        cuda_probs = np.asarray(
            _ppmd_cuda.cuda.byte_probs_batched(state, [b""])
        )
        assert cuda_probs.shape == (1, 256)
        max_diff = float(np.abs(cuda_probs[0] - cpu_base).max())
        assert max_diff <= 1e-15, f"empty window max diff = {max_diff}"

    def test_zero_windows_returns_empty_array(self):
        """An empty window list returns an (0, 256) array."""
        state, _ = _build_state_with_windows(1, seed=0)
        result = np.asarray(_ppmd_cuda.cuda.byte_probs_batched(state, []))
        assert result.shape == (0, 256)

    def test_virtual_state_passthrough(self):
        """Passing a VirtualPPMDState (clone_virtual result) also works."""
        state, windows = _build_state_with_windows(50, seed=11)
        vstate = state.clone_virtual()
        # CPU reference: all windows computed from the virtual state.
        cpu = np.zeros((len(windows), 256), dtype=np.float64)
        for i, w in enumerate(windows):
            v = vstate
            for b in w:
                v = v.fork_and_update(b)
            cpu[i, :] = list(v.byte_probs())
        # VirtualPPMDState (no overlay) should be equivalent to PPMDState.
        cuda_probs = np.asarray(
            _ppmd_cuda.cuda.byte_probs_batched(vstate, windows)
        )
        max_diff = float(np.abs(cuda_probs - cpu).max())
        assert max_diff <= 1e-15, (
            f"VirtualPPMDState passthrough max diff = {max_diff}"
        )

    def test_larger_batch_correctness(self):
        """500 windows still pass the 1e-15 bar."""
        state, windows = _build_state_with_windows(500, seed=13)
        cpu_probs  = _cpu_reference(state, windows)
        cuda_probs = np.asarray(_ppmd_cuda.cuda.byte_probs_batched(state, windows))
        max_diff = float(np.abs(cuda_probs - cpu_probs).max())
        assert max_diff <= 1e-15, f"500-window max diff = {max_diff}"
