"""Smoke tests for the `_ppmd_cuda` pybind11 extension (Phase 2).

These tests skip cleanly when _ppmd_cuda cannot be imported (e.g. CPU-only
HPC where nvcc is unavailable and the CUDA extension hasn't been built).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "scripts" / "ppmd_cpp"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

# Skip entire module if the CUDA extension isn't present.
_ppmd_cuda = pytest.importorskip("_ppmd_cuda")


def test_cuda_module_imports():
    """_ppmd_cuda imported successfully (already guaranteed by importorskip)."""
    assert _ppmd_cuda is not None


def test_cuda_module_reexports_cpp_surface():
    """_ppmd_cuda re-exports the full CPU surface from _ppmd_cpp."""
    assert hasattr(_ppmd_cuda, "VirtualPPMDState")
    assert hasattr(_ppmd_cuda, "PPMDState")
    assert hasattr(_ppmd_cuda, "Trie")
    assert callable(_ppmd_cuda.version)
    assert _ppmd_cuda.version() == "0.0.1"
    assert hasattr(_ppmd_cuda, "score_path_a_arrays")
    assert hasattr(_ppmd_cuda, "trie_partial_z_and_target")


def test_cuda_runtime_available_when_built():
    """CUDA runtime reports a valid version; device probe is consistent."""
    rt_ver = _ppmd_cuda.cuda.runtime_version()
    assert rt_ver > 0, "Expected positive CUDA runtime version"

    n_devices = _ppmd_cuda.cuda.device_count()
    assert isinstance(n_devices, int)
    assert n_devices >= 0

    if n_devices == 0:
        pytest.skip("No CUDA device available on this host — probe tests skipped")

    assert _ppmd_cuda.cuda.available() is True
    major = _ppmd_cuda.cuda.compute_capability_major(0)
    assert major >= 7, (
        "Expected compute capability >= 7.x on a modern GPU, got {}".format(major)
    )


def test_cuda_byte_probs_match_cpp_smoke():
    """PPMDState (from re-exported CPU surface) returns 256 probs summing to 1."""
    state = _ppmd_cuda.PPMDState(order=5)
    probs = state.byte_probs()
    assert len(probs) == 256
    total = float(sum(probs))
    assert abs(total - 1.0) < 1e-9, "byte_probs() should sum to 1.0, got {}".format(total)
