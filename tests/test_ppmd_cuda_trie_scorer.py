"""Phase 4 tests: CUDA trie scorer — trie_partial_z_and_target_batched and
score_path_a_arrays_cuda.

Skip-gates:
  * _ppmd_cuda not built / no CUDA device (hardware not available on this HPC)
  * _ppmd_cpp not built (needed for reference comparison)
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "scripts" / "ppmd_cpp"
for p in (_BUILD_DIR,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _try_import_cuda():
    try:
        import _ppmd_cuda  # type: ignore[import-not-found]
        return _ppmd_cuda
    except ImportError:
        return None


def _try_import_cpp():
    try:
        import _ppmd_cpp  # type: ignore[import-not-found]
        return _ppmd_cpp
    except ImportError:
        return None


def _make_synthetic_state(seed: int = 42, n_bytes: int = 20000, order: int = 5):
    cuda = _try_import_cuda()
    if cuda is None:
        return None
    rng = random.Random(seed)
    state = cuda.PPMDState(order=order)
    state.update_bytes(bytes(rng.randrange(256) for _ in range(n_bytes)))
    return state


def _make_trie_and_candidates(rng, vocab_size: int = 100, min_len: int = 1, max_len: int = 4):
    """Build a ppmd::Trie and numpy arrays for score_path_a_arrays_cuda."""
    cpp = _try_import_cpp()
    if cpp is None:
        return None
    token_bytes_list = [
        bytes(rng.randrange(256) for _ in range(rng.randint(min_len, max_len)))
        for _ in range(vocab_size)
    ]
    trie = cpp.Trie()
    for tid, tb in enumerate(token_bytes_list):
        trie.insert(tid, tb)

    bnd_flat = b"".join(token_bytes_list)
    bnd_off = np.zeros(vocab_size + 1, dtype=np.int32)
    for i, tb in enumerate(token_bytes_list):
        bnd_off[i + 1] = bnd_off[i] + len(tb)
    nbnd_flat = bnd_flat
    nbnd_off = bnd_off.copy()
    emit = np.ones(vocab_size, dtype=np.uint8)
    isb = np.zeros(vocab_size, dtype=np.uint8)
    return trie, token_bytes_list, bnd_flat, bnd_off, nbnd_flat, nbnd_off, emit, isb


class TriePartialZAndTargetBatchedTest(unittest.TestCase):
    """Tests for _ppmd_cuda.cuda.trie_partial_z_and_target_batched."""

    def setUp(self):
        cuda = _try_import_cuda()
        if cuda is None:
            self.skipTest("_ppmd_cuda not built")
        if cuda.cuda.device_count() == 0:
            self.skipTest("no CUDA device available")
        cpp = _try_import_cpp()
        if cpp is None:
            self.skipTest("_ppmd_cpp not built (needed for reference comparison)")
        if not hasattr(cpp, "trie_partial_z_and_target"):
            self.skipTest("_ppmd_cpp does not expose trie_partial_z_and_target")
        self.cuda = cuda
        self.cpp = cpp

    def _make_state(self, seed: int = 42):
        rng = random.Random(seed)
        state = self.cuda.PPMDState(order=5)
        state.update_bytes(bytes(rng.randrange(256) for _ in range(10000)))
        return state

    def test_trie_partial_z_and_target_matches_cpp_single_position(self):
        """CUDA trie_partial_z_and_target_batched must match cpp reference for a single target."""
        rng = random.Random(1111)
        state = self._make_state(seed=1111)
        vocab_size = 64
        token_bytes_list = [
            bytes(rng.randrange(256) for _ in range(rng.randint(1, 4)))
            for _ in range(vocab_size)
        ]
        trie = self.cpp.Trie()
        for tid, tb in enumerate(token_bytes_list):
            trie.insert(tid, tb)

        target_ids = [0, 7, 31, vocab_size - 1]
        for target_id in target_ids:
            with self.subTest(target_id=target_id):
                # CPU reference
                cpp_z, cpp_tq = self.cpp.trie_partial_z_and_target(state, trie, target_id)
                # CUDA
                result = self.cuda.cuda.trie_partial_z_and_target_batched(
                    state, trie, np.array([target_id], dtype=np.int32)
                )
                self.assertEqual(result.shape, (1, 2))
                cuda_z = float(result[0, 0])
                cuda_tq = float(result[0, 1])
                z_diff = abs(cuda_z - cpp_z)
                tq_diff = abs(cuda_tq - cpp_tq)
                self.assertLessEqual(
                    z_diff, 1e-12,
                    f"Z mismatch for target_id={target_id}: cuda={cuda_z} cpp={cpp_z} diff={z_diff}",
                )
                self.assertLessEqual(
                    tq_diff, 1e-12,
                    f"target_q mismatch for target_id={target_id}: cuda={cuda_tq} cpp={cpp_tq} diff={tq_diff}",
                )

    def test_trie_partial_batched_all_targets_match_cpp(self):
        """Batch query for all token IDs must match cpp one-by-one."""
        rng = random.Random(2222)
        state = self._make_state(seed=2222)
        vocab_size = 32
        token_bytes_list = [
            bytes(rng.randrange(256) for _ in range(rng.randint(1, 3)))
            for _ in range(vocab_size)
        ]
        trie = self.cpp.Trie()
        for tid, tb in enumerate(token_bytes_list):
            trie.insert(tid, tb)

        target_ids = np.arange(vocab_size, dtype=np.int32)
        result = self.cuda.cuda.trie_partial_z_and_target_batched(state, trie, target_ids)
        self.assertEqual(result.shape, (vocab_size, 2))

        for tid in range(vocab_size):
            cpp_z, cpp_tq = self.cpp.trie_partial_z_and_target(state, trie, tid)
            cuda_z = float(result[tid, 0])
            cuda_tq = float(result[tid, 1])
            with self.subTest(tid=tid):
                self.assertLessEqual(abs(cuda_z - cpp_z), 1e-12)
                self.assertLessEqual(abs(cuda_tq - cpp_tq), 1e-12)

    def test_z_is_same_for_all_targets_in_batch(self):
        """Z must be identical for all token IDs in the same batch."""
        rng = random.Random(3333)
        state = self._make_state(seed=3333)
        vocab_size = 50
        token_bytes_list = [
            bytes(rng.randrange(256) for _ in range(rng.randint(1, 4)))
            for _ in range(vocab_size)
        ]
        trie = self.cpp.Trie()
        for tid, tb in enumerate(token_bytes_list):
            trie.insert(tid, tb)

        target_ids = np.arange(vocab_size, dtype=np.int32)
        result = self.cuda.cuda.trie_partial_z_and_target_batched(state, trie, target_ids)
        z_vals = result[:, 0]
        self.assertLessEqual(float(np.max(z_vals) - np.min(z_vals)), 1e-15,
                             "Z should be identical for all targets in a batch")


class ScorePathAArraysCudaTest(unittest.TestCase):
    """Tests for _ppmd_cuda.cuda.score_path_a_arrays_cuda."""

    def setUp(self):
        cuda = _try_import_cuda()
        if cuda is None:
            self.skipTest("_ppmd_cuda not built")
        if cuda.cuda.device_count() == 0:
            self.skipTest("no CUDA device available")
        cpp = _try_import_cpp()
        if cpp is None:
            self.skipTest("_ppmd_cpp not built (needed for reference comparison)")
        if not hasattr(cpp, "score_path_a_arrays"):
            self.skipTest("_ppmd_cpp does not expose score_path_a_arrays")
        self.cuda = cuda
        self.cpp = cpp

    def _run_score(self, backend_mod, fn_name: str,
                   target_ids, prev_ids, nll_nats,
                   bnd_flat, bnd_off, nbnd_flat, nbnd_off,
                   emit, isb, hp):
        fn = getattr(backend_mod, fn_name)
        return fn(
            np.ascontiguousarray(target_ids, dtype=np.int32),
            np.ascontiguousarray(prev_ids, dtype=np.int32),
            np.ascontiguousarray(nll_nats, dtype=np.float64),
            np.frombuffer(bnd_flat, dtype=np.uint8),
            bnd_off,
            np.frombuffer(nbnd_flat, dtype=np.uint8),
            nbnd_off,
            emit,
            isb,
            hp,
        )

    def test_score_path_a_cuda_matches_cpp_50_positions(self):
        """CUDA Path A scorer must produce BPB within 1e-12 of CPP reference for 50 positions."""
        rng = random.Random(9999)
        vocab_size = 50
        n_positions = 50

        token_bytes_list = [
            bytes(rng.randrange(256) for _ in range(rng.randint(1, 4)))
            for _ in range(vocab_size)
        ]
        bnd_flat = b"".join(token_bytes_list)
        bnd_off = np.zeros(vocab_size + 1, dtype=np.int32)
        for i, tb in enumerate(token_bytes_list):
            bnd_off[i + 1] = bnd_off[i] + len(tb)
        emit = np.ones(vocab_size, dtype=np.uint8)
        isb = np.zeros(vocab_size, dtype=np.uint8)

        target_ids = np.array([rng.randrange(vocab_size) for _ in range(n_positions)], dtype=np.int32)
        prev_ids = np.full(n_positions, -1, dtype=np.int32)
        nll_nats = np.full(n_positions, float(np.log(vocab_size)), dtype=np.float64)
        hp = {
            "order": 5, "lambda_hi": 0.9, "lambda_lo": 0.05,
            "conf_threshold": 0.9, "update_after_score": True,
        }

        cuda_out = self._run_score(
            self.cuda.cuda, "score_path_a_arrays_cuda",
            target_ids, prev_ids, nll_nats,
            bnd_flat, bnd_off, bnd_flat, bnd_off, emit, isb, hp,
        )
        cpp_out = self._run_score(
            self.cpp, "score_path_a_arrays",
            target_ids, prev_ids, nll_nats,
            bnd_flat, bnd_off, bnd_flat, bnd_off, emit, isb, hp,
        )

        cuda_bpb = float(cuda_out["bpb"])
        cpp_bpb = float(cpp_out["bpb"])
        diff = abs(cuda_bpb - cpp_bpb)
        self.assertEqual(int(cuda_out["positions"]), n_positions)
        self.assertEqual(int(cuda_out["total_bytes"]), int(cpp_out["total_bytes"]))
        self.assertLessEqual(
            diff, 1e-12,
            f"BPB diff {diff!r}: cuda={cuda_bpb!r} cpp={cpp_bpb!r}",
        )

    def test_score_path_a_cuda_matches_cpp_200_positions_with_boundary(self):
        """CUDA scorer must match CPP over 200 positions with mixed boundary tokens."""
        rng = random.Random(7777)
        vocab_size = 80
        n_positions = 200

        token_bytes_list = [
            bytes(rng.randrange(256) for _ in range(rng.randint(1, 5)))
            for _ in range(vocab_size)
        ]
        bnd_flat = b"".join(token_bytes_list)
        bnd_off = np.zeros(vocab_size + 1, dtype=np.int32)
        for i, tb in enumerate(token_bytes_list):
            bnd_off[i + 1] = bnd_off[i] + len(tb)
        # Non-boundary bytes slightly different (shifted)
        nbnd_bytes_list = [bytes((b + 1) % 256 for b in tb) for tb in token_bytes_list]
        nbnd_flat = b"".join(nbnd_bytes_list)
        nbnd_off = bnd_off.copy()

        emit = np.ones(vocab_size, dtype=np.uint8)
        isb = np.array([1 if i % 7 == 0 else 0 for i in range(vocab_size)], dtype=np.uint8)

        target_ids = np.array([rng.randrange(vocab_size) for _ in range(n_positions)], dtype=np.int32)
        prev_ids = np.array(
            [-1 if i == 0 else int(target_ids[i - 1]) for i in range(n_positions)],
            dtype=np.int32,
        )
        nll_nats = np.array([rng.uniform(0.5, 3.0) for _ in range(n_positions)], dtype=np.float64)
        hp = {
            "order": 5, "lambda_hi": 0.9, "lambda_lo": 0.05,
            "conf_threshold": 0.9, "update_after_score": True,
        }

        cuda_out = self._run_score(
            self.cuda.cuda, "score_path_a_arrays_cuda",
            target_ids, prev_ids, nll_nats,
            bnd_flat, bnd_off, nbnd_flat, nbnd_off, emit, isb, hp,
        )
        cpp_out = self._run_score(
            self.cpp, "score_path_a_arrays",
            target_ids, prev_ids, nll_nats,
            bnd_flat, bnd_off, nbnd_flat, nbnd_off, emit, isb, hp,
        )

        cuda_bpb = float(cuda_out["bpb"])
        cpp_bpb = float(cpp_out["bpb"])
        diff = abs(cuda_bpb - cpp_bpb)
        self.assertEqual(int(cuda_out["positions"]), n_positions)
        self.assertEqual(int(cuda_out["total_bytes"]), int(cpp_out["total_bytes"]))
        self.assertLessEqual(
            diff, 1e-12,
            f"BPB diff {diff!r}: cuda={cuda_bpb!r} cpp={cpp_bpb!r}",
        )

    def test_score_path_a_cuda_output_has_required_keys(self):
        """score_path_a_arrays_cuda must return required keys."""
        rng = random.Random(5555)
        vocab_size = 20
        n_positions = 10
        token_bytes_list = [bytes([rng.randrange(256)]) for _ in range(vocab_size)]
        bnd_flat = b"".join(token_bytes_list)
        bnd_off = np.zeros(vocab_size + 1, dtype=np.int32)
        for i in range(vocab_size):
            bnd_off[i + 1] = bnd_off[i] + 1
        emit = np.ones(vocab_size, dtype=np.uint8)
        isb = np.zeros(vocab_size, dtype=np.uint8)
        target_ids = np.array([rng.randrange(vocab_size) for _ in range(n_positions)], dtype=np.int32)
        prev_ids = np.full(n_positions, -1, dtype=np.int32)
        nll_nats = np.full(n_positions, 1.0, dtype=np.float64)
        hp = {"order": 5, "lambda_hi": 0.9, "lambda_lo": 0.05,
              "conf_threshold": 0.9, "update_after_score": True}

        out = self._run_score(
            self.cuda.cuda, "score_path_a_arrays_cuda",
            target_ids, prev_ids, nll_nats,
            bnd_flat, bnd_off, bnd_flat, bnd_off, emit, isb, hp,
        )
        for key in ("positions", "total_bits", "total_bytes", "bpb",
                    "start_state_digest", "end_state_digest"):
            self.assertIn(key, out, f"Missing key: {key}")
        self.assertEqual(int(out["positions"]), n_positions)
        self.assertGreater(float(out["bpb"]), 0.0)


if __name__ == "__main__":
    unittest.main()
