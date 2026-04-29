"""Phase 3 tests: C++ candidate-trie scorer + vocab sharding equivalence.

Skip-gates (mirroring Phase 1+2):
  * /bin/python3.8 (no Python.h / no built extension)
  * `_ppmd_cpp` not built
  * `_ppmd_cpp` does not yet expose `Trie` / `score_path_a_arrays`
    (i.e. Phase 3 not built yet).

Conformance contract (from plan):
  * trie_partial single-shard match Python to >= 14 decimals
  * shard reduction exact (sums to single-shard to >= 14 decimals)
  * end-to-end BPB to >= 10 decimals
  * OMP_NUM_THREADS=1 vs =4 must give identical BPB
"""

from __future__ import annotations

import math
import os
import random
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


def _running_under_system_python38() -> bool:
    exe = os.path.realpath(sys.executable)
    return exe == "/bin/python3.8" or (
        exe.endswith("/python3.8") and "venv" not in exe
    )


def _make_random_vocab(rng: random.Random, vocab_size: int):
    """Build a vocab where each token has a short pseudo-random byte string.

    Returns:
      tokens: list of bytes (length = vocab_size)
      emittable: list[bool]
      is_boundary: list[bool]  (whether token starts new word; here random)
    """
    tokens = []
    emittable = []
    is_boundary = []
    for tid in range(vocab_size):
        n = rng.randint(1, 5)
        b = bytes(rng.randint(0, 255) for _ in range(n))
        tokens.append(b)
        emittable.append(True)
        is_boundary.append(rng.random() < 0.3)
    return tokens, emittable, is_boundary


def _build_py_candidates(tokens, emittable, is_boundary):
    """Build PY-side CandidateBytes list. We treat after_boundary == after_non_boundary
    (no leading-space marker handling) so that boundary and non-boundary tries are identical;
    the test then exercises the choice-of-trie code path harmlessly.
    """
    from scripts.eval_path_a_ppmd import CandidateBytes

    cands = []
    for tid, b in enumerate(tokens):
        cands.append(
            CandidateBytes(
                token_id=tid,
                after_boundary=b,
                after_non_boundary=b,
                emittable=bool(emittable[tid]),
            )
        )
    return cands


def _build_cpp_trie(_ppmd_cpp, tokens, emittable):
    trie = _ppmd_cpp.Trie()
    for tid, b in enumerate(tokens):
        if emittable[tid]:
            trie.insert(int(tid), b)
    return trie


def _vocab_to_csr(tokens):
    """Pack a vocab into (flat bytes uint8, offsets int32 [V+1])."""
    flat = bytearray()
    offsets = [0]
    for b in tokens:
        flat.extend(b)
        offsets.append(len(flat))
    return (
        np.frombuffer(bytes(flat), dtype=np.uint8).copy(),
        np.asarray(offsets, dtype=np.int32),
    )


class PpmdCppScorerTestBase(unittest.TestCase):
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

        if not hasattr(_ppmd_cpp, "Trie"):
            self.skipTest(
                "_ppmd_cpp does not expose Trie (Phase 3 not built)"
            )
        if not hasattr(_ppmd_cpp, "trie_partial_z_and_target"):
            self.skipTest(
                "_ppmd_cpp does not expose trie_partial_z_and_target"
            )
        if not hasattr(_ppmd_cpp, "score_path_a_arrays"):
            self.skipTest(
                "_ppmd_cpp does not expose score_path_a_arrays"
            )
        try:
            from scripts.eval_path_a_ppmd import (  # noqa: F401
                PPMDState as PyPPMDState,
                build_candidate_tries,
                trie_partial_z_and_target,
                combine_path_a_partials,
                score_path_a_arrays,
            )
        except Exception as e:
            self.skipTest(f"cannot import Python reference: {e}")


class TestPpmdCppScorer(PpmdCppScorerTestBase):
    def test_ppmd_cpp_trie_partial_matches_python_single_shard(self) -> None:
        import _ppmd_cpp
        from scripts.eval_path_a_ppmd import (
            PPMDState as PyPPMDState,
            build_candidate_tries,
            trie_partial_z_and_target as py_partial,
        )

        rng = random.Random(0xBEEF)
        tokens, emittable, is_boundary = _make_random_vocab(rng, vocab_size=32)
        candidates = _build_py_candidates(tokens, emittable, is_boundary)
        boundary_root, _ = build_candidate_tries(candidates)

        # Identical 200-byte stream into both states.
        seed = bytes(rng.randint(0, 255) for _ in range(200))
        py_state = PyPPMDState(order=5)
        cpp_state = _ppmd_cpp.PPMDState(5)
        py_state.update_bytes(seed)
        cpp_state.update_bytes(seed)
        # Sanity: same backend digest after identical feed.
        self.assertEqual(py_state.state_digest(), cpp_state.state_digest())

        cpp_trie = _build_cpp_trie(_ppmd_cpp, tokens, emittable)

        target_id = 7  # arbitrary
        py_z, py_q, py_n = py_partial(
            boundary_root, py_state, target_id, shard_start=0, shard_end=32
        )

        cpp_virtual = cpp_state.clone_virtual()
        out = _ppmd_cpp.trie_partial_z_and_target(
            cpp_virtual, cpp_trie, int(target_id), 0, 32
        )
        cpp_z = float(out["z"])
        cpp_q = float(out["target_q"])
        cpp_n = int(out["terminal_count"])

        self.assertEqual(py_n, cpp_n)
        self.assertLessEqual(abs(py_z - cpp_z), 1e-14,
                             f"z diff {abs(py_z - cpp_z)!r}")
        self.assertLessEqual(abs(py_q - cpp_q), 1e-14,
                             f"q diff {abs(py_q - cpp_q)!r}")

    def test_ppmd_cpp_trie_shard_reduction_exact(self) -> None:
        import _ppmd_cpp
        from scripts.eval_path_a_ppmd import combine_path_a_partials

        rng = random.Random(0xCAFE)
        tokens, emittable, is_boundary = _make_random_vocab(rng, vocab_size=32)
        seed = bytes(rng.randint(0, 255) for _ in range(200))
        cpp_state = _ppmd_cpp.PPMDState(5)
        cpp_state.update_bytes(seed)
        cpp_trie = _build_cpp_trie(_ppmd_cpp, tokens, emittable)
        target_id = 11

        full = _ppmd_cpp.trie_partial_z_and_target(
            cpp_state.clone_virtual(), cpp_trie, target_id, 0, 32
        )
        a = _ppmd_cpp.trie_partial_z_and_target(
            cpp_state.clone_virtual(), cpp_trie, target_id, 0, 16
        )
        b = _ppmd_cpp.trie_partial_z_and_target(
            cpp_state.clone_virtual(), cpp_trie, target_id, 16, 32
        )
        merged = _ppmd_cpp.combine_path_a_partials([a, b])

        self.assertEqual(int(full["terminal_count"]),
                         int(merged["terminal_count"]))
        self.assertLessEqual(
            abs(float(full["z"]) - float(merged["z"])), 1e-14,
            f"shard z reduction diff: {full['z']!r} vs {merged['z']!r}"
        )
        self.assertLessEqual(
            abs(float(full["target_q"]) - float(merged["target_q"])), 1e-14,
            f"shard q reduction diff: {full['target_q']!r} vs {merged['target_q']!r}"
        )
        # Cross-check Python combine on the same partials reproduces.
        py_merged = combine_path_a_partials([
            (float(a["z"]), float(a["target_q"]), int(a["terminal_count"])),
            (float(b["z"]), float(b["target_q"]), int(b["terminal_count"])),
        ])
        self.assertEqual(int(merged["terminal_count"]), int(py_merged[2]))
        self.assertEqual(float(merged["z"]), float(py_merged[0]))
        self.assertEqual(float(merged["target_q"]), float(py_merged[1]))

    def _build_synthetic_stream(self, rng, vocab_size, n_positions):
        """Build a synthetic but reproducible stream for end-to-end scoring."""
        tokens, emittable, is_boundary = _make_random_vocab(rng, vocab_size)
        # Ensure target_ids are valid and prev_ids reference real tokens.
        target_ids = np.asarray(
            [rng.randrange(vocab_size) for _ in range(n_positions)], dtype=np.int32
        )
        prev_ids = np.asarray(
            [-1 if i == 0 else int(target_ids[i - 1]) for i in range(n_positions)],
            dtype=np.int32,
        )
        # Synthetic NLLs in nats; bounded so exp(-nll) is reasonable.
        nll_nats = np.asarray(
            [rng.uniform(0.5, 4.0) for _ in range(n_positions)], dtype=np.float64
        )
        return tokens, emittable, is_boundary, target_ids, prev_ids, nll_nats

    def test_ppmd_cpp_score_path_a_arrays_matches_python(self) -> None:
        import _ppmd_cpp
        import time as _time
        from scripts.eval_path_a_ppmd import score_path_a_arrays as py_score

        rng = random.Random(20240501)
        vocab_size = 64
        n_positions = 256
        (tokens, emittable, is_boundary,
         target_ids, prev_ids, nll_nats) = self._build_synthetic_stream(
            rng, vocab_size, n_positions
        )
        candidates = _build_py_candidates(tokens, emittable, is_boundary)

        # Python reference
        t0 = _time.perf_counter()
        py_out = py_score(
            target_ids.tolist(),
            prev_ids.tolist(),
            nll_nats.tolist(),
            candidates,
            list(is_boundary),
            order=5,
            lambda_hi=0.9,
            lambda_lo=0.05,
            conf_threshold=0.9,
        )
        t_py = _time.perf_counter() - t0

        # C++ end-to-end
        boundary_flat, boundary_off = _vocab_to_csr(tokens)
        # after_non_boundary == after_boundary in this synthetic vocab.
        nonb_flat, nonb_off = boundary_flat.copy(), boundary_off.copy()
        emit_arr = np.asarray([1 if e else 0 for e in emittable], dtype=np.uint8)
        isb_arr = np.asarray([1 if b else 0 for b in is_boundary], dtype=np.uint8)

        t0 = _time.perf_counter()
        cpp_out = _ppmd_cpp.score_path_a_arrays(
            target_ids,
            prev_ids,
            nll_nats,
            boundary_flat, boundary_off,
            nonb_flat, nonb_off,
            emit_arr, isb_arr,
            {
                "order": 5,
                "lambda_hi": 0.9,
                "lambda_lo": 0.05,
                "conf_threshold": 0.9,
                "update_after_score": True,
            },
        )
        t_cpp = _time.perf_counter() - t0

        py_bpb = float(py_out["bpb"])
        cpp_bpb = float(cpp_out["bpb"])
        diff = abs(py_bpb - cpp_bpb)
        # Stash for reporting at module exit.
        TestPpmdCppScorer._end_to_end_bpb_diff = diff
        TestPpmdCppScorer._end_to_end_t_py = t_py
        TestPpmdCppScorer._end_to_end_t_cpp = t_cpp
        self.assertEqual(int(py_out["total_bytes"]),
                         int(cpp_out["total_bytes"]))
        self.assertLess(diff, 1e-10,
                        f"bpb diff {diff!r} (py={py_bpb!r} cpp={cpp_bpb!r})")

    def test_ppmd_cpp_score_first_invariant_holds_across_arrays(self) -> None:
        import _ppmd_cpp

        rng = random.Random(20240502)
        vocab_size = 32
        n_positions = 64
        (tokens, emittable, is_boundary,
         target_ids, prev_ids, nll_nats) = self._build_synthetic_stream(
            rng, vocab_size, n_positions
        )
        boundary_flat, boundary_off = _vocab_to_csr(tokens)
        emit_arr = np.asarray([1 if e else 0 for e in emittable], dtype=np.uint8)
        isb_arr = np.asarray([1 if b else 0 for b in is_boundary], dtype=np.uint8)

        # Run with update_after_score=False: state must NOT change.
        out_noup = _ppmd_cpp.score_path_a_arrays(
            target_ids, prev_ids, nll_nats,
            boundary_flat, boundary_off,
            boundary_flat, boundary_off,
            emit_arr, isb_arr,
            {"order": 5, "update_after_score": False},
        )
        self.assertIn("start_state_digest", out_noup)
        self.assertIn("end_state_digest", out_noup)
        self.assertEqual(out_noup["start_state_digest"],
                         out_noup["end_state_digest"],
                         "scoring mutated state when update_after_score=False")

        # Run with update_after_score=True: state must change after at least one
        # non-empty update.
        out_up = _ppmd_cpp.score_path_a_arrays(
            target_ids, prev_ids, nll_nats,
            boundary_flat, boundary_off,
            boundary_flat, boundary_off,
            emit_arr, isb_arr,
            {"order": 5, "update_after_score": True},
        )
        self.assertNotEqual(out_up["start_state_digest"],
                            out_up["end_state_digest"],
                            "update_after_score=True did not mutate state")
        # The two runs intentionally produce different total_bits: with
        # update_after_score=False the PPM state never advances so every
        # position sees the same (empty) state, whereas update_after_score=True
        # is the real eval path. The score-first invariant is captured by the
        # digest-equality check above; cross-run bits parity is not required
        # and not meaningful.

    def test_ppmd_cpp_openmp_thread_count_does_not_change_result(self) -> None:
        import _ppmd_cpp

        if not hasattr(_ppmd_cpp, "set_num_threads"):
            self.skipTest("_ppmd_cpp.set_num_threads not exposed")

        rng = random.Random(20240503)
        vocab_size = 64
        n_positions = 128
        (tokens, emittable, is_boundary,
         target_ids, prev_ids, nll_nats) = self._build_synthetic_stream(
            rng, vocab_size, n_positions
        )
        boundary_flat, boundary_off = _vocab_to_csr(tokens)
        emit_arr = np.asarray([1 if e else 0 for e in emittable], dtype=np.uint8)
        isb_arr = np.asarray([1 if b else 0 for b in is_boundary], dtype=np.uint8)

        def run_with(threads):
            _ppmd_cpp.set_num_threads(int(threads))
            return _ppmd_cpp.score_path_a_arrays(
                target_ids, prev_ids, nll_nats,
                boundary_flat, boundary_off,
                boundary_flat, boundary_off,
                emit_arr, isb_arr,
                {"order": 5, "update_after_score": True},
            )

        out1 = run_with(1)
        out4 = run_with(4)
        # Bit-for-bit identical (deterministic reduction order).
        self.assertEqual(float(out1["total_bits"]), float(out4["total_bits"]))
        self.assertEqual(float(out1["bpb"]), float(out4["bpb"]))
        self.assertEqual(int(out1["total_bytes"]), int(out4["total_bytes"]))


if __name__ == "__main__":
    unittest.main()
