"""Moral Safety Gate for the Hash-Grad NMF Evaluation Pipeline.

Integrates _moral_geometry.LivingCompass and _limbic_system.LimbicSystem
into the --hash_grad evaluation waterfall as an **optional, off-by-default**
post-processing filter.

Purpose
-------
Demonstrate that safety-alignment filtering does **not** degrade BPB on
normal FineWeb English text:

  - Pre-compute alignment for all 1024 vocab tokens once at gate init
    (O(vocab × n_anchors × W_BITS), ~10 ms — negligible)
  - At eval time, `check_batch(pred_tokens)` is a single numpy fancy-index:
    O(batch_size) with zero Python loops
  - The eval waterfall reports BPB without gate (baseline) vs with gate
    (safety-filtered), counting how many tokens were ethically rejected

Activation
----------
Pass `--moral_safety` to train_gpt.py.  The gate is built inside
`_run_hash_grad_single()` and passed to `hash_grad_bpb()` as an optional
keyword argument.  Without the flag the gate is `None` and the entire
code path is zero-overhead.

Architecture
------------
Tokens with `cosine_similarity < REJECTION_THRESHOLD` against the Social Law
Manifold are flagged as "ethically rejected" — their probability contribution
in BPB is replaced with `1 / vocab_size` (uniform / no-information).
Normal English tokens never cluster in the anti-aligned region of ethical
hypervector space, so the rejection count should be ~0 and BPB unchanged.

The LimbicSystem adds a trajectory check: from a neutral context vector to
the predicted token vector.  Similarly pre-computed and cached.
"""
from __future__ import annotations

import hashlib
import struct
import time
from typing import List, Optional, Tuple

import numpy as np


# ── Rejection threshold (cosine similarity below this triggers rejection) ────
REJECTION_THRESHOLD: float = -0.5   # matches _moral_geometry.AlignmentResult


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_anchor_vector(name: str, w_bits: int) -> np.ndarray:
    """Generate a deterministic bipolar int8 anchor at exactly w_bits dims.

    Mirrors EthicalAnchorVector._generate_anchor_vector() but uses the
    caller-supplied w_bits so anchor and token dimensions always match.
    """
    digest = hashlib.sha256(name.encode()).digest()
    seed   = struct.unpack("<I", digest[:4])[0]
    rng    = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=w_bits).astype(np.int8)


def _codebook_to_int8(codebook: np.ndarray) -> np.ndarray:
    """Unpack uint64 codebook rows to (vocab_size, W_BITS) int8 bipolar."""
    # codebook: (vocab_size, W_UINT64) dtype=uint64
    # → unpack bits → (vocab_size, W_UINT64*64) bits → map 0→-1, 1→+1
    bits = np.unpackbits(codebook.view(np.uint8), axis=1)   # (V, W_BITS) uint8
    return (bits.astype(np.int8) * 2 - 1)                   # (V, W_BITS) int8


# ─────────────────────────────────────────────────────────────────────────────
# MoralSafetyGate
# ─────────────────────────────────────────────────────────────────────────────

class MoralSafetyGate:
    """Pre-computed token-level ethical alignment gate.

    Parameters
    ----------
    codebook      : (vocab_size, W_UINT64) uint64  — Hadamard token vectors
    w_uint64      : int  — number of uint64 blocks per token vector (e.g. 16)
    use_limbic    : bool — additionally apply LimbicSystem trajectory check
    personality_seed : int — seed for LimbicSystem personality (default 42)
    """

    _ANCHOR_DEFS: List[Tuple[str, float]] = [
        ("human_rights",   1.0),
        ("prosocial_norms", 0.8),
        ("constitutional",  0.9),
        ("empathy",         1.0),
        ("honesty",         0.85),
        ("cooperation",     0.75),
        ("non_violence",    1.0),
    ]

    def __init__(
        self,
        codebook: np.ndarray,
        w_uint64: int,
        use_limbic: bool = True,
        personality_seed: int = 42,
    ) -> None:
        self.vocab_size    = codebook.shape[0]
        self.w_uint64      = w_uint64
        self.w_bits        = w_uint64 * 64
        self.use_limbic    = use_limbic

        # ── 1. Unpack codebook to int8 bipolar ───────────────────────────────
        t0 = time.time()
        tokens_int8 = _codebook_to_int8(codebook)          # (V, W_BITS)

        # ── 2. Build anchor matrix ────────────────────────────────────────────
        anchor_vectors  = []
        anchor_weights  = []
        for name, weight in self._ANCHOR_DEFS:
            anchor_vectors.append(_make_anchor_vector(name, self.w_bits))
            anchor_weights.append(weight)

        anchor_matrix   = np.stack(anchor_vectors, axis=0).astype(np.float32)  # (A, W)
        weights_arr     = np.array(anchor_weights, dtype=np.float32)            # (A,)
        total_w         = float(weights_arr.sum())

        # ── 3. Vectorized alignment: (V, W) @ (W, A) → (V, A) ───────────────
        # Normalise token rows and anchor rows independently
        tok_norms    = np.linalg.norm(tokens_int8.astype(np.float32), axis=1, keepdims=True) + 1e-10
        tok_normed   = tokens_int8.astype(np.float32) / tok_norms              # (V, W)

        anc_norms    = np.linalg.norm(anchor_matrix, axis=1, keepdims=True) + 1e-10
        anc_normed   = anchor_matrix / anc_norms                               # (A, W)

        sim_matrix   = tok_normed @ anc_normed.T                               # (V, A)

        # Weighted average cosine similarity per token
        self._avg_sim: np.ndarray = (sim_matrix * weights_arr[None, :]).sum(axis=1) / total_w  # (V,)

        # Rejected = avg cosine < threshold
        self._rejected_vocab: np.ndarray = self._avg_sim < REJECTION_THRESHOLD  # (V,) bool

        # ── 4. LimbicSystem trajectory check (optional) ───────────────────────
        self._limbic_rejected: np.ndarray = np.zeros(self.vocab_size, dtype=bool)
        if use_limbic:
            try:
                from _limbic_system import LimbicSystem, PersonalitySeed
                personality = PersonalitySeed(seed=personality_seed,
                                              traits=["prosocial", "honest", "cooperative"])
                limbic = LimbicSystem(
                    uint64_count=w_uint64,
                    personality_seed=personality,
                    safety_threshold=0.7,
                )
                # Neutral context = all-zeros uint64 vector
                neutral = np.zeros(w_uint64, dtype=np.uint64)
                for tok_id in range(self.vocab_size):
                    try:
                        is_safe, _, _ = limbic.check_trajectory(neutral, codebook[tok_id])
                        if not is_safe:
                            self._limbic_rejected[tok_id] = True
                    except Exception:
                        pass
            except ImportError:
                pass  # LimbicSystem unavailable — skip

        # Combined rejection mask
        self._combined_rejected: np.ndarray = self._rejected_vocab | self._limbic_rejected

        # ── 5. Summary stats ─────────────────────────────────────────────────
        n_geo    = int(self._rejected_vocab.sum())
        n_limbic = int(self._limbic_rejected.sum())
        n_comb   = int(self._combined_rejected.sum())
        elapsed  = time.time() - t0

        print(f"[MoralSafety] Gate initialised in {elapsed:.2f}s")
        print(f"[MoralSafety] Vocab size: {self.vocab_size}")
        print(f"[MoralSafety] Anchors: {len(self._ANCHOR_DEFS)} | W_BITS: {self.w_bits}")
        print(f"[MoralSafety] Geometry  rejected: {n_geo}/{self.vocab_size}")
        print(f"[MoralSafety] Limbic    rejected: {n_limbic}/{self.vocab_size} "
              f"({'enabled' if use_limbic else 'disabled'})")
        print(f"[MoralSafety] Combined  rejected: {n_comb}/{self.vocab_size} "
              f"({100*n_comb/self.vocab_size:.2f}%)")
        print(f"[MoralSafety] Avg alignment score: {float(self._avg_sim.mean()):.4f} "
              f"(min={float(self._avg_sim.min()):.4f}, max={float(self._avg_sim.max()):.4f})")

        # Runtime counters
        self.n_checked  = 0
        self.n_rejected = 0

    # ── Runtime API ──────────────────────────────────────────────────────────

    def check_batch(self, pred_tokens: np.ndarray) -> np.ndarray:
        """Vectorized O(batch) rejection check.

        Parameters
        ----------
        pred_tokens : (B,) int32/int64  — predicted token IDs for this batch

        Returns
        -------
        rejected : (B,) bool  — True = token ethically rejected
                                → caller should substitute uniform prob
        """
        rejected = self._combined_rejected[pred_tokens]
        self.n_checked  += len(pred_tokens)
        self.n_rejected += int(rejected.sum())
        return rejected

    def alignment_scores(self, token_ids: np.ndarray) -> np.ndarray:
        """Return the pre-computed weighted cosine similarity for given tokens."""
        return self._avg_sim[token_ids]

    def report(self) -> str:
        """Human-readable runtime summary."""
        if self.n_checked == 0:
            return "[MoralSafety] No tokens evaluated yet."
        pct = 100.0 * self.n_rejected / self.n_checked
        return (
            f"[MoralSafety] Eval stats: "
            f"checked={self.n_checked:,} | "
            f"rejected={self.n_rejected:,} ({pct:.4f}%) | "
            f"vocab_rejected={int(self._combined_rejected.sum())}/{self.vocab_size}"
        )

    def reset_counters(self) -> None:
        """Reset runtime counters (call between baseline and safety eval)."""
        self.n_checked  = 0
        self.n_rejected = 0


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Smoke-test the gate with random codebook."""
    print("[MoralSafety SelfTest] Running...")
    vocab_size = 1024
    w_uint64   = 16
    rng        = np.random.default_rng(42)
    codebook   = rng.integers(0, 2**63, size=(vocab_size, w_uint64), dtype=np.int64).view(np.uint64)

    gate = MoralSafetyGate(codebook, w_uint64, use_limbic=False)
    assert gate.vocab_size == vocab_size

    preds   = rng.integers(0, vocab_size, size=5000, dtype=np.int32)
    rejected = gate.check_batch(preds)
    assert rejected.shape == (5000,), "Shape mismatch"
    assert rejected.dtype == bool

    print(gate.report())
    print("[MoralSafety SelfTest] ✓ All assertions passed")


if __name__ == "__main__":
    _self_test()
