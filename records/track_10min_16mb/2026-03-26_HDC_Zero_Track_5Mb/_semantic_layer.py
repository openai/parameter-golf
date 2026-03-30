"""DirectionalSemanticVec — long-range relational metacognition layer.

Fixes two structural gaps in the original DualVectorProjection so that the
metacognition can see ALL corpus relationships simultaneously, at any range,
with correct directionality.

GAP 1 — COLLISION DENSITY
--------------------------
Original: rel_window = (idx_A XOR idx_B) & mask
With vocab_size=1024 and idx = token_id % uint64_count, every token index
is < 1024. The XOR of any two such indices is also < 1024, so all ~500K
pairs collapse into only 1024 distinct windows out of 16384 available.
~500 pairs share each window on average — signal quality is severely degraded.

Fix: token-addressed windows.
  Token T owns window [T*W : (T+1)*W] exclusively.
  vocab_size=1024, W=16 → 1024*16 = 16384 = uint64_count → zero collision.
  Every token has its own 1024-bit region. Pairs never mix.

GAP 2 — DIRECTIONALITY
------------------------
Original: XOR is commutative, so A→B and B→A map to the same window.
The model cannot distinguish "fox PRECEDES jumps" from "jumps PRECEDES fox".

Fix: two separate vectors, sem_fwd and sem_bwd.
  sem_fwd[T*W:(T+1)*W]: XOR-bundle of Hadamard rows of all tokens that
                         FOLLOWED token T in the corpus.
  sem_bwd[T*W:(T+1)*W]: XOR-bundle of Hadamard rows of all tokens that
                         PRECEDED token T in the corpus.

  Query "does A predict C?" → check sem_fwd[A's window] against C's vector.
  Query "does C expect A before it?" → check sem_bwd[C's window] against A's.
  These are different arrays → direction is unambiguous.

INSTANT ACCESS
--------------
Query any relationship at inference time in O(W) = O(16) uint64 operations,
regardless of how far apart A and C appeared in the corpus.
The metacognition thus has simultaneous visibility of all positions.

CREATIVITY
----------
sem_bwd provides TENSION signal: if the backward context of C strongly
expects a different token than the forward context of A predicts, the model
is in a state of creative tension. This is the coherence vs. surprise
tradeoff that drives creativity in the CreativeCoherenceManager.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

# Minimum signed semantic score for a prediction override.
# Range is (-1, 1) across CTX_LEN*2 votes. Values above this threshold
# indicate a genuine corpus relationship, not noise.
SEM_CONFIDENCE_MIN = 0.15


class DirectionalSemanticVec:
    """Two-vector directional semantic layer with zero-collision token addressing.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary (e.g. 1024).
    W : int
        Number of uint64 blocks per token window (e.g. 16 = 1024 bits).
        Must satisfy vocab_size * W == uint64_count.
    uint64_count : int
        Total number of uint64 elements in the full HDC vector
        (= hdc_dim // 64, e.g. 16384 for hdc_dim=2^20).
    """

    def __init__(self, vocab_size: int, W: int, uint64_count: int) -> None:
        if vocab_size * W != uint64_count:
            raise ValueError(
                f"Token-addressed tiling requires vocab_size*W == uint64_count, "
                f"got {vocab_size}*{W}={vocab_size*W} vs {uint64_count}."
            )
        self.vocab_size = vocab_size
        self.W = W
        self.uint64_count = uint64_count
        self._neutral = 32 * W          # expected popcount for a random vector
        self._neutral_f = float(self._neutral)

        # Two full HDC vectors. Token T owns blocks [T*W : (T+1)*W] in each.
        self.sem_fwd = np.zeros(uint64_count, dtype=np.uint64)  # A precedes B
        self.sem_bwd = np.zeros(uint64_count, dtype=np.uint64)  # B is preceded by A

    # ------------------------------------------------------------------
    # Building from corpus
    # ------------------------------------------------------------------

    @classmethod
    def build_from_tokens(
        cls,
        tokens: np.ndarray,
        codebook: np.ndarray,
        ctx_len: int,
        vocab_size: int,
        W: int,
        uint64_count: int,
        time_budget_s: float = 60.0,
        chunk_size: int = 2_000_000,
        label: str = "SemanticBuild",
    ) -> "DirectionalSemanticVec":
        """Build sem_fwd and sem_bwd from the full token array in O(N) time.

        For each pair (tokens[p-c], tokens[p]) where c in 1..ctx_len,
        records the directional relationship: tokens[p-c] PRECEDES tokens[p].

        The XOR accumulation is vectorized per unique context-token value so
        each numpy call operates on a contiguous slab rather than one element
        at a time.

        Parameters
        ----------
        tokens     : 1-D uint16 token array (already loaded and clipped).
        codebook   : (vocab_size, W) uint64 — one W-block vector per token.
        ctx_len    : Number of preceding context positions to record (e.g. 8).
        vocab_size : Vocabulary size.
        W          : Blocks per token window.
        uint64_count : Total uint64 count of the HDC vector.
        time_budget_s : Wall-clock seconds to spend building (soft limit).
        chunk_size : Tokens processed per inner chunk.
        label      : Log prefix.
        """
        dsv = cls(vocab_size, W, uint64_count)
        N = len(tokens)
        start = time.time()

        print(f"\n[{label}] Building directional semantic vectors "
              f"(vocab={vocab_size}, W={W}, ctx_len={ctx_len})")

        total_pairs = 0
        for c in range(1, ctx_len + 1):
            if time.time() - start > time_budget_s:
                print(f"[{label}] Time budget reached at context depth c={c-1}")
                break

            # a_toks[i] preceded b_toks[i] by c positions
            a_toks = tokens[: N - c].astype(np.int32)   # shape (N-c,)
            b_toks = tokens[c:].astype(np.int32)         # shape (N-c,)

            # Process in chunks to keep memory footprint low
            M = len(a_toks)
            for chunk_start in range(0, M, chunk_size):
                if time.time() - start > time_budget_s:
                    break
                chunk_end = min(chunk_start + chunk_size, M)
                a_chunk = a_toks[chunk_start:chunk_end]  # (K,)
                b_chunk = b_toks[chunk_start:chunk_end]  # (K,)

                # --- Forward: A→B ---
                # For each unique value of A, XOR-reduce all codebook[B] rows
                # into sem_fwd[A*W : (A+1)*W].
                dsv._scatter_xor(dsv.sem_fwd, a_chunk, b_chunk, codebook)

                # --- Backward: A←B (B is preceded by A) ---
                # For each unique value of B, XOR-reduce all codebook[A] rows
                # into sem_bwd[B*W : (B+1)*W].
                dsv._scatter_xor(dsv.sem_bwd, b_chunk, a_chunk, codebook)

                total_pairs += chunk_end - chunk_start

            elapsed = time.time() - start
            print(f"[{label}] c={c}/{ctx_len}  pairs={total_pairs:,}  "
                  f"elapsed={elapsed:.1f}s")

        elapsed = time.time() - start
        print(f"[{label}] Done. {total_pairs:,} A→B pairs recorded in {elapsed:.1f}s")
        return dsv

    @staticmethod
    def _scatter_xor(
        vec: np.ndarray,           # (uint64_count,) — modified in-place
        index_toks: np.ndarray,    # (K,) int32 — the "owner" token index
        value_toks: np.ndarray,    # (K,) int32 — which token to XOR in
        codebook: np.ndarray,      # (vocab_size, W) uint64
    ) -> None:
        """For each unique index token T, XOR-reduce codebook[V] into vec[T's window].

        Groups positions by their index_tok value, then performs a single
        np.bitwise_xor.reduce per group — one numpy call per unique token,
        not one per position.
        """
        W = codebook.shape[1]
        unique_toks = np.unique(index_toks)
        for tok in unique_toks:
            mask = index_toks == tok
            values = codebook[value_toks[mask]]   # (count, W) uint64
            if len(values) == 0:
                continue
            # XOR-reduce: equivalent to XOR of all contributing codebook rows
            bundle = np.bitwise_xor.reduce(values, axis=0)  # (W,)
            win_start = int(tok) * W
            vec[win_start: win_start + W] ^= bundle

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_forward(self, token_a: int, token_b: int, codebook: np.ndarray) -> float:
        """O(W): Signed confidence that token_b tends to follow token_a.

        Returns a value in roughly (-1, 1):
          > 0 : positive co-occurrence (B frequently follows A)
          ≈ 0 : no evidence
          < 0 : negative correlation (B rarely follows A)
        """
        win = slice(token_a * self.W, (token_a + 1) * self.W)
        signal = self.sem_fwd[win] ^ codebook[token_b]
        pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        return (pc - self._neutral_f) / self._neutral_f

    def query_backward(self, token_b: int, token_a: int, codebook: np.ndarray) -> float:
        """O(W): Signed confidence that token_a tends to precede token_b."""
        win = slice(token_b * self.W, (token_b + 1) * self.W)
        signal = self.sem_bwd[win] ^ codebook[token_a]
        pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        return (pc - self._neutral_f) / self._neutral_f

    def vote_scores_for_context_tok(
        self, ctx_tok: int, codebook: np.ndarray
    ) -> np.ndarray:
        """O(vocab_size * W): Score all vocab candidates given one context token.

        Returns float32 array of shape (vocab_size,).
        Combines forward signal (what follows ctx_tok?) and backward signal
        (what has ctx_tok as a predecessor?).
        """
        win = slice(ctx_tok * self.W, (ctx_tok + 1) * self.W)
        fwd_win = self.sem_fwd[win]                          # (W,) uint64
        bwd_win = self.sem_bwd[win]                          # (W,) uint64

        # Broadcast XOR against all codebook rows: (vocab_size, W)
        fwd_signals = np.ascontiguousarray(fwd_win[None, :] ^ codebook)
        bwd_signals = np.ascontiguousarray(bwd_win[None, :] ^ codebook)

        # Popcount via uint8 view + unpackbits
        # (vocab_size, W) uint64 → (vocab_size, W*8) uint8 → (vocab_size, W*64) bits
        fwd_pc = np.unpackbits(fwd_signals.view(np.uint8), axis=1).sum(axis=1)
        bwd_pc = np.unpackbits(bwd_signals.view(np.uint8), axis=1).sum(axis=1)

        # Signed score: positive = co-occurrence, negative = anti-correlation
        scores = (
            (fwd_pc.astype(np.float32) - self._neutral_f)
            + (bwd_pc.astype(np.float32) - self._neutral_f)
        ) / self._neutral_f

        return scores  # (vocab_size,)

    def augment_predictions(
        self,
        preds: np.ndarray,         # (chunk_n,) uint16 — current predictions
        table_conf: np.ndarray,    # (chunk_n,) int32 — Boyer-Moore confidence
        context_matrix: np.ndarray,# (CTX_LEN, chunk_n) int32 — context tokens
        codebook: np.ndarray,      # (vocab_size, W) uint64
        conf_threshold: int = 3,
        sem_min: float = SEM_CONFIDENCE_MIN,
    ) -> Tuple[np.ndarray, int]:
        """Augment low-confidence predictions with the directional semantic vote.

        Only overrides predictions where:
          1. The table confidence is below conf_threshold (uncertain lookup), AND
          2. The semantic vote score exceeds sem_min (genuine relationship found).

        High-confidence table entries (crystallized through Boyer-Moore) are
        never touched — the semantic layer fills gaps, not overrides certainty.

        Returns
        -------
        preds_out : updated prediction array (uint16)
        n_overrides : number of positions overridden
        """
        chunk_n = len(preds)
        low_conf_mask = table_conf < conf_threshold
        if not np.any(low_conf_mask):
            return preds, 0

        # Accumulate semantic vote scores over all context positions
        sem_vote = np.zeros((chunk_n, self.vocab_size), dtype=np.float32)
        ctx_len = context_matrix.shape[0]

        for c in range(ctx_len):
            ctx_slice = context_matrix[c]  # (chunk_n,) int32
            for ctx_tok in np.unique(ctx_slice):
                pos_mask = ctx_slice == ctx_tok
                scores = self.vote_scores_for_context_tok(int(ctx_tok), codebook)
                sem_vote[pos_mask] += scores[None, :] if pos_mask.sum() > 1 else scores

        # Best semantic candidate per position
        sem_preds = np.argmax(sem_vote, axis=1).astype(np.uint16)

        # Tension signal: did forward and backward disagree?
        # (Could be used for creativity routing in future)
        sem_best_score = sem_vote[np.arange(chunk_n), sem_preds.astype(np.int64)]

        # Override: low-confidence table AND confident semantic signal
        override_mask = low_conf_mask & (sem_best_score > sem_min)
        n_overrides = int(override_mask.sum())

        preds_out = preds.copy()
        preds_out[override_mask] = sem_preds[override_mask]

        return preds_out, n_overrides

    # ------------------------------------------------------------------
    # Sleep / consolidation
    # ------------------------------------------------------------------

    def slow_wave(self, noise_threshold: float = 0.15) -> Tuple[int, int]:
        """Decay per-token-window signals that are near-neutral toward neutral.

        Unlike the original slow_wave_consolidation which iterates scalar
        uint64 elements, this operates on W-element windows so confidence
        is measured over 1024 bits (not 64), giving a much more reliable
        signal-vs-noise distinction.

        Returns (windows_pruned, windows_nudged) — one window = one token.
        """
        pruned = 0
        nudged = 0
        neutral = self._neutral  # 32 * W

        for tok in range(self.vocab_size):
            win = slice(tok * self.W, (tok + 1) * self.W)

            for vec in (self.sem_fwd, self.sem_bwd):
                block = vec[win]  # (W,) uint64
                pc = int(np.unpackbits(block.view(np.uint8)).sum())
                conf = abs(pc - neutral) / neutral

                if conf < noise_threshold:
                    # Nudge one random bit per uint64 toward neutral
                    if pc > neutral:
                        # Too many ones — pick random set bits and clear
                        for i in range(self.W):
                            bit = np.random.randint(64)
                            if (int(block[i]) >> bit) & 1:
                                block[i] = np.uint64(int(block[i]) & ~(1 << bit))
                        vec[win] = block
                        pruned += 1
                    elif pc < neutral:
                        for i in range(self.W):
                            bit = np.random.randint(64)
                            if not ((int(block[i]) >> bit) & 1):
                                block[i] = np.uint64(int(block[i]) | (1 << bit))
                        vec[win] = block
                        nudged += 1

        return pruned, nudged

    def crystallized_relationships(
        self, codebook: np.ndarray, threshold: float = 0.6
    ) -> List[Tuple[int, int, float, float, str]]:
        """Return high-confidence token pairs from sem_fwd.

        Returns list of (token_a, token_b, fwd_conf, bwd_conf, rel_type)
        for all pairs where fwd_conf > threshold.  O(vocab_size^2 * W).
        """
        results = []
        for a in range(self.vocab_size):
            win = slice(a * self.W, (a + 1) * self.W)
            fwd_win = self.sem_fwd[win]
            for b in range(self.vocab_size):
                if a == b:
                    continue
                signal = fwd_win ^ codebook[b]
                pc = int(np.unpackbits(signal.view(np.uint8)).sum())
                fwd_conf = (pc - self._neutral_f) / self._neutral_f
                if fwd_conf > threshold:
                    bwd_conf = self.query_backward(b, a, codebook)
                    if fwd_conf > 0.85 and bwd_conf > 0.85:
                        rel_type = "SYNONYM"
                    elif fwd_conf > 0.7 and bwd_conf < 0.3:
                        rel_type = "PRECEDES"
                    elif fwd_conf > 0.5:
                        rel_type = "ASSOCIATES-WITH"
                    else:
                        rel_type = "AMBIGUOUS"
                    results.append((a, b, fwd_conf, bwd_conf, rel_type))
        return results

    def summary(self) -> dict:
        """Return mean confidence and coverage statistics for both vectors."""
        stats = {}
        for name, vec in (("sem_fwd", self.sem_fwd), ("sem_bwd", self.sem_bwd)):
            confs = []
            for tok in range(self.vocab_size):
                win = slice(tok * self.W, (tok + 1) * self.W)
                pc = int(np.unpackbits(vec[win].view(np.uint8)).sum())
                confs.append(abs(pc - self._neutral) / self._neutral)
            arr = np.array(confs)
            stats[name] = {
                "mean_confidence": float(arr.mean()),
                "high_conf_tokens": int((arr > 0.5).sum()),
                "neutral_tokens": int((arr < 0.1).sum()),
            }
        return stats
