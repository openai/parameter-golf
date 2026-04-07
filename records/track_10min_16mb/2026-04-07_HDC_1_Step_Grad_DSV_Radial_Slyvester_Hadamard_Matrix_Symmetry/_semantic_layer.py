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

    @staticmethod
    def _scatter_xor_fast(
        vec_2d: np.ndarray,        # (vocab_size, W) uint64 — modified in-place
        a_toks: np.ndarray,        # (M,) int32 — owner token indices
        b_toks: np.ndarray,        # (M,) int32 — value token indices
        codebook: np.ndarray,      # (vocab_size, W) uint64
        chunk_size: int = 8_000_000,
    ) -> None:
        """Chunked vectorised scatter-XOR using argsort + reduceat.

        Processes the M-pair array in chunks to bound peak RAM.  Each chunk
        allocates at most chunk_size × W × 8 bytes (e.g. 8M × 16 × 8 = 1 GB).
        Within each chunk the operation is a single numpy argsort + reduceat —
        no GPU round-trips, no per-chunk exception overhead.

        CUDA note: cp.bitwise_xor.reduceat is not supported on CUDA (CuPy
        raises NotImplementedError).  Rather than paying the exception +
        D2H fallback overhead on every chunk, we skip the GPU entirely and
        use this CPU reduceat path which is already fully vectorised.

        Speed: ~3–4 s per 500M-token pass on a modern CPU (8M-token chunks,
        ~62 chunks, each ~50 ms).  The old GPU path with exception fallback
        paid ~200 ms overhead per chunk → ~12 s per pass.
        """
        M = len(a_toks)
        if M == 0:
            return
        for cs in range(0, M, chunk_size):
            ce = min(cs + chunk_size, M)
            a_c = a_toks[cs:ce]
            b_c = b_toks[cs:ce]
            # Sort by owner token so equal owners are contiguous within chunk
            order   = a_c.argsort(kind="stable")
            a_sort  = a_c[order]
            b_sort  = b_c[order]
            # Gather codebook rows: (chunk, W) — bounded peak RAM
            vecs    = codebook[b_sort]
            # XOR-reduce each contiguous group: (n_unique, W)
            unique_a, first_idx = np.unique(a_sort, return_index=True)
            bundles = np.bitwise_xor.reduceat(vecs, first_idx, axis=0)
            # Scatter-accumulate into the (vocab_size, W) result
            vec_2d[unique_a] ^= bundles

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
        chunk_size: int = 8_000_000,
        label: str = "SemanticBuild",
    ) -> "DirectionalSemanticVec":
        """Build sem_fwd and sem_bwd from the full token array.

        Uses chunked vectorised numpy reduceat — no GPU round-trips.
        cp.bitwise_xor.reduceat is not supported on CUDA so the old GPU path
        paid exception + D2H overhead on every chunk; this implementation
        skips that entirely.  Each chunk is one argsort + reduceat call
        (~50 ms for 8M tokens), giving ~3–4 s per context depth over 500M tokens.
        """
        dsv = cls(vocab_size, W, uint64_count)
        N   = len(tokens)
        start = time.time()

        print(f"\n[{label}] Building directional semantic vectors "
              f"(vocab={vocab_size}, W={W}, ctx_len={ctx_len})")

        # Pre-cast tokens once — reused for all context depths
        tokens_i32 = tokens.astype(np.int32)

        sf_2d = dsv.sem_fwd.reshape(vocab_size, W)   # views into dsv arrays
        sb_2d = dsv.sem_bwd.reshape(vocab_size, W)

        total_pairs = 0
        for c in range(1, ctx_len + 1):
            if time.time() - start > time_budget_s:
                print(f"[{label}] Time budget reached at context depth c={c-1}")
                break

            M = N - c
            a_toks = tokens_i32[:M]
            b_toks = tokens_i32[c:]

            cls._scatter_xor_fast(sf_2d, a_toks, b_toks, codebook, chunk_size)
            cls._scatter_xor_fast(sb_2d, b_toks, a_toks, codebook, chunk_size)
            total_pairs += M

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
        """Thin wrapper around _scatter_xor_fast for callers that hold a 1-D vec."""
        if len(index_toks) == 0:
            return
        W          = codebook.shape[1]
        vocab_size = codebook.shape[0]
        vec_2d     = vec.reshape(vocab_size, W)
        DirectionalSemanticVec._scatter_xor_fast(vec_2d, index_toks, value_toks, codebook)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_forward(self, token_a: int, token_b: int, codebook: np.ndarray) -> float:
        """O(W): Signed confidence that token_b tends to follow token_a.

        Returns a value in roughly (-1, 1):
          > 0 : positive co-occurrence (B frequently follows A)
          ≈ 0 : no evidence
          < 0 : negative correlation (B rarely follows A)

        FIX: XOR-similarity is INVERTED — low popcount means high similarity.
        sem_fwd[A] stores XOR-bundle of codebook[B] for all B that followed A.
        If B is the dominant follower: sem_fwd[A] ≈ codebook[B], so
        sem_fwd[A] XOR codebook[B] ≈ 0 → popcount ≈ 0 → score should be +1.
        Original formula (pc - neutral)/neutral gave -1 for the correct token.
        Fix: (neutral - pc)/neutral so low XOR popcount → high positive score.
        """
        win = slice(token_a * self.W, (token_a + 1) * self.W)
        signal = self.sem_fwd[win] ^ codebook[token_b]
        pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        return (self._neutral_f - pc) / self._neutral_f  # FIXED: negated

    def query_backward(self, token_b: int, token_a: int, codebook: np.ndarray) -> float:
        """O(W): Signed confidence that token_a tends to precede token_b.

        FIX: same sign inversion as query_forward — negated so low XOR popcount
        (high similarity of sem_bwd[B] to codebook[A]) gives a positive score.
        """
        win = slice(token_b * self.W, (token_b + 1) * self.W)
        signal = self.sem_bwd[win] ^ codebook[token_a]
        pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        return (self._neutral_f - pc) / self._neutral_f  # FIXED: negated

    def vote_scores_for_context_tok(
        self, ctx_tok: int, codebook: np.ndarray
    ) -> np.ndarray:
        """O(vocab_size * W): Score all vocab candidates given one context token.

        Returns float32 array of shape (vocab_size,).
        Combines forward signal (what follows ctx_tok?) and backward signal
        (does ctx_tok tend to precede each candidate?).

        FIXES applied:
        1. Sign inversion: XOR-similarity is inverted — low popcount = high
           similarity = should give HIGH POSITIVE score.  Original formula
           (pc - neutral)/neutral gave -1 for the correct token, causing argmax
           to always pick the LEAST correlated candidate.
           Fix: (neutral - pc)/neutral.

        2. Wrong backward-query direction: the original code computed
           sem_bwd[ctx_tok's window] XOR codebook[B_cand], which measures
           "how similar is B_cand to things that preceded ctx_tok?" — irrelevant
           noise for next-token prediction.
           Fix: sem_bwd[B_cand's window] XOR codebook[ctx_tok], which measures
           "how strongly did ctx_tok precede B_cand?" — the correct signal.
        """
        win = slice(ctx_tok * self.W, (ctx_tok + 1) * self.W)
        fwd_win = self.sem_fwd[win]                          # (W,) uint64

        # Forward query: sem_fwd[ctx_tok] XOR codebook[B_cand]
        # Low popcount → B_cand is the dominant follower of ctx_tok → HIGH score
        fwd_signals = np.ascontiguousarray(fwd_win[None, :] ^ codebook)  # (vocab_size, W)
        fwd_pc = np.unpackbits(fwd_signals.view(np.uint8), axis=1).sum(axis=1)  # (vocab_size,)

        # Backward query (CORRECTED direction):
        # sem_bwd[B_cand's window] XOR codebook[ctx_tok]
        # Low popcount → ctx_tok is a dominant predecessor of B_cand → HIGH score
        ctx_vec = codebook[ctx_tok]                          # (W,) uint64
        sem_bwd_matrix = self.sem_bwd.reshape(self.vocab_size, self.W)  # (vocab_size, W)
        bwd_signals = np.ascontiguousarray(sem_bwd_matrix ^ ctx_vec[None, :])  # (vocab_size, W)
        bwd_pc = np.unpackbits(bwd_signals.view(np.uint8), axis=1).sum(axis=1)  # (vocab_size,)

        # FIXED sign: (neutral - pc) so low XOR popcount → high positive score
        scores = (
            (self._neutral_f - fwd_pc.astype(np.float32))
            + (self._neutral_f - bwd_pc.astype(np.float32))
        ) / self._neutral_f

        return scores  # (vocab_size,)

    def vote_scores_for_context_tok_batch(
        self, ctx_toks: np.ndarray, codebook: np.ndarray
    ) -> np.ndarray:
        """Vectorized batch version: Score all vocab candidates for multiple context tokens.

        Parameters
        ----------
        ctx_toks : (K,) int32
            Array of unique context tokens to compute scores for.
        codebook : (vocab_size, W) uint64
            Token codebook for similarity computation.

        Returns
        -------
        scores : (K, vocab_size) float32
            Score matrix where scores[k] = vote_scores_for_context_tok(ctx_toks[k]).

        FIXES applied (same as vote_scores_for_context_tok):
        1. Sign inversion: (neutral - pc) so low XOR popcount → high positive score.
        2. Wrong backward direction: use sem_bwd[B_cand] XOR codebook[ctx_tok]
           instead of sem_bwd[ctx_tok] XOR codebook[B_cand].
        """
        K = len(ctx_toks)
        vocab_size = codebook.shape[0]
        W = self.W

        # Extract forward windows for all context tokens at once: (K, W) uint64
        win_starts = ctx_toks * W
        win_ends = win_starts + W

        fwd_windows = np.zeros((K, W), dtype=np.uint64)
        for k in range(K):
            fwd_windows[k] = self.sem_fwd[win_starts[k]:win_ends[k]]

        # Forward query: sem_fwd[ctx_tok] XOR codebook[B_cand] → (K, vocab_size, W)
        fwd_signals = fwd_windows[:, None, :] ^ codebook[None, :, :]  # (K, vocab_size, W)
        fwd_pc = np.unpackbits(fwd_signals.view(np.uint8), axis=2).sum(axis=2)  # (K, vocab_size)

        # Backward query (CORRECTED direction):
        # sem_bwd[B_cand's window] XOR codebook[ctx_tok] → (K, vocab_size, W)
        # sem_bwd reshaped to (vocab_size, W); ctx codebook rows are (K, W)
        ctx_vecs = codebook[ctx_toks]                              # (K, W) uint64
        sem_bwd_matrix = self.sem_bwd.reshape(vocab_size, W)      # (vocab_size, W)
        # Broadcast: (1, vocab_size, W) XOR (K, 1, W) → (K, vocab_size, W)
        bwd_signals = sem_bwd_matrix[None, :, :] ^ ctx_vecs[:, None, :]  # (K, vocab_size, W)
        bwd_pc = np.unpackbits(bwd_signals.view(np.uint8), axis=2).sum(axis=2)  # (K, vocab_size)

        # FIXED sign: (neutral - pc) so low XOR popcount → high positive score
        scores = (
            (self._neutral_f - fwd_pc.astype(np.float32))
            + (self._neutral_f - bwd_pc.astype(np.float32))
        ) / self._neutral_f

        return scores  # (K, vocab_size)

    def vote_scores_for_context_tok_gpu(
        self, ctx_tok: int, codebook: np.ndarray, gpu_manager
    ) -> np.ndarray:
        """GPU-accelerated version using TensorCoreGPUManager.

        Uses GPU for the XOR+popcount operations which are the computational bottleneck.
        Falls back to CPU if GPU is not available or on error.

        Parameters
        ----------
        ctx_tok : int
            Context token to query.
        codebook : (vocab_size, W) uint64
            Token codebook for similarity computation.
        gpu_manager : TensorCoreGPUManager
            GPU manager from train_gpt.py for GPU operations.

        Returns
        -------
        scores : (vocab_size,) float32
            Score array for all vocabulary tokens.
        """
        try:
            import cupy as cp

            win = slice(ctx_tok * self.W, (ctx_tok + 1) * self.W)
            fwd_win = self.sem_fwd[win]  # (W,) uint64

            # ctx_tok's own codebook vector — used for the CORRECTED backward query
            ctx_vec = codebook[ctx_tok]  # (W,) uint64

            # Move to GPU
            fwd_win_gpu  = gpu_manager.to_gpu(fwd_win)
            ctx_vec_gpu  = gpu_manager.to_gpu(ctx_vec)
            codebook_gpu = gpu_manager.to_gpu(codebook)

            # Forward query: sem_fwd[ctx_tok] XOR codebook[B_cand] → (vocab_size, W)
            fwd_signals = fwd_win_gpu[None, :] ^ codebook_gpu

            # Backward query (CORRECTED direction):
            # sem_bwd[B_cand's window] XOR codebook[ctx_tok] → (vocab_size, W)
            # Measures "how strongly did ctx_tok precede B_cand?" (correct signal).
            # Original used sem_bwd[ctx_tok] XOR codebook[B_cand] which measured
            # "how similar is B_cand to things that preceded ctx_tok?" (irrelevant noise).
            sem_bwd_cpu = self.sem_bwd.reshape(self.vocab_size, self.W)  # (vocab_size, W)
            sem_bwd_gpu = gpu_manager.to_gpu(sem_bwd_cpu)
            bwd_signals = sem_bwd_gpu ^ ctx_vec_gpu[None, :]             # (vocab_size, W)

            # Popcount on GPU using vectorized popcount
            # cupy doesn't have unpackbits, so we use a manual popcount
            # For uint64, popcount = sum of bits set
            def gpu_popcount_uint64(arr):
                """Vectorized popcount for uint64 array on GPU.

                Fix #8: arr has shape (vocab_size, W) as uint64.  Calling
                arr.view(cp.uint8) on a 2-D contiguous array produces a 1-D
                view of length vocab_size*W*8 — the first dimension is NOT
                preserved.  We must reshape explicitly after the view.
                """
                rows = arr.shape[0]
                # Ensure contiguous memory before reinterpreting dtype
                arr_c = cp.ascontiguousarray(arr)
                x = arr_c.view(cp.uint8).reshape(rows, -1)  # (vocab_size, W*8)
                try:
                    bits = cp.unpackbits(x, axis=1)  # (vocab_size, W*64)
                    return bits.sum(axis=1)
                except (AttributeError, NotImplementedError):
                    # Fallback: use CPU for popcount
                    return None

            fwd_pc = gpu_popcount_uint64(fwd_signals)
            bwd_pc = gpu_popcount_uint64(bwd_signals)

            if fwd_pc is None or bwd_pc is None:
                # GPU popcount failed, fall back to CPU
                return self.vote_scores_for_context_tok(ctx_tok, codebook)

            # Compute scores on GPU
            # FIXED sign: (neutral - pc) so low XOR popcount → high positive score
            neutral = self._neutral_f
            scores = ((neutral - fwd_pc.astype(cp.float32)) +
                      (neutral - bwd_pc.astype(cp.float32))) / neutral

            return gpu_manager.to_cpu(scores).astype(np.float32)

        except (ImportError, RuntimeError, Exception):
            # GPU not available or error occurred, fall back to CPU
            return self.vote_scores_for_context_tok(ctx_tok, codebook)

    # ------------------------------------------------------------------
    # Sleep / consolidation
    # ------------------------------------------------------------------

    def slow_wave(self, noise_threshold: float = 0.15) -> Tuple[int, int]:
        """Decay per-token-window signals that are near-neutral toward neutral.

        Unlike the original slow_wave_consolidation which iterates scalar
        uint64 elements, this operates on W-element windows so confidence
        is measured over 1024 bits (not 64), giving a much more reliable
        signal-vs-noise distinction.

        Improvement #18: the inner per-uint64 loop is replaced with a
        vectorised NumPy random bit-flip using np.random.randint + masking.

        Returns (windows_pruned, windows_nudged) — one window = one token.
        """
        pruned = 0
        nudged = 0
        neutral = self._neutral  # 32 * W

        for tok in range(self.vocab_size):
            win = slice(tok * self.W, (tok + 1) * self.W)

            for vec in (self.sem_fwd, self.sem_bwd):
                block = vec[win].copy()  # (W,) uint64 — work on a copy
                pc = int(np.unpackbits(block.view(np.uint8)).sum())
                conf = abs(pc - neutral) / neutral

                if conf < noise_threshold:
                    # Vectorised: pick one random bit position per uint64 element
                    bit_positions = np.random.randint(0, 64, size=self.W)  # (W,)
                    masks = np.array([np.uint64(1) << np.uint64(b) for b in bit_positions],
                                     dtype=np.uint64)                       # (W,)
                    if pc > neutral:
                        # Too many ones — clear the chosen bits that are currently set
                        set_bits = (block & masks) != np.uint64(0)
                        block[set_bits] &= ~masks[set_bits]
                        vec[win] = block
                        pruned += 1
                    elif pc < neutral:
                        # Too many zeros — set the chosen bits that are currently clear
                        clear_bits = (block & masks) == np.uint64(0)
                        block[clear_bits] |= masks[clear_bits]
                        vec[win] = block
                        nudged += 1

        return pruned, nudged

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

    # ------------------------------------------------------------------
    # Skip-bigram lag vectors (lags 2–5)
    # ------------------------------------------------------------------

    def build_skip_bigram_lags(
        self,
        tokens: np.ndarray,
        codebook: np.ndarray,
        max_lag: int = 5,
        time_budget_s: float = 20.0,
        chunk_size: int = 8_000_000,
        label: str = "SkipBigram",
    ) -> None:
        """Build skip-bigram lag vectors for lags 2 through max_lag.

        Uses the same chunked vectorised numpy reduceat as build_from_tokens —
        no GPU round-trips.  Each lag is ~62 chunks of 8M tokens, each chunk
        one argsort + reduceat call (~50 ms), giving ~3–4 s per lag.
        """
        N = len(tokens)
        start = time.time()

        self.sem_fwd_lag: dict = {}
        for lag in range(2, max_lag + 1):
            self.sem_fwd_lag[lag] = np.zeros(self.uint64_count, dtype=np.uint64)

        print(f"\n[{label}] Building skip-bigram lags 2..{max_lag} "
              f"(N={N:,}, vocab={self.vocab_size}, W={self.W})")

        # Pre-cast once
        tokens_i32 = tokens.astype(np.int32)

        for lag in range(2, max_lag + 1):
            if time.time() - start > time_budget_s:
                print(f"[{label}] Time budget reached at lag={lag}")
                break

            M = N - lag
            a_toks = tokens_i32[:M]
            b_toks = tokens_i32[lag:]
            lag_2d = self.sem_fwd_lag[lag].reshape(self.vocab_size, self.W)
            self._scatter_xor_fast(lag_2d, a_toks, b_toks, codebook, chunk_size)

            elapsed = time.time() - start
            print(f"[{label}] lag={lag} done in {elapsed:.2f}s")

        elapsed = time.time() - start
        print(f"[{label}] All lags built in {elapsed:.2f}s | "
              f"{(max_lag - 1) * self.uint64_count * 8 // 1024} KB total")

    def get_lag_matrix(self, lag: int) -> np.ndarray:
        """Return sem_fwd_lag[lag] reshaped to (vocab_size, W).

        Returns zeros array if lag not built.
        """
        if not hasattr(self, 'sem_fwd_lag') or lag not in self.sem_fwd_lag:
            return np.zeros((self.vocab_size, self.W), dtype=np.uint64)
        return self.sem_fwd_lag[lag].reshape(self.vocab_size, self.W)

    # ------------------------------------------------------------------
    # XOR orbit diagonal table R[k]
    # ------------------------------------------------------------------

    def build_xor_orbit_table(
        self,
        tokens: np.ndarray,
        codebook: np.ndarray,
        threshold: int = 3,
        time_budget_s: float = 10.0,
        label: str = "XOROrbit",
    ) -> None:
        """Build XOR orbit diagonal table R[k].

        R[k] = XOR-bundle of codebook[s] for all (t, s) bigram pairs where:
            t XOR s == k  (same XOR orbit)
            bigram_count[t, s] > threshold

        R[k] encodes: "what semantic jump does XOR offset k represent?"

        For a BPE tokenizer with regularities in ID assignment (morphological
        variants, related words), R[k] for small k encodes structured semantic
        relationships. For random k, R[k] is near-uniform noise.

        At query time: diagonal_prediction(S_p, R) finds which XOR offset
        the current context is traveling along, then predicts
        recent_token XOR winning_k.

        Storage: vocab_size × W × 8 bytes = 1024 × 16 × 8 = 128 KB

        Parameters
        ----------
        tokens       : (N,) uint16 — full token array
        codebook     : (vocab_size, W) uint64
        threshold    : int — minimum bigram count to include a pair
        time_budget_s: float — soft wall-clock limit
        label        : str — log prefix
        """
        N = len(tokens)
        start = time.time()

        print(f"\n[{label}] Building XOR orbit diagonal table "
              f"(vocab={self.vocab_size}, W={self.W}, threshold={threshold})")

        # R[k] has same shape as one token window: (vocab_size, W)
        self.xor_orbit_R = np.zeros((self.vocab_size, self.W), dtype=np.uint64)
        bigram_counts: dict = {}   # (t, s) → count

        # Count bigrams
        a_toks = tokens[:N - 1].astype(np.int32)
        b_toks = tokens[1:].astype(np.int32)

        if time.time() - start > time_budget_s:
            print(f"[{label}] Time budget reached before counting")
            return

        # Vectorised bigram counting using np.unique on pairs
        pairs = a_toks.astype(np.int64) * self.vocab_size + b_toks.astype(np.int64)
        unique_pairs, counts = np.unique(pairs, return_counts=True)

        if time.time() - start > time_budget_s:
            print(f"[{label}] Time budget reached after counting")
            return

        # Build R[k] from frequent bigrams
        for pair_val, count in zip(unique_pairs, counts):
            if count <= threshold:
                continue
            t = int(pair_val // self.vocab_size)
            s = int(pair_val % self.vocab_size)
            k = t ^ s   # XOR orbit offset
            if 0 <= k < self.vocab_size:
                self.xor_orbit_R[k] ^= codebook[s]

        elapsed = time.time() - start
        filled = int(np.any(self.xor_orbit_R != 0, axis=1).sum())
        print(f"[{label}] Done in {elapsed:.2f}s | "
              f"{filled}/{self.vocab_size} orbit slots filled | "
              f"{self.vocab_size * self.W * 8 // 1024} KB")

    # ------------------------------------------------------------------
    # Pre-training semantic prior (frozen)
    # ------------------------------------------------------------------

    @classmethod
    def build_pretrain_prior(
        cls,
        tokens: np.ndarray,
        codebook: np.ndarray,
        vocab_size: int,
        W: int,
        uint64_count: int,
        n_tokens: int = 2_000_000,
        label: str = "SemanticPrior",
    ) -> "DirectionalSemanticVec":
        """Build a frozen pre-training semantic prior from the first n_tokens.

        This prior is computed BEFORE Phase 2 touches anything, from a clean
        statistical pass over the raw corpus. It has no knowledge of bucket
        assignments, seeds, or collision patterns.

        When Phase 4 consults it, it gets an opinion from something that has
        never been exposed to training noise — which is exactly what error
        correction needs.

        The returned DSV is intended to be FROZEN — never modified by
        Phase 2/3/4. Its independence from training noise is its value.

        Parameters
        ----------
        tokens       : (N,) uint16 — full token array
        codebook     : (vocab_size, W) uint64
        vocab_size   : int
        W            : int — uint64 blocks per token window
        uint64_count : int — total uint64 count (= vocab_size * W)
        n_tokens     : int — number of tokens to use (default 2M)
        label        : str — log prefix

        Returns
        -------
        DirectionalSemanticVec — frozen prior (sem_fwd + sem_bwd)
        """
        start = time.time()
        sample = tokens[:min(n_tokens, len(tokens))]
        N = len(sample)

        print(f"\n[{label}] Building pre-training semantic prior "
              f"(n_tokens={N:,}, vocab={vocab_size}, W={W})")

        prior = cls(vocab_size, W, uint64_count)

        # Single lag-1 pass over the sample
        a_toks = sample[:N - 1].astype(np.int32)
        b_toks = sample[1:].astype(np.int32)

        prior._scatter_xor(prior.sem_fwd, a_toks, b_toks, codebook)
        prior._scatter_xor(prior.sem_bwd, b_toks, a_toks, codebook)

        elapsed = time.time() - start
        print(f"[{label}] Done in {elapsed:.2f}s | "
              f"sem_prior_fwd+bwd = {2 * uint64_count * 8 // 1024} KB")
        return prior

    def build_correction_map(
        self,
        vocab_size: int,
        k_neighbors: int = 8,
        label: str = "CorrectionMap",
    ) -> dict:
        """Build a correction map: token → k nearest semantic neighbors.

        Uses one-step gradient in token ID space: for each token t, evaluates
        all single-bit-flip neighbors (flip each bit of t's ID) and measures
        sem_fwd similarity.

        This is the token-space analog of the seed gradient search:
        instead of flipping bits of the seed, we flip bits of the token ID
        and find which neighbors are semantically closest.

        Storage: vocab_size × k_neighbors × ~4 bytes = 1024 × 8 × 4 = 32 KB
        Compute: vocab_size × log2(vocab_size) × W ops ≈ instant

        Parameters
        ----------
        vocab_size   : int
        k_neighbors  : int — neighbors to keep per token
        label        : str — log prefix

        Returns
        -------
        Dict[int, List[Tuple[int, float]]]
            correction_map[t] = [(neighbor_token, similarity), ...] sorted desc
        """
        import math
        n_bits = int(math.ceil(math.log2(max(vocab_size, 2))))
        correction_map: dict = {}

        for t in range(vocab_size):
            win_t = self.sem_fwd[t * self.W: (t + 1) * self.W]
            one_hop = []

            # One-step: flip each bit of token ID
            for k in range(n_bits):
                neighbor = t ^ (1 << k)
                if neighbor < 0 or neighbor >= vocab_size:
                    continue
                win_n = self.sem_fwd[neighbor * self.W: (neighbor + 1) * self.W]
                xor = win_t ^ win_n
                bits = int(np.unpackbits(xor.view(np.uint8)).sum())
                sim = 1.0 - bits / (self.W * 64)
                one_hop.append((neighbor, sim))

            # Two-hop for top-4 one-hop neighbors (captures slightly further relationships)
            one_hop.sort(key=lambda x: x[1], reverse=True)
            two_hop = []
            for neighbor, sim in one_hop[:4]:
                win_n = self.sem_fwd[neighbor * self.W: (neighbor + 1) * self.W]
                for k in range(n_bits):
                    two_neighbor = neighbor ^ (1 << k)
                    if two_neighbor < 0 or two_neighbor >= vocab_size or two_neighbor == t:
                        continue
                    win_2n = self.sem_fwd[two_neighbor * self.W: (two_neighbor + 1) * self.W]
                    xor = win_t ^ win_2n
                    bits = int(np.unpackbits(xor.view(np.uint8)).sum())
                    sim2 = (1.0 - bits / (self.W * 64)) * 0.7   # discount 2-hop
                    two_hop.append((two_neighbor, sim2))

            # Merge, deduplicate, sort
            all_candidates = one_hop + two_hop
            seen: set = set()
            unique: list = []
            for tok, conf in sorted(all_candidates, key=lambda x: x[1], reverse=True):
                if tok not in seen:
                    seen.add(tok)
                    unique.append((tok, conf))

            correction_map[t] = unique[:k_neighbors]

        print(f"[{label}] Correction map built: {vocab_size} tokens × "
              f"{k_neighbors} neighbors = "
              f"{vocab_size * k_neighbors * 4 // 1024} KB")
        return correction_map

    def build_token_distributions(
        self,
        vocab_size: int,
        codebook: np.ndarray,
        top_k: int = 8,
        label: str = "TokenDist",
    ) -> dict:
        """Pre-compute P(next | t) as sparse distributions for all tokens.

        For each token t, uses WHT over sem_fwd[t] to get all next-token
        correlations, then keeps the top-k with butterfly consistency filtering.

        Used in Phase 2 conflict resolution and Phase 3 repair queue annotation
        as an uncontaminated prior over likely next tokens.

        Storage: vocab_size × top_k × ~4 bytes = 1024 × 8 × 4 = 32 KB
        Compute: vocab_size × O(vocab × log(vocab)) WHT passes

        Parameters
        ----------
        vocab_size : int
        codebook   : (vocab_size, W) uint64
        top_k      : int — candidates to keep per token
        label      : str — log prefix

        Returns
        -------
        Dict[int, List[Tuple[int, float]]]
            prior_distributions[t] = [(next_token, probability), ...] sorted desc
        """
        try:
            from _semantic_rolling_hash import wht_vectorised as _wht, bipolar as _bipolar
        except ImportError:
            def _bipolar(hv):
                bits = np.unpackbits(hv.view(np.uint8))
                return bits.astype(np.float32) * 2.0 - 1.0

            def _wht(x):
                x = x.copy()
                n = len(x)
                h = 1
                while h < n:
                    x_r = x.reshape(-1, 2 * h)
                    u = x_r[:, :h].copy()
                    v = x_r[:, h:].copy()
                    x_r[:, :h] = u + v
                    x_r[:, h:] = u - v
                    x = x_r.reshape(-1)
                    h *= 2
                return x

        distributions: dict = {}

        for t in range(vocab_size):
            win = self.sem_fwd[t * self.W: (t + 1) * self.W]
            bipolar_win = _bipolar(win)
            correlations = _wht(bipolar_win)[:vocab_size] / float(len(bipolar_win))

            top_k_indices = np.argsort(correlations)[-top_k:][::-1]
            top_k_corrs   = correlations[top_k_indices]

            # Butterfly consistency filter — only keep tokens with clean signals
            valid = []
            for tok_idx, corr in zip(top_k_indices, top_k_corrs):
                if corr <= 0:
                    continue
                # Simple butterfly check
                n_levels = min(10, int(np.log2(max(vocab_size, 2))))
                partner_ratios = []
                for k in range(n_levels):
                    partner = int(tok_idx) ^ (1 << k)
                    if 0 <= partner < vocab_size:
                        partner_ratios.append(
                            abs(float(correlations[partner])) / (abs(float(corr)) + 1e-8)
                        )
                consistency = max(0.0, 1.0 - max(partner_ratios)) if partner_ratios else 1.0
                if consistency > 0.5:
                    valid.append((int(tok_idx), float(corr) * consistency))

            # Normalise to probabilities
            if valid:
                total = sum(v for _, v in valid)
                if total > 0:
                    valid = [(tok, v / total) for tok, v in valid]

            distributions[t] = valid

        filled = sum(1 for v in distributions.values() if v)
        print(f"[{label}] Token distributions built: "
              f"{filled}/{vocab_size} tokens have prior | "
              f"~{vocab_size * top_k * 4 // 1024} KB")
        return distributions
