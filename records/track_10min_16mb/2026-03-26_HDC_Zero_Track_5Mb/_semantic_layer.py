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
        """
        K = len(ctx_toks)
        vocab_size = codebook.shape[0]
        W = self.W

        # Extract windows for all context tokens at once: (K, W) uint64
        win_starts = ctx_toks * W
        win_ends = win_starts + W

        # Use advanced indexing to gather windows
        fwd_windows = np.zeros((K, W), dtype=np.uint64)
        bwd_windows = np.zeros((K, W), dtype=np.uint64)
        for k in range(K):
            fwd_windows[k] = self.sem_fwd[win_starts[k]:win_ends[k]]
            bwd_windows[k] = self.sem_bwd[win_starts[k]:win_ends[k]]

        # Broadcast XOR: (K, vocab_size, W)
        # fwd_windows[:, None, :] broadcasts to (K, 1, W), codebook broadcasts to (vocab_size, W)
        fwd_signals = fwd_windows[:, None, :] ^ codebook[None, :, :]  # (K, vocab_size, W)
        bwd_signals = bwd_windows[:, None, :] ^ codebook[None, :, :]  # (K, vocab_size, W)

        # Popcount via uint8 view + unpackbits
        # (K, vocab_size, W) uint64 → (K, vocab_size, W*8) uint8 → (K, vocab_size, W*64) bits
        fwd_pc = np.unpackbits(fwd_signals.view(np.uint8), axis=2).sum(axis=2)  # (K, vocab_size)
        bwd_pc = np.unpackbits(bwd_signals.view(np.uint8), axis=2).sum(axis=2)  # (K, vocab_size)

        # Signed score: positive = co-occurrence, negative = anti-correlation
        scores = (
            (fwd_pc.astype(np.float32) - self._neutral_f)
            + (bwd_pc.astype(np.float32) - self._neutral_f)
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
            bwd_win = self.sem_bwd[win]  # (W,) uint64

            # Move to GPU
            fwd_win_gpu = gpu_manager.to_gpu(fwd_win)
            bwd_win_gpu = gpu_manager.to_gpu(bwd_win)
            codebook_gpu = gpu_manager.to_gpu(codebook)

            # Broadcast XOR on GPU: (vocab_size, W)
            fwd_signals = fwd_win_gpu[None, :] ^ codebook_gpu
            bwd_signals = bwd_win_gpu[None, :] ^ codebook_gpu

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
            neutral = self._neutral_f
            scores = ((fwd_pc.astype(cp.float32) - neutral) +
                      (bwd_pc.astype(cp.float32) - neutral)) / neutral

            return gpu_manager.to_cpu(scores).astype(np.float32)

        except (ImportError, RuntimeError, Exception):
            # GPU not available or error occurred, fall back to CPU
            return self.vote_scores_for_context_tok(ctx_tok, codebook)

    # ------------------------------------------------------------------
    # Shared accumulation helper (Bug #18 fix — DRY)
    # ------------------------------------------------------------------

    def _accumulate_sem_votes(
        self,
        context_matrix: np.ndarray,  # (CTX_LEN, chunk_n) int32
        codebook: np.ndarray,         # (vocab_size, W) uint64
        chunk_n: int,
    ) -> np.ndarray:
        """Accumulate semantic vote scores for all positions — fully vectorized.

        Replaces the O(K × CTX_LEN) Python double-loop that appeared in both
        ``augment_predictions`` and ``continuous_attention_blend`` (Bug #17 and
        Bug #18).

        Algorithm
        ---------
        1. Find all unique context tokens K across the whole context_matrix.
        2. Batch-compute scores for those K tokens: (K, vocab_size) float32.
        3. For each context position c, build an index array mapping each of the
           chunk_n positions to its row in the batch_scores matrix, then use
           ``np.add.at`` to scatter-add the scores — no Python loop over K.

        Returns
        -------
        sem_vote : (chunk_n, vocab_size) float32
        """
        all_ctx_toks = np.unique(context_matrix)
        all_ctx_toks = all_ctx_toks[all_ctx_toks >= 0]  # strip padding (-1)

        sem_vote = np.zeros((chunk_n, self.vocab_size), dtype=np.float32)
        if len(all_ctx_toks) == 0:
            return sem_vote

        # (K, vocab_size) — one score row per unique context token
        batch_scores = self.vote_scores_for_context_tok_batch(all_ctx_toks, codebook)

        # Build a reverse-lookup array: tok_id → row index in batch_scores.
        # We use a dense array sized to the max token id for O(1) lookup.
        max_tok = int(all_ctx_toks.max()) + 1
        tok_to_row = np.full(max_tok, -1, dtype=np.int32)
        tok_to_row[all_ctx_toks] = np.arange(len(all_ctx_toks), dtype=np.int32)

        ctx_len = context_matrix.shape[0]
        for c in range(ctx_len):
            ctx_slice = context_matrix[c]  # (chunk_n,) int32
            # Map each position's context token to its batch_scores row index.
            # Tokens outside [0, max_tok) or padding (-1) map to row -1 → skip.
            valid = (ctx_slice >= 0) & (ctx_slice < max_tok)
            if not np.any(valid):
                continue
            row_idx = np.where(valid, tok_to_row[np.clip(ctx_slice, 0, max_tok - 1)], -1)
            valid &= (row_idx >= 0)
            if not np.any(valid):
                continue
            # scatter-add: sem_vote[pos] += batch_scores[row_idx[pos]]
            # np.add.at handles repeated indices correctly.
            np.add.at(sem_vote, np.where(valid)[0], batch_scores[row_idx[valid]])

        return sem_vote

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

        # Accumulate semantic vote scores over all context positions (VECTORIZED)
        ctx_len = context_matrix.shape[0]

        # Bug #17 fix: replace the O(K × CTX_LEN) Python inner loop with
        # fully vectorized advanced indexing.  See _accumulate_sem_votes().
        sem_vote = self._accumulate_sem_votes(context_matrix, codebook, chunk_n)

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

    def continuous_attention_blend(
        self,
        table_preds: np.ndarray,    # (chunk_n,) uint16 — Boyer-Moore predictions
        table_conf: np.ndarray,    # (chunk_n,) int32 — Boyer-Moore confidence
        context_matrix: np.ndarray,# (CTX_LEN, chunk_n) int32 — context tokens
        codebook: np.ndarray,      # (vocab_size, W) uint64
        blend_mode: str = "confidence_weighted",
        max_table_weight: float = 0.85,
        min_sem_weight: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous attention blending — semantic layer contributes at ALL confidence levels.

        This transforms the semantic layer from a fallback mechanism into a continuous
        attention-like system that always contributes to predictions, similar to how
        transformer attention combines multiple heads.

        Blending Modes
        --------------
        "confidence_weighted": Weight = sigmoid(table_conf). High conf → more table weight.
        "additive": Semantic scores added to table confidence for each token.
        "multiplicative": Table confidence multiplied by semantic agreement.
        "cluster_enhanced": Semantic layer clusters relationships and amplifies patterns.

        Parameters
        ----------
        table_preds : (chunk_n,) uint16
            Predictions from Boyer-Moore table lookup.
        table_conf : (chunk_n,) int32
            Confidence values from Boyer-Moore (higher = more confident).
        context_matrix : (CTX_LEN, chunk_n) int32
            Context tokens for each position.
        codebook : (vocab_size, W) uint64
            Token codebook for similarity computation.
        blend_mode : str
            How to blend table and semantic predictions.
        max_table_weight : float
            Maximum weight given to table predictions (even at high confidence).
        min_sem_weight : float
            Minimum weight given to semantic layer (even at high table confidence).

        Returns
        -------
        blended_preds : (chunk_n,) uint16
            Final predictions after blending.
        blend_weights : (chunk_n,) float32
            Weight given to table predictions (1 - weight = semantic weight).
        sem_vote_scores : (chunk_n, vocab_size) float32
            Full semantic vote matrix for analysis.
        """
        chunk_n = len(table_preds)

        # 1. Compute semantic vote scores for ALL positions (VECTORIZED)
        # Bug #18 fix: share the accumulation logic with augment_predictions via
        # _accumulate_sem_votes() instead of duplicating the inner loop here.
        sem_vote = self._accumulate_sem_votes(context_matrix, codebook, chunk_n)

        # Normalize semantic votes to [-1, 1] range per position
        sem_max = np.abs(sem_vote).max(axis=1, keepdims=True)
        sem_max = np.maximum(sem_max, 1e-6)  # Avoid division by zero
        sem_vote_norm = sem_vote / sem_max

        # 2. Compute blend weights based on table confidence
        # Higher confidence → more weight to table, but semantic ALWAYS contributes
        # sigmoid-like weighting: w_table = max_table_weight * sigmoid(conf)
        conf_scaled = np.clip(table_conf / 10.0, -2, 2)  # Scale to reasonable range
        table_weight = max_table_weight * (1.0 / (1.0 + np.exp(-conf_scaled)))
        table_weight = np.clip(table_weight, min_sem_weight, max_table_weight)

        # 3. Get best semantic predictions
        sem_preds = np.argmax(sem_vote, axis=1).astype(np.uint16)
        sem_best_score = sem_vote[np.arange(chunk_n), sem_preds.astype(np.int64)]

        # 4. Blend based on mode
        blended_preds = table_preds.copy()

        if blend_mode == "confidence_weighted":
            # Use semantic prediction when it strongly disagrees with table
            # AND has high semantic confidence
            table_agrees = (sem_preds == table_preds)
            sem_confident = sem_best_score > SEM_CONFIDENCE_MIN

            # Blend: use semantic when table is uncertain OR semantic is very confident
            use_semantic = (~table_agrees) & sem_confident & (table_weight < 0.7)
            blended_preds[use_semantic] = sem_preds[use_semantic]

        elif blend_mode == "additive":
            # Improvement #19: fully vectorised — no Python loop.
            # For each position, build a combined score = sem_vote * 5 + table_conf
            # boosted at the table-predicted token, then argmax.
            idx = np.arange(chunk_n)
            tok_ids = table_preds.astype(np.int64)                    # (chunk_n,)
            sem_vote_scaled = sem_vote * 5.0                           # (chunk_n, vocab)
            # Add table_conf only at the table-predicted token column
            sem_vote_adjusted = sem_vote_scaled.copy()
            sem_vote_adjusted[idx, tok_ids] += table_conf.astype(np.float32)
            best_combined = np.argmax(sem_vote_adjusted, axis=1)       # (chunk_n,)
            # Override where the best combined token beats the table token by > 0.5
            gain = sem_vote[idx, best_combined] - sem_vote[idx, tok_ids]
            override = gain > 0.5
            blended_preds[override] = best_combined[override].astype(np.uint16)

        elif blend_mode == "multiplicative":
            # Improvement #19: fully vectorised — no Python loop.
            # Override where semantic strongly disagrees (norm < -0.3) and is confident.
            tok_ids = table_preds.astype(np.int64)
            agreement = sem_vote_norm[np.arange(chunk_n), tok_ids]    # (chunk_n,)
            override = (agreement < -0.3) & (sem_best_score > SEM_CONFIDENCE_MIN)
            blended_preds[override] = sem_preds[override]

        elif blend_mode == "cluster_enhanced":
            # Improvement #19: fully vectorised — no Python loop.
            tok_ids = table_preds.astype(np.int64)
            sem_score_for_tok = sem_vote[np.arange(chunk_n), tok_ids]  # (chunk_n,)
            has_better = sem_best_score > sem_score_for_tok + 0.3
            low_table_conf = table_weight < 0.6
            override = has_better & low_table_conf
            blended_preds[override] = sem_preds[override]

        return blended_preds, table_weight.astype(np.float32), sem_vote

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

    def crystallized_relationships(
        self, codebook: np.ndarray, threshold: float = 0.6
    ) -> List[Tuple[int, int, float, float, str]]:
        """Return high-confidence token pairs from sem_fwd.

        Returns list of (token_a, token_b, fwd_conf, bwd_conf, rel_type)
        for all pairs where fwd_conf > threshold.

        Improvement #17: replaced the O(vocab²) nested Python loop with a
        fully vectorised broadcast XOR + popcount pass.  For vocab_size=1024
        and W=16 this reduces ~1M Python iterations to a single NumPy call.

        Window size / vocab size method — less expensive approach
        -------------------------------------------------------
        Instead of iterating every (a, b) pair in Python, we:
          1. Reshape sem_fwd into (vocab_size, W) — one row per token.
          2. Broadcast XOR against the full codebook: (vocab_size, 1, W) ^
             (1, vocab_size, W) → (vocab_size, vocab_size, W).
          3. Compute popcount via uint8 view + unpackbits in one call.
          4. Derive fwd_conf for all pairs simultaneously.
          5. Filter with a boolean mask — no Python loop needed.
        Memory: vocab_size² × W × 8 bytes = 1024² × 16 × 8 = 128 MB peak.
        For larger vocabularies, process in row-batches to cap memory.
        """
        V = self.vocab_size
        W = self.W
        neutral_f = self._neutral_f

        # Reshape sem_fwd into (V, W) — one window per token
        fwd_mat = self.sem_fwd.reshape(V, W)   # (V, W) uint64

        # Process in row-batches to keep peak memory ≤ ~64 MB
        BATCH = max(1, min(V, 64 * 1024 * 1024 // (V * W * 8)))

        results = []
        for a_start in range(0, V, BATCH):
            a_end = min(a_start + BATCH, V)
            batch = fwd_mat[a_start:a_end]          # (B, W) uint64

            # Broadcast XOR: (B, 1, W) ^ (1, V, W) → (B, V, W)
            xor_bv = batch[:, None, :] ^ codebook[None, :, :]  # (B, V, W)

            # Popcount via uint8 view
            # xor_bv is (B, V, W) uint64 → view as uint8 → (B, V, W*8)
            xor_u8 = np.ascontiguousarray(xor_bv).view(np.uint8).reshape(
                a_end - a_start, V, W * 8
            )
            pc_bv = np.unpackbits(xor_u8, axis=2).sum(axis=2)  # (B, V) int

            fwd_conf_bv = (pc_bv.astype(np.float32) - neutral_f) / neutral_f  # (B, V)

            # Find pairs above threshold (exclude diagonal a==b)
            above = fwd_conf_bv > threshold
            for bi, a in enumerate(range(a_start, a_end)):
                above[bi, a] = False  # exclude self-pairs
            row_a, col_b = np.where(above)

            for bi, b in zip(row_a, col_b):
                a = a_start + bi
                fwd_conf = float(fwd_conf_bv[bi, b])
                bwd_conf = self.query_backward(int(b), a, codebook)
                if fwd_conf > 0.85 and bwd_conf > 0.85:
                    rel_type = "SYNONYM"
                elif fwd_conf > 0.7 and bwd_conf < 0.3:
                    rel_type = "PRECEDES"
                elif fwd_conf > 0.5:
                    rel_type = "ASSOCIATES-WITH"
                else:
                    rel_type = "AMBIGUOUS"
                results.append((a, int(b), fwd_conf, bwd_conf, rel_type))

        return results

    # ------------------------------------------------------------------
    # Multi-hop / depth inference
    # ------------------------------------------------------------------

    def query_forward_2hop(
        self, token_a: int, codebook: np.ndarray, top_k: int = 5
    ) -> np.ndarray:
        """2-hop forward inference: A → intermediate → C.

        A transformer's depth gives it the ability to form intermediate
        abstract representations: layer 1 learns A→B patterns, layer 2
        learns (A→B)→C patterns.  In HDC algebra this is achievable
        without learned weights because ``sem_fwd[A]`` is itself a
        hypervector in the same space as the codebook.

        Algorithm
        ---------
        Step 1 — find intermediate tokens B whose follower-distribution
                 most resembles A's follower-distribution:

            sim(A, B) = popcount( sem_fwd[A] ^ sem_fwd[B] )

            High popcount (near W*64) → distributions are *opposite*.
            Low popcount (near 0)     → distributions are *identical*.
            We want low XOR popcount, i.e. high similarity.

        Step 2 — for each top-k intermediate B, score all candidates C
                 via the standard 1-hop query on B:

            score_C = popcount( sem_fwd[B] ^ codebook[C] )

        Step 3 — aggregate: sum the 1-hop scores from all top-k
                 intermediates, weighted by their similarity to A.

        This is the HDC equivalent of a 2-layer transformer: the
        intermediate B tokens are the "hidden layer" representations
        learned purely from corpus co-occurrence statistics.

        Parameters
        ----------
        token_a : int
            The context token (the "query").
        codebook : (vocab_size, W) uint64
            Token codebook.
        top_k : int
            Number of intermediate tokens to use (analogous to number
            of "neurons" in a hidden layer).  Default 5.

        Returns
        -------
        scores : (vocab_size,) float32
            Aggregated 2-hop scores for all vocabulary tokens.
            Positive = predicted to follow A via at least one
            intermediate; negative = anti-correlated.
        """
        V = self.vocab_size
        W = self.W
        neutral_f = self._neutral_f

        # ── Step 1: find top-k intermediate tokens B ──────────────────
        # sem_fwd reshaped to (V, W) — one follower-distribution per token
        fwd_mat = self.sem_fwd.reshape(V, W)          # (V, W) uint64
        query_win = fwd_mat[token_a]                   # (W,) uint64

        # XOR query against every row: (V, W) uint64
        xor_ab = query_win[None, :] ^ fwd_mat          # (V, W)
        # Popcount via uint8 view
        pc_ab = np.unpackbits(
            np.ascontiguousarray(xor_ab).view(np.uint8), axis=1
        ).sum(axis=1).astype(np.float32)               # (V,)

        # Similarity score: low XOR popcount = high similarity
        # Normalise to (-1, 1): 0 popcount → +1, W*64 popcount → -1
        sim_ab = (neutral_f - pc_ab) / neutral_f       # (V,) in (-1, 1)
        sim_ab[token_a] = -1.0                         # exclude self

        # Top-k intermediates by similarity
        top_k_actual = min(top_k, V - 1)
        top_b_idx = np.argpartition(sim_ab, -top_k_actual)[-top_k_actual:]
        top_b_sim = sim_ab[top_b_idx]                  # (top_k,) weights

        # ── Step 2 & 3: aggregate 1-hop scores from each intermediate ─
        # For each B in top_k, compute sem_fwd[B] ^ codebook[C] for all C
        # Shape: (top_k, W) ^ (1, V, W) → (top_k, V, W)
        b_windows = fwd_mat[top_b_idx]                 # (top_k, W) uint64
        xor_bc = b_windows[:, None, :] ^ codebook[None, :, :]  # (top_k, V, W)
        pc_bc = np.unpackbits(
            np.ascontiguousarray(xor_bc).view(np.uint8), axis=2
        ).sum(axis=2).astype(np.float32)               # (top_k, V)

        # 1-hop scores for each intermediate: positive = co-occurrence
        scores_bc = (pc_bc - neutral_f) / neutral_f    # (top_k, V)

        # Weight each intermediate's contribution by its similarity to A
        # top_b_sim: (top_k,) → broadcast to (top_k, V)
        weights = np.clip(top_b_sim, 0.0, None)        # only positive sim
        if weights.sum() < 1e-9:
            weights = np.ones(top_k_actual, dtype=np.float32)
        weights = weights / weights.sum()              # normalise

        aggregated = (scores_bc * weights[:, None]).sum(axis=0)  # (V,)
        return aggregated.astype(np.float32)

    def query_forward_nhop(
        self,
        token_a: int,
        codebook: np.ndarray,
        n_hops: int = 2,
        top_k: int = 5,
        blend_direct: float = 0.4,
    ) -> np.ndarray:
        """N-hop forward inference by iterating the 2-hop mechanism.

        Each hop uses the previous hop's aggregated score vector as a
        soft "intermediate token distribution", then queries sem_fwd
        again.  This is the HDC analogue of stacking transformer layers:

            hop 0 (direct):  sem_fwd[A] ^ codebook[C]
            hop 1:           aggregate over top-k B of sem_fwd[B] ^ codebook[C]
            hop 2:           aggregate over top-k B' of sem_fwd[B'] ^ codebook[C]
                             where B' are the top-k tokens from hop 1's scores

        The final score blends all hops with exponentially decaying
        weight so that direct (1-hop) evidence always dominates.

        Parameters
        ----------
        token_a : int
            Context token.
        codebook : (vocab_size, W) uint64
            Token codebook.
        n_hops : int
            Total number of hops (1 = direct only, 2 = one intermediate
            layer, etc.).  Values above 3 rarely help and add cost.
        top_k : int
            Intermediate tokens per hop.
        blend_direct : float
            Weight of the direct (1-hop) score in the final blend.
            Remaining weight is split equally across deeper hops.

        Returns
        -------
        scores : (vocab_size,) float32
            Blended multi-hop scores.
        """
        V = self.vocab_size
        W = self.W
        neutral_f = self._neutral_f
        fwd_mat = self.sem_fwd.reshape(V, W)

        # ── Hop 0: direct 1-hop scores ────────────────────────────────
        query_win = fwd_mat[token_a]                   # (W,)
        xor_direct = query_win[None, :] ^ codebook     # (V, W)
        pc_direct = np.unpackbits(
            np.ascontiguousarray(xor_direct).view(np.uint8), axis=1
        ).sum(axis=1).astype(np.float32)
        direct_scores = (pc_direct - neutral_f) / neutral_f  # (V,)

        if n_hops <= 1:
            return direct_scores.astype(np.float32)

        # ── Deeper hops ───────────────────────────────────────────────
        hop_scores = [direct_scores]
        current_query = query_win.copy()               # (W,) — evolves each hop

        for _hop in range(1, n_hops):
            # Find top-k tokens whose sem_fwd window is most similar to
            # the current query window (the "intermediate layer" tokens)
            xor_ab = current_query[None, :] ^ fwd_mat  # (V, W)
            pc_ab = np.unpackbits(
                np.ascontiguousarray(xor_ab).view(np.uint8), axis=1
            ).sum(axis=1).astype(np.float32)
            sim_ab = (neutral_f - pc_ab) / neutral_f   # (V,)
            sim_ab[token_a] = -1.0                     # exclude origin

            top_k_actual = min(top_k, V - 1)
            top_b_idx = np.argpartition(sim_ab, -top_k_actual)[-top_k_actual:]
            top_b_sim = np.clip(sim_ab[top_b_idx], 0.0, None)

            if top_b_sim.sum() < 1e-9:
                # No positive-similarity intermediates — stop early
                break

            weights = top_b_sim / top_b_sim.sum()

            # Score all C via each intermediate B
            b_windows = fwd_mat[top_b_idx]             # (top_k, W)
            xor_bc = b_windows[:, None, :] ^ codebook[None, :, :]  # (top_k, V, W)
            pc_bc = np.unpackbits(
                np.ascontiguousarray(xor_bc).view(np.uint8), axis=2
            ).sum(axis=2).astype(np.float32)
            scores_bc = (pc_bc - neutral_f) / neutral_f  # (top_k, V)
            hop_score = (scores_bc * weights[:, None]).sum(axis=0)  # (V,)
            hop_scores.append(hop_score)

            # Advance query: weighted average of top-k intermediate windows
            # (soft "residual stream" update — the HDC analogue of a
            # transformer residual connection)
            current_query = (b_windows.astype(np.float64) * weights[:, None]).sum(axis=0)
            # Re-binarise via majority vote (threshold at 0.5 of max)
            threshold = current_query.max() * 0.5
            current_query_bin = np.where(
                current_query >= threshold,
                np.uint64(0xFFFFFFFFFFFFFFFF),
                np.uint64(0),
            ).astype(np.uint64)
            current_query = current_query_bin

        # ── Blend all hops ────────────────────────────────────────────
        n_deep = len(hop_scores) - 1
        if n_deep == 0:
            return hop_scores[0].astype(np.float32)

        deep_weight_each = (1.0 - blend_direct) / n_deep
        blended = hop_scores[0] * blend_direct
        for hs in hop_scores[1:]:
            blended = blended + hs * deep_weight_each

        return blended.astype(np.float32)

    def vote_scores_multihop(
        self,
        ctx_toks: np.ndarray,
        codebook: np.ndarray,
        n_hops: int = 2,
        top_k: int = 5,
        blend_direct: float = 0.4,
    ) -> np.ndarray:
        """Multi-hop vote scores for a batch of context tokens.

        Drop-in replacement for ``vote_scores_for_context_tok_batch``
        that adds N-hop depth.  For each context token, computes
        ``query_forward_nhop`` and returns the (K, vocab_size) matrix.

        Parameters
        ----------
        ctx_toks : (K,) int32
            Unique context tokens.
        codebook : (vocab_size, W) uint64
        n_hops : int
            Depth (1 = same as existing 1-hop batch method).
        top_k : int
            Intermediate tokens per hop.
        blend_direct : float
            Weight of direct 1-hop evidence in the blend.

        Returns
        -------
        scores : (K, vocab_size) float32
        """
        K = len(ctx_toks)
        V = self.vocab_size
        scores = np.zeros((K, V), dtype=np.float32)
        for k, tok in enumerate(ctx_toks):
            scores[k] = self.query_forward_nhop(
                int(tok), codebook,
                n_hops=n_hops, top_k=top_k, blend_direct=blend_direct,
            )
        return scores

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
