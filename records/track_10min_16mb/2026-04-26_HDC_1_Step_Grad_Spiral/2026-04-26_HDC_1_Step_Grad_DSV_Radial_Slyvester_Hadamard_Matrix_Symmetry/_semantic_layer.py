
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

SEM_CONFIDENCE_MIN = 0.15

# ---------------------------------------------------------------------------
# GoldenAxisShift — circular bit rotation for lag-separated DSV encoding
# ---------------------------------------------------------------------------
# phi_offset = round(φ × 64) = 39  (φ = golden ratio ≈ 0.6180339887)
# For lag c: rotate each codebook hypervector by  c × PHI_OFFSET  bits.
# Same irrational-step principle as position-encoding in RoPE / GoldenShift.
# Each lag occupies a non-repeating, Weyl-equidistributed angular sector in
# the W×64-bit hypercube, so lag-1 and lag-2 contributions in sem_fwd are
# geometrically orthogonal.
# ---------------------------------------------------------------------------
_GOLDEN_PHI_OFFSET = 39  # round(golden_ratio * 64)


def _golden_rotate_codebook(
    codebook: np.ndarray,          # (vocab_size, W) uint64
    lag: int,
) -> np.ndarray:
    """Circularly rotate every hypervector in codebook by lag × phi_offset bits.

    Uses full W×64-bit rotation (not a word-level roll) via unpack/roll/repack.
    Only the bits within the W-word hypervector are rotated — no cross-token
    information leakage.

    Args:
        codebook : (vocab_size, W) uint64  — original codebook
        lag      : int — lag index (rotation = lag × PHI_OFFSET % n_bits )

    Returns:
        (vocab_size, W) uint64 — lag-rotated version (new array)
    """
    vocab_size, W = codebook.shape
    n_bits = W * 64
    rot = int(lag * _GOLDEN_PHI_OFFSET) % n_bits
    if rot == 0:
        return codebook.copy()

    # Unpack all bits, roll circularly, repack
    bits = np.unpackbits(
        codebook.view(np.uint8).reshape(vocab_size, W * 8),
        axis=1, bitorder='big',
    )                                                   # (vocab_size, n_bits)
    bits_rot = np.roll(bits, -rot, axis=1)              # left-rotate by rot
    rotated = (
        np.packbits(bits_rot, axis=1, bitorder='big')
        .view(np.uint64)
        .reshape(vocab_size, W)
        .copy()
    )
    return rotated


def build_golden_codebook_table(
    codebook: np.ndarray,
    max_lag: int = 5,
) -> List[np.ndarray]:
    """Precompute rotated codebooks for lags 1..max_lag.

    Returns a list where entry c-1 (0-indexed) = rotate(codebook, lag=c).
    Used at both build-time (scatter_xor) and eval-time (XOR query).
    """
    return [_golden_rotate_codebook(codebook, c) for c in range(1, max_lag + 1)]


class DirectionalSemanticVec:

    def __init__(self, vocab_size: int, W: int, uint64_count: int) -> None:
        if vocab_size * W != uint64_count:
            raise ValueError(
                f"Token-addressed tiling requires vocab_size*W == uint64_count, "
                f"got {vocab_size}*{W}={vocab_size*W} vs {uint64_count}."
            )
        self.vocab_size = vocab_size
        self.W = W
        self.uint64_count = uint64_count
        self._neutral = 32 * W
        self._neutral_f = float(self._neutral)

        self.sem_fwd = np.zeros(uint64_count, dtype=np.uint64)
        self.sem_bwd = np.zeros(uint64_count, dtype=np.uint64)

    @staticmethod
    def _scatter_xor_fast(
        vec_2d: np.ndarray,
        a_toks: np.ndarray,
        b_toks: np.ndarray,
        codebook: np.ndarray,
        chunk_size: int = 8_000_000,
    ) -> None:
        M = len(a_toks)
        if M == 0:
            return
        for cs in range(0, M, chunk_size):
            ce = min(cs + chunk_size, M)
            a_c = a_toks[cs:ce]
            b_c = b_toks[cs:ce]
            order   = a_c.argsort(kind="stable")
            a_sort  = a_c[order]
            b_sort  = b_c[order]
            vecs    = codebook[b_sort]
            unique_a, first_idx = np.unique(a_sort, return_index=True)
            bundles = np.bitwise_xor.reduceat(vecs, first_idx, axis=0)
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
        verbose: bool = True,
        use_golden_axis: bool = False,
    ) -> "DirectionalSemanticVec":
        """Build forward/backward XOR-bundle DSV tables from token sequence.

        use_golden_axis=False (default — recommended for sem_fwd):
            Classic unrotated XOR-bundle.  ALL lags 1..ctx_len contribute to the
            SAME codebook direction for each token → constructive interference for
            common bigrams → maximum prediction power.
            Used for the main sem_fwd where multi-lag reinforcement is beneficial.

        use_golden_axis=True (recommended for skip-bigrams via build_skip_bigram_lags):
            Apply GoldenAxisShift circular rotation per lag.  Only for SINGLE-LAG
            arrays (each `sem_fwd_lag[c]` has one lag) where there is no multi-lag
            mixing → no orthogonal noise.  Gives each lag a distinct angular subspace
            for geometric separability and multi-agent / multi-dimensional queries.
        """
        dsv = cls(vocab_size, W, uint64_count)
        N   = len(tokens)
        start = time.time()

        # Precompute rotated codebooks for each lag (once, outside the loop)
        if use_golden_axis:
            rotated_cbs = build_golden_codebook_table(codebook, max_lag=ctx_len)
            if verbose:
                print(f"\n[{label}] Building GoldenAxisShift DSV "
                      f"(vocab={vocab_size}, W={W}, ctx_len={ctx_len}, "
                      f"phi_offset={_GOLDEN_PHI_OFFSET})")
        else:
            rotated_cbs = None
            if verbose:
                print(f"\n[{label}] Building directional semantic vectors "
                      f"(vocab={vocab_size}, W={W}, ctx_len={ctx_len})")

        tokens_i32 = tokens.astype(np.int32)
        sf_2d = dsv.sem_fwd.reshape(vocab_size, W)
        sb_2d = dsv.sem_bwd.reshape(vocab_size, W)
        total_pairs = 0

        # GoldenAxisShift: ALL lags 1..ctx_len are merged into sem_fwd, each
        # occupying a Weyl-equidistributed geometric subspace (lag-c = rotation
        # c×phi_offset).  This encodes multi-lag temporal context in ONE vector,
        # allowing any subset of lags to be queried independently at eval time —
        # the "multi-dimensional / hivemind" property.
        # At eval time, query EACH lag rotation and blend to extract all subspaces.
        # (Querying only one rotation wastes the other lags; blending is necessary.)
        for c in range(1, ctx_len + 1):
            if time.time() - start > time_budget_s:
                if verbose:
                    print(f"[{label}] Time budget reached at context depth c={c-1}")
                break
            cb = rotated_cbs[c - 1] if rotated_cbs is not None else codebook
            M = N - c
            a_toks = tokens_i32[:M]
            b_toks = tokens_i32[c:]
            cls._scatter_xor_fast(sf_2d, a_toks, b_toks, cb, chunk_size)
            # sem_bwd intentionally skipped: hash_grad_bpb() only queries sem_fwd
            # (collision + miss paths both index sem_fwd[prev_t], never sem_bwd).
            # Removing this call saves exactly 50% of the DSV build time (~155s for
            # 80-shard 125M-token-per-rank build), bringing training under 600s.
            # sem_bwd is left as all-zeros (not used in eval).
            total_pairs += M
            elapsed = time.time() - start
            if verbose:
                print(f"[{label}] c={c}/{ctx_len}  pairs={total_pairs:,}  "
                      f"elapsed={elapsed:.1f}s")
        elapsed = time.time() - start
        if verbose:
            print(f"[{label}] Done. {total_pairs:,} A→B pairs recorded in {elapsed:.1f}s")
        # Store golden-axis flag for eval-time callers
        dsv._use_golden_axis = use_golden_axis
        dsv._ctx_len          = ctx_len
        return dsv

    @staticmethod
    def _scatter_xor(
        vec: np.ndarray,
        index_toks: np.ndarray,
        value_toks: np.ndarray,
        codebook: np.ndarray,
    ) -> None:
        """Thin wrapper around _scatter_xor_fast for callers that hold a 1-D vec."""
        if len(index_toks) == 0:
            return
        W          = codebook.shape[1]
        vocab_size = codebook.shape[0]
        vec_2d     = vec.reshape(vocab_size, W)
        DirectionalSemanticVec._scatter_xor_fast(vec_2d, index_toks, value_toks, codebook)

    def query_forward(self, token_a: int, token_b: int, codebook: np.ndarray) -> float:
        win = slice(token_a * self.W, (token_a + 1) * self.W)
        signal = self.sem_fwd[win] ^ codebook[token_b]
        pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        return (self._neutral_f - pc) / self._neutral_f

    def query_backward(self, token_b: int, token_a: int, codebook: np.ndarray) -> float:
        win = slice(token_b * self.W, (token_b + 1) * self.W)
        signal = self.sem_bwd[win] ^ codebook[token_a]
        pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        return (self._neutral_f - pc) / self._neutral_f

    def vote_scores_for_context_tok(
        self, ctx_tok: int, codebook: np.ndarray
    ) -> np.ndarray:
        win = slice(ctx_tok * self.W, (ctx_tok + 1) * self.W)
        fwd_win = self.sem_fwd[win]

        fwd_signals = np.ascontiguousarray(fwd_win[None, :] ^ codebook)
        fwd_pc = np.unpackbits(fwd_signals.view(np.uint8), axis=1).sum(axis=1)

        ctx_vec = codebook[ctx_tok]
        sem_bwd_matrix = self.sem_bwd.reshape(self.vocab_size, self.W)
        bwd_signals = np.ascontiguousarray(sem_bwd_matrix ^ ctx_vec[None, :])
        bwd_pc = np.unpackbits(bwd_signals.view(np.uint8), axis=1).sum(axis=1)

        scores = (
            (self._neutral_f - fwd_pc.astype(np.float32))
            + (self._neutral_f - bwd_pc.astype(np.float32))
        ) / self._neutral_f

        return scores

    def vote_scores_for_context_tok_batch(
        self, ctx_toks: np.ndarray, codebook: np.ndarray
    ) -> np.ndarray:
        K = len(ctx_toks)
        vocab_size = codebook.shape[0]
        W = self.W

        win_starts = ctx_toks * W
        win_ends = win_starts + W

        fwd_windows = np.zeros((K, W), dtype=np.uint64)
        for k in range(K):
            fwd_windows[k] = self.sem_fwd[win_starts[k]:win_ends[k]]

        fwd_signals = fwd_windows[:, None, :] ^ codebook[None, :, :]
        fwd_pc = np.unpackbits(fwd_signals.view(np.uint8), axis=2).sum(axis=2)

        ctx_vecs = codebook[ctx_toks]
        sem_bwd_matrix = self.sem_bwd.reshape(vocab_size, W)
        bwd_signals = sem_bwd_matrix[None, :, :] ^ ctx_vecs[:, None, :]
        bwd_pc = np.unpackbits(bwd_signals.view(np.uint8), axis=2).sum(axis=2)

        scores = (
            (self._neutral_f - fwd_pc.astype(np.float32))
            + (self._neutral_f - bwd_pc.astype(np.float32))
        ) / self._neutral_f

        return scores

    def vote_scores_for_context_tok_gpu(
        self, ctx_tok: int, codebook: np.ndarray, gpu_manager
    ) -> np.ndarray:
        try:
            import cupy as cp

            win = slice(ctx_tok * self.W, (ctx_tok + 1) * self.W)
            fwd_win = self.sem_fwd[win]

            ctx_vec = codebook[ctx_tok]

            fwd_win_gpu  = gpu_manager.to_gpu(fwd_win)
            ctx_vec_gpu  = gpu_manager.to_gpu(ctx_vec)
            codebook_gpu = gpu_manager.to_gpu(codebook)

            fwd_signals = fwd_win_gpu[None, :] ^ codebook_gpu

            sem_bwd_cpu = self.sem_bwd.reshape(self.vocab_size, self.W)
            sem_bwd_gpu = gpu_manager.to_gpu(sem_bwd_cpu)
            bwd_signals = sem_bwd_gpu ^ ctx_vec_gpu[None, :]

            def gpu_popcount_uint64(arr):

                rows = arr.shape[0]
                arr_c = cp.ascontiguousarray(arr)
                x = arr_c.view(cp.uint8).reshape(rows, -1)
                try:
                    bits = cp.unpackbits(x, axis=1)
                    return bits.sum(axis=1)
                except (AttributeError, NotImplementedError):
                    return None

            fwd_pc = gpu_popcount_uint64(fwd_signals)
            bwd_pc = gpu_popcount_uint64(bwd_signals)

            if fwd_pc is None or bwd_pc is None:
                return self.vote_scores_for_context_tok(ctx_tok, codebook)

            neutral = self._neutral_f
            scores = ((neutral - fwd_pc.astype(cp.float32)) +
                      (neutral - bwd_pc.astype(cp.float32))) / neutral

            return gpu_manager.to_cpu(scores).astype(np.float32)

        except (ImportError, RuntimeError, Exception):
            return self.vote_scores_for_context_tok(ctx_tok, codebook)

    def slow_wave(self, noise_threshold: float = 0.15) -> Tuple[int, int]:
        pruned = 0
        nudged = 0
        neutral = self._neutral

        for tok in range(self.vocab_size):
            win = slice(tok * self.W, (tok + 1) * self.W)

            for vec in (self.sem_fwd, self.sem_bwd):
                block = vec[win].copy()
                pc = int(np.unpackbits(block.view(np.uint8)).sum())
                conf = abs(pc - neutral) / neutral

                if conf < noise_threshold:
                    bit_positions = np.random.randint(0, 64, size=self.W)
                    masks = np.array([np.uint64(1) << np.uint64(b) for b in bit_positions],
                                     dtype=np.uint64)
                    if pc > neutral:
                        set_bits = (block & masks) != np.uint64(0)
                        block[set_bits] &= ~masks[set_bits]
                        vec[win] = block
                        pruned += 1
                    elif pc < neutral:
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

    def build_skip_bigram_lags(
        self,
        tokens: np.ndarray,
        codebook: np.ndarray,
        max_lag: int = 5,
        time_budget_s: float = 20.0,
        chunk_size: int = 8_000_000,
        label: str = "SkipBigram",
        verbose: bool = True,
        use_golden_axis: bool = True,
    ) -> None:
        """Build skip-bigram lag arrays for lags 2..max_lag.

        use_golden_axis=True (default — independent of sem_fwd default):
          Each single-lag array `sem_fwd_lag[c]` is built with the GoldenAxisShift
          rotation for lag c.  Because these arrays each contain ONLY ONE LAG,
          there is no cross-lag mixing → no orthogonal noise.  Each lag lives in a
          distinct angular subspace (c × phi_offset bits) for clean geometric
          separation — the multi-dimensional / hivemind property.

        use_golden_axis=False:
          Classic unrotated skip-bigrams (original HDC behavior).

        Note: deliberately independent of self._use_golden_axis from build_from_tokens
        so that sem_fwd and skip-bigrams can use different strategies.
        """

        N = len(tokens)
        start = time.time()
        self.sem_fwd_lag: dict = {}
        for lag in range(2, max_lag + 1):
            self.sem_fwd_lag[lag] = np.zeros(self.uint64_count, dtype=np.uint64)

        if use_golden_axis:
            rotated_cbs = build_golden_codebook_table(codebook, max_lag=max_lag)
            if verbose:
                print(f"\n[{label}] Building GoldenAxisShift skip-bigram lags 2..{max_lag} "
                      f"(N={N:,}, vocab={self.vocab_size}, W={self.W}, "
                      f"phi_offset={_GOLDEN_PHI_OFFSET})")
        else:
            rotated_cbs = None
            if verbose:
                print(f"\n[{label}] Building skip-bigram lags 2..{max_lag} "
                      f"(N={N:,}, vocab={self.vocab_size}, W={self.W})")

        tokens_i32 = tokens.astype(np.int32)
        for lag in range(2, max_lag + 1):
            if time.time() - start > time_budget_s:
                if verbose:
                    print(f"[{label}] Time budget reached at lag={lag}")
                break
            cb = rotated_cbs[lag - 1] if rotated_cbs is not None else codebook
            M = N - lag
            a_toks = tokens_i32[:M]
            b_toks = tokens_i32[lag:]
            lag_2d = self.sem_fwd_lag[lag].reshape(self.vocab_size, self.W)
            self._scatter_xor_fast(lag_2d, a_toks, b_toks, cb, chunk_size)
            elapsed = time.time() - start
            if verbose:
                print(f"[{label}] lag={lag} done in {elapsed:.2f}s")
        elapsed = time.time() - start
        if verbose:
            print(f"[{label}] All lags built in {elapsed:.2f}s | "
                  f"{(max_lag - 1) * self.uint64_count * 8 // 1024} KB total")

    def get_lag_matrix(self, lag: int) -> np.ndarray:
        """Return sem_fwd_lag[lag] reshaped to (vocab_size, W).

        Returns zeros array if lag not built.
        """
        if not hasattr(self, 'sem_fwd_lag') or lag not in self.sem_fwd_lag:
            return np.zeros((self.vocab_size, self.W), dtype=np.uint64)
        return self.sem_fwd_lag[lag].reshape(self.vocab_size, self.W)

    def build_xor_orbit_table(
        self,
        tokens: np.ndarray,
        codebook: np.ndarray,
        threshold: int = 3,
        time_budget_s: float = 10.0,
        label: str = "XOROrbit",
    ) -> None:
        N = len(tokens)
        start = time.time()

        print(f"\n[{label}] Building XOR orbit diagonal table "
              f"(vocab={self.vocab_size}, W={self.W}, threshold={threshold})")

        self.xor_orbit_R = np.zeros((self.vocab_size, self.W), dtype=np.uint64)
        bigram_counts: dict = {}

        a_toks = tokens[:N - 1].astype(np.int32)
        b_toks = tokens[1:].astype(np.int32)

        if time.time() - start > time_budget_s:
            print(f"[{label}] Time budget reached before counting")
            return

        pairs = a_toks.astype(np.int64) * self.vocab_size + b_toks.astype(np.int64)
        unique_pairs, counts = np.unique(pairs, return_counts=True)

        if time.time() - start > time_budget_s:
            print(f"[{label}] Time budget reached after counting")
            return

        for pair_val, count in zip(unique_pairs, counts):
            if count <= threshold:
                continue
            t = int(pair_val // self.vocab_size)
            s = int(pair_val % self.vocab_size)
            k = t ^ s
            if 0 <= k < self.vocab_size:
                self.xor_orbit_R[k] ^= codebook[s]

        elapsed = time.time() - start
        filled = int(np.any(self.xor_orbit_R != 0, axis=1).sum())
        print(f"[{label}] Done in {elapsed:.2f}s | "
              f"{filled}/{self.vocab_size} orbit slots filled | "
              f"{self.vocab_size * self.W * 8 // 1024} KB")

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
        start = time.time()
        sample = tokens[:min(n_tokens, len(tokens))]
        N = len(sample)

        print(f"\n[{label}] Building pre-training semantic prior "
              f"(n_tokens={N:,}, vocab={vocab_size}, W={W})")

        prior = cls(vocab_size, W, uint64_count)

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

        import math
        n_bits = int(math.ceil(math.log2(max(vocab_size, 2))))
        correction_map: dict = {}

        for t in range(vocab_size):
            win_t = self.sem_fwd[t * self.W: (t + 1) * self.W]
            one_hop = []

            for k in range(n_bits):
                neighbor = t ^ (1 << k)
                if neighbor < 0 or neighbor >= vocab_size:
                    continue
                win_n = self.sem_fwd[neighbor * self.W: (neighbor + 1) * self.W]
                xor = win_t ^ win_n
                bits = int(np.unpackbits(xor.view(np.uint8)).sum())
                sim = 1.0 - bits / (self.W * 64)
                one_hop.append((neighbor, sim))

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
                    sim2 = (1.0 - bits / (self.W * 64)) * 0.7
                    two_hop.append((two_neighbor, sim2))

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

            valid = []
            for tok_idx, corr in zip(top_k_indices, top_k_corrs):
                if corr <= 0:
                    continue
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
