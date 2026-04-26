
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

SEM_CONFIDENCE_MIN = 0.15

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
    ) -> "DirectionalSemanticVec":
        dsv = cls(vocab_size, W, uint64_count)
        N   = len(tokens)
        start = time.time()
        if verbose:
            print(f"\n[{label}] Building directional semantic vectors "
                  f"(vocab={vocab_size}, W={W}, ctx_len={ctx_len})")
        tokens_i32 = tokens.astype(np.int32)
        sf_2d = dsv.sem_fwd.reshape(vocab_size, W)
        total_pairs = 0
        for c in range(1, ctx_len + 1):
            if time.time() - start > time_budget_s:
                if verbose:
                    print(f"[{label}] Time budget reached at context depth c={c-1}")
                break
            M = N - c
            a_toks = tokens_i32[:M]
            b_toks = tokens_i32[c:]
            cls._scatter_xor_fast(sf_2d, a_toks, b_toks, codebook, chunk_size)
            total_pairs += M
            elapsed = time.time() - start
            if verbose:
                print(f"[{label}] c={c}/{ctx_len}  pairs={total_pairs:,}  "
                      f"elapsed={elapsed:.1f}s")
        elapsed = time.time() - start
        if verbose:
            print(f"[{label}] Done. {total_pairs:,} A→B pairs recorded in {elapsed:.1f}s")
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

    def build_skip_bigram_lags(
        self,
        tokens: np.ndarray,
        codebook: np.ndarray,
        max_lag: int = 5,
        time_budget_s: float = 20.0,
        chunk_size: int = 8_000_000,
        label: str = "SkipBigram",
        verbose: bool = True,
    ) -> None:
        N = len(tokens)
        start = time.time()
        self.sem_fwd_lag: dict = {}
        for lag in range(2, max_lag + 1):
            self.sem_fwd_lag[lag] = np.zeros(self.uint64_count, dtype=np.uint64)
        if verbose:
            print(f"\n[{label}] Building skip-bigram lags 2..{max_lag} "
                  f"(N={N:,}, vocab={self.vocab_size}, W={self.W})")
        tokens_i32 = tokens.astype(np.int32)
        for lag in range(2, max_lag + 1):
            if time.time() - start > time_budget_s:
                if verbose:
                    print(f"[{label}] Time budget reached at lag={lag}")
                break
            M = N - lag
            a_toks = tokens_i32[:M]
            b_toks = tokens_i32[lag:]
            lag_2d = self.sem_fwd_lag[lag].reshape(self.vocab_size, self.W)
            self._scatter_xor_fast(lag_2d, a_toks, b_toks, codebook, chunk_size)
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

    def summary(self) -> dict:
        """Return mean confidence and coverage statistics for sem_fwd."""
        confs = []
        for tok in range(self.vocab_size):
            win = slice(tok * self.W, (tok + 1) * self.W)
            pc = int(np.unpackbits(self.sem_fwd[win].view(np.uint8)).sum())
            confs.append(abs(pc - self._neutral) / self._neutral)
        arr = np.array(confs)
        return {
            "sem_fwd": {
                "mean_confidence": float(arr.mean()),
                "high_conf_tokens": int((arr > 0.5).sum()),
                "neutral_tokens": int((arr < 0.1).sum()),
            }
        }
