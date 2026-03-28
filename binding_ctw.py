"""
binding_ctw.py — Binding-Energy-Modulated Dirichlet CTW

Extends the Dirichlet CTW mixing from PR #986 with context-dependent
concentration parameters derived from epistemic hypergraph binding energy.

Instead of fixed concentration c=5.0 for all contexts:
  c(ctx) = c_base × (1 + β × B(ctx))

where B(ctx) measures the structural coherence of the context tokens:
  - High B → rare, specific context → trust the n-gram more (higher c)
  - Low B → common, ambiguous context → smooth more toward backup (lower c)

This is theoretically grounded: the Dirichlet concentration controls how
much prior mass vs observed counts to trust. Binding energy measures
exactly how informative a context is — the tighter the binding, the more
the observed n-gram counts should dominate.
"""

import math
import numpy as np
from collections import Counter
from typing import Optional, Tuple


class BindingCTW:
    """
    N-gram cache with binding-energy-modulated Dirichlet CTW mixing.

    Compatible with PR #986's NgramCache interface but replaces
    fixed concentration with context-adaptive concentration.
    """

    PRIMES = [np.uint64(p) for p in [
        36313, 27191, 51647, 81929, 131071, 174763, 233017,
        299993, 350377, 412391, 479909, 541267, 613651, 700897, 786433
    ]]

    def __init__(self, max_order: int = 13, min_order: int = 2,
                 num_buckets: int = 131072, min_count: int = 2,
                 c_base: float = 5.0, beta: float = 2.0,
                 vocab_size: int = 1024):
        self.max_order = max_order
        self.min_order = min_order
        self.num_buckets = num_buckets
        self.min_count = min_count
        self.c_base = c_base
        self.beta = beta  # binding sensitivity
        self.vocab_size = vocab_size
        self.mask = np.uint64(num_buckets - 1)
        self.num_orders = max_order - min_order + 1

        # Count arrays (same structure as PR #986)
        self.ctx_counts = [np.zeros(num_buckets, dtype=np.uint32)
                           for _ in range(self.num_orders)]
        self.full_counts = [np.zeros(num_buckets, dtype=np.uint32)
                            for _ in range(self.num_orders)]

        # Token frequency for specificity (built during scan/warmup)
        self.token_freq = np.zeros(vocab_size, dtype=np.float64)
        self.total_tokens = 0

    # -----------------------------------------------------------------
    # Binding energy computation
    # -----------------------------------------------------------------

    def _specificity(self, token_id: int) -> float:
        """σ(t) = log(N/freq(t)) — IDF-like specificity."""
        freq = self.token_freq[token_id]
        if freq <= 0 or self.total_tokens <= 0:
            return 0.0
        return math.log(self.total_tokens / freq)

    def binding_energy(self, context_tokens: np.ndarray) -> float:
        """
        B(ctx) for a sequence of context tokens.
        Combines pairwise specificity and sequential coherence.

        B = (1/n) × Σ σ(t_i) × (1 + adjacency_bonus)

        High B = rare, specific tokens in coherent sequence.
        Low B = common tokens or incoherent mix.
        """
        n = len(context_tokens)
        if n == 0:
            return 0.0

        # Average specificity
        specs = np.array([self._specificity(int(t)) for t in context_tokens])
        avg_spec = specs.mean()

        if n < 2:
            return avg_spec

        # Pairwise specificity product (geometric mean of adjacent pairs)
        pair_products = []
        for i in range(n - 1):
            s1 = self._specificity(int(context_tokens[i]))
            s2 = self._specificity(int(context_tokens[i + 1]))
            pair_products.append(s1 * s2)

        if pair_products:
            pair_score = np.mean(pair_products)
        else:
            pair_score = 0.0

        # Combine: average specificity × pairwise coherence
        return avg_spec * (1.0 + pair_score)

    def binding_energy_batch(self, val_np: np.ndarray, positions: np.ndarray,
                              context_len: int) -> np.ndarray:
        """
        Compute binding energy for a batch of positions.

        Args:
            val_np: full token array
            positions: (N,) array of positions to score
            context_len: how many preceding tokens to use as context

        Returns:
            binding: (N,) array of binding energies
        """
        n = len(positions)
        binding = np.zeros(n, dtype=np.float64)

        # Precompute IDF for all tokens
        if self.total_tokens <= 0:
            return binding

        # Vectorized IDF lookup
        log_N = math.log(max(self.total_tokens, 1))
        idf = np.zeros(self.vocab_size, dtype=np.float64)
        nonzero = self.token_freq > 0
        idf[nonzero] = log_N - np.log(self.token_freq[nonzero])

        for i in range(n):
            pos = positions[i]
            ctx_start = max(0, pos - context_len)
            ctx = val_np[ctx_start:pos + 1]
            if len(ctx) == 0:
                continue

            # Clamp token ids to vocab range
            ctx_ids = np.clip(ctx.astype(np.int64), 0, self.vocab_size - 1)
            specs = idf[ctx_ids]
            avg_spec = specs.mean()

            if len(ctx) >= 2:
                pair_prods = specs[:-1] * specs[1:]
                pair_score = pair_prods.mean()
                binding[i] = avg_spec * (1.0 + pair_score)
            else:
                binding[i] = avg_spec

        return binding

    def concentration_for_binding(self, binding: np.ndarray) -> np.ndarray:
        """
        Map binding energy to Dirichlet concentration.

        In the Dirichlet CTW formula p = (c × p_prev + count) / (c + ctx_count):
          - HIGH c → trust the prior/backup more (smooth)
          - LOW c → trust the observed counts more (sharp)

        So the mapping is INVERSE:
          - High binding (rare, specific) → LOW c → trust counts (they're reliable)
          - Low binding (common, ambiguous) → HIGH c → smooth (counts are noisy)

        c(B) = c_base × (1 + β × (1 - sigmoid(B - median_B)))
             = c_base × (1 + β × sigmoid(median_B - B))
        """
        median_b = np.median(binding[binding > 0]) if np.any(binding > 0) else 1.0
        # INVERSE sigmoid: high binding → low value → low concentration
        inv_normalized = 1.0 / (1.0 + np.exp(-(median_b - binding)))
        return self.c_base * (1.0 + self.beta * inv_normalized)

    # -----------------------------------------------------------------
    # Cache operations (compatible with PR #986 NgramCache)
    # -----------------------------------------------------------------

    def build_full(self, val_np: np.ndarray, log_fn=None):
        """Build complete cache from all tokens (for two-pass rescoring)."""
        n = len(val_np) - 1
        mask = self.mask
        primes = self.PRIMES

        # Also build token frequencies
        counts = np.bincount(val_np.astype(np.int32), minlength=self.vocab_size)
        self.token_freq[:min(len(counts), self.vocab_size)] += counts[:self.vocab_size]
        self.total_tokens += len(val_np)

        for oi in range(self.num_orders):
            order = self.min_order + oi
            cw = order - 1
            if n <= cw:
                continue
            valid_start = cw
            n_pos = n - valid_start

            ctx_hash = np.zeros(n_pos, dtype=np.uint64)
            for k in range(cw):
                t = val_np[valid_start - cw + k:valid_start - cw + k + n_pos].astype(np.uint64)
                ctx_hash ^= t * np.uint64(primes[k])
            ctx_key = (ctx_hash & mask).astype(np.int64)

            targets = val_np[valid_start + 1:valid_start + 1 + n_pos].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)

            np.add.at(self.ctx_counts[oi], ctx_key, 1)
            np.add.at(self.full_counts[oi], full_key, 1)

            if log_fn:
                log_fn(f"binding_ctw: order {order} built, {n_pos} positions")

    def warm_from_training(self, token_freq: np.ndarray, total_tokens: int):
        """Warm up token frequencies from training data scan."""
        self.token_freq[:len(token_freq)] += token_freq[:self.vocab_size]
        self.total_tokens += total_tokens

    def update(self, val_np: np.ndarray, start: int, end: int):
        """Update cache with tokens from [start, end)."""
        seg_len = end - start
        mask = self.mask
        primes = self.PRIMES

        for oi in range(self.num_orders):
            order = self.min_order + oi
            cw = order - 1
            first_valid = max(cw, start) - start
            n_pos = seg_len - first_valid
            if n_pos <= 0:
                continue
            abs_s = start + first_valid

            ctx_hash = np.zeros(n_pos, dtype=np.uint64)
            for k in range(cw):
                t = val_np[abs_s - cw + k:abs_s - cw + k + n_pos].astype(np.uint64)
                ctx_hash ^= t * np.uint64(primes[k])
            ctx_key = (ctx_hash & mask).astype(np.int64)

            targets = val_np[abs_s + 1:abs_s + 1 + n_pos].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)

            np.add.at(self.ctx_counts[oi], ctx_key, 1)
            np.add.at(self.full_counts[oi], full_key, 1)

    # -----------------------------------------------------------------
    # The key method: binding-modulated hierarchical Dirichlet
    # -----------------------------------------------------------------

    def lookup_hierarchical_binding(
        self, val_np: np.ndarray, start: int, end: int,
        base_p: np.ndarray,
        context_len: int = 8,
    ) -> np.ndarray:
        """
        Hierarchical Dirichlet CTW mixing with evidence-aware concentration.

        The key insight: concentration should be LOWER when n-gram evidence
        is strong (high ctx_count at high orders) and HIGHER when evidence
        is weak. This is the self-model: the compression knows when to
        trust itself.

        For each order, concentration adapts based on:
          c_eff = c_base / (1 + β × log1p(ctx_count) × specificity_boost)

        where specificity_boost = avg IDF of context tokens.
        High counts + rare context → very low c → trust counts fully.
        Low counts + common context → c ≈ c_base → smooth toward backup.

        Args:
            val_np: full token array
            start, end: position range to score
            base_p: (seg_len,) base neural model probabilities
            context_len: context window for binding computation

        Returns:
            blended: (seg_len,) final blended probabilities
        """
        seg_len = end - start
        blended = base_p.copy()
        mask = self.mask
        primes = self.PRIMES

        # Precompute IDF for specificity boost
        if self.total_tokens > 0:
            log_N = math.log(max(self.total_tokens, 1))
            idf = np.zeros(self.vocab_size, dtype=np.float64)
            nonzero = self.token_freq > 0
            idf[nonzero] = log_N - np.log(self.token_freq[nonzero])
            max_idf = idf.max() if idf.max() > 0 else 1.0
            idf_norm = idf / max_idf  # normalize to [0, 1]
        else:
            idf_norm = np.ones(self.vocab_size, dtype=np.float64)

        # Iterate lowest to highest order
        for oi in range(self.num_orders):
            order = self.min_order + oi
            cw = order - 1
            first_valid = max(cw, start) - start
            n_pos = seg_len - first_valid
            if n_pos <= 0:
                continue
            abs_s = start + first_valid

            ctx_hash = np.zeros(n_pos, dtype=np.uint64)
            for k in range(cw):
                t = val_np[abs_s - cw + k:abs_s - cw + k + n_pos].astype(np.uint64)
                ctx_hash ^= t * np.uint64(primes[k])
            ctx_key = (ctx_hash & mask).astype(np.int64)

            targets = val_np[abs_s + 1:abs_s + 1 + n_pos].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)

            ctx_c = self.ctx_counts[oi][ctx_key]
            full_c = np.minimum(self.full_counts[oi][full_key], ctx_c)
            valid = (ctx_c >= self.min_count) & (full_c > 0)

            if valid.any():
                idx = np.nonzero(valid)[0]
                fc = full_c[idx].astype(np.float64)
                cc = ctx_c[idx].astype(np.float64)
                prev_p = blended[first_valid + idx]

                # Compute specificity boost from context tokens
                spec_boost = np.ones(len(idx), dtype=np.float64)
                for k in range(min(cw, context_len)):
                    ctx_tok = val_np[abs_s + idx - cw + k].astype(np.int64)
                    ctx_tok = np.clip(ctx_tok, 0, self.vocab_size - 1)
                    spec_boost += idf_norm[ctx_tok]
                spec_boost /= (min(cw, context_len) + 1)  # normalize

                # Evidence-aware concentration:
                # More evidence + rare context → lower c → trust counts
                c_eff = self.c_base / (1.0 + self.beta * np.log1p(cc) * spec_boost)
                c_eff = np.clip(c_eff, 0.1, self.c_base * 5)

                blended[first_valid + idx] = (c_eff * prev_p + fc) / (c_eff + cc)

        return blended

    def lookup_hierarchical_fixed(
        self, val_np: np.ndarray, start: int, end: int,
        base_p: np.ndarray, concentration: float = 5.0,
    ) -> np.ndarray:
        """Standard fixed-concentration hierarchical Dirichlet (for comparison)."""
        seg_len = end - start
        blended = base_p.copy()
        mask = self.mask
        primes = self.PRIMES

        for oi in range(self.num_orders):
            order = self.min_order + oi
            cw = order - 1
            first_valid = max(cw, start) - start
            n_pos = seg_len - first_valid
            if n_pos <= 0:
                continue
            abs_s = start + first_valid

            ctx_hash = np.zeros(n_pos, dtype=np.uint64)
            for k in range(cw):
                t = val_np[abs_s - cw + k:abs_s - cw + k + n_pos].astype(np.uint64)
                ctx_hash ^= t * np.uint64(primes[k])
            ctx_key = (ctx_hash & mask).astype(np.int64)

            targets = val_np[abs_s + 1:abs_s + 1 + n_pos].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)

            ctx_c = self.ctx_counts[oi][ctx_key]
            full_c = np.minimum(self.full_counts[oi][full_key], ctx_c)
            valid = (ctx_c >= self.min_count) & (full_c > 0)

            if valid.any():
                idx = np.nonzero(valid)[0]
                fc = full_c[idx].astype(np.float64)
                cc = ctx_c[idx].astype(np.float64)
                prev_p = blended[first_valid + idx]
                blended[first_valid + idx] = (concentration * prev_p + fc) / (concentration + cc)

        return blended

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def stats(self) -> dict:
        total_ctx = sum(int(c.sum()) for c in self.ctx_counts)
        total_full = sum(int(c.sum()) for c in self.full_counts)
        return {
            'max_order': self.max_order,
            'min_order': self.min_order,
            'num_buckets': self.num_buckets,
            'total_ctx_entries': total_ctx,
            'total_full_entries': total_full,
            'token_freq_nonzero': int(np.sum(self.token_freq > 0)),
            'total_tokens': self.total_tokens,
            'c_base': self.c_base,
            'beta': self.beta,
        }
