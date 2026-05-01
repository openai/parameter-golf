"""
Causal N-gram Cache — eval-time additive logit contribution.

LEGALITY (per Issue #1017 Four Conditions + valerio-oai rulings #993, #1185, #959):

1. EXACT non-hashed counting  (counters a Python dict of dict; NO hash buckets).
   valerio-oai closed #993 for "hashed n-gram models in this way are disallowed".

2. FULL-VOCAB LOG-PROB tensor over Sigma is produced and added to neural logits
   BEFORE softmax, so the blend is an additive-logit shift and the final softmax
   is a valid normalized distribution over Sigma, independent of x_t.
   valerio-oai closed #1185 for computing the blend only for the target token.

3. UPDATE-AFTER-SCORE discipline: counts are frozen at the start of a scoring
   region. Only after all windows in the region are scored may counts be updated
   with tokens that were just scored. No token influences its own probability.

4. SINGLE left-to-right pass: the scoring region is processed once, no rescoring.

5. Alpha is a fixed scalar baked into the artifact. No x_t-dependent mixing.

DATA STRUCTURE:
    counts[k] is a dict mapping context tuple (length k-1) -> Counter of token ids.
    counts[1] is the unigram Counter (context = empty tuple).
    We store order 1..K. Backoff walks K, K-1, ..., 1.

SCORING:
    For a context c of max length K-1 at position t:
      - Walk from order K down: if c[-(k-1):] in counts[k], return the smoothed
        log-prob vector for that context.
      - Else back off.
      - Order 1 (unigram) always has a defined distribution (uniform prior smoothing).

    Smoothing is add-delta (delta=0.5) applied within each order's lookup — no
    cross-order mixing, so the distribution is well-defined and normalized.
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import List, Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None


class CausalNGram:
    """Exact non-hashed causal n-gram with backoff. See module docstring."""

    def __init__(self, vocab_size: int, order: int = 4, delta: float = 0.5,
                 min_context_count: int = 2):
        """
        Args:
            vocab_size: size of Sigma (token alphabet).
            order: max n-gram order (K). Backoff goes K -> K-1 -> ... -> 1.
            delta: add-delta smoothing parameter.
            min_context_count: minimum total observations of a context before we
                trust it (else back off to shorter order). Helps avoid the
                degenerate order-82 failure mode of closed PRs.
        """
        assert order >= 1
        assert vocab_size > 0
        self.V = vocab_size
        self.K = order
        self.delta = delta
        self.min_ctx = min_context_count

        # counts[k] maps context tuple of length k-1 -> Counter of next tokens.
        # counts[1] uses the empty tuple () as its only key.
        self.counts = {k: defaultdict(Counter) for k in range(1, order + 1)}

        # Totals per context (for normalization without re-summing the counter).
        self.totals = {k: defaultdict(int) for k in range(1, order + 1)}

        # Frozen snapshot (for update-after-score):
        #   After call to `freeze()`, lookups use the snapshot; subsequent `add()`
        #   calls update the live counts only. `thaw()` re-points lookups to live.
        self._frozen_counts = None
        self._frozen_totals = None

        # Cached log-prob vectors, invalidated when a new snapshot is taken.
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def add_token(self, history: List[int], token: int) -> None:
        """Accumulate one (history, token) observation into LIVE counts.

        history[-(k-1):] is used as the context for order k. Updates unigram
        through order K in one shot.
        """
        assert 0 <= token < self.V
        for k in range(1, self.K + 1):
            # context is the last (k-1) tokens of history
            ctx_len = k - 1
            if ctx_len == 0:
                ctx = ()
            else:
                if len(history) < ctx_len:
                    continue  # not enough history for this order
                ctx = tuple(history[-ctx_len:])
            self.counts[k][ctx][token] += 1
            self.totals[k][ctx] += 1

    def add_sequence(self, tokens: List[int]) -> None:
        """Add a whole sequence. Equivalent to `add_token` called left-to-right."""
        for i, tok in enumerate(tokens):
            self.add_token(tokens[:i], tok)

    def freeze(self) -> None:
        """Snapshot current counts. Subsequent lookups use this snapshot.

        This is how we implement update-after-score: freeze before scoring,
        then `add_token`/`add_sequence` to the live counts during/after scoring,
        then `thaw()` to swap.
        """
        # Deep copy is O(N) — fine for bounded cache sizes. Python dict copy
        # is shallow but Counter copy via Counter(c) re-allocates.
        self._frozen_counts = {k: {ctx: Counter(c) for ctx, c in d.items()}
                                for k, d in self.counts.items()}
        self._frozen_totals = {k: dict(d) for k, d in self.totals.items()}
        self._cache.clear()

    def thaw(self) -> None:
        """Swap live counts into the "scoring" slot. Used at chunk boundary.

        Policy: at the end of a scoring region, the accumulated updates become
        the new frozen snapshot for the NEXT region. Equivalent to calling
        freeze() again but on the LIVE counts.
        """
        self.freeze()

    # ------------------------------------------------------------------
    # Lookup (reads frozen snapshot, not live)
    # ------------------------------------------------------------------

    def _get_frozen(self, k: int, ctx: tuple):
        if self._frozen_counts is None:
            src = self.counts
            tot = self.totals
        else:
            src = self._frozen_counts
            tot = self._frozen_totals
        return src[k].get(ctx, None), tot[k].get(ctx, 0)

    def log_probs(self, history: List[int]) -> np.ndarray:
        """Return log_p(v | history) for all v in Sigma. Length = V.

        Walks backoff from order K down. First order where the context has
        at least `min_ctx` observations is used. Unigram always available (we
        fall through to a uniform if even unigram has no mass, which shouldn't
        happen after any real data).

        Output is a FULL normalized log-distribution: exp(log_probs).sum() == 1.
        """
        cache_key = tuple(history[-(self.K - 1):]) if self.K > 1 else ()
        if cache_key in self._cache:
            return self._cache[cache_key]

        log_p = None
        for k in range(self.K, 0, -1):
            ctx_len = k - 1
            if ctx_len == 0:
                ctx = ()
            elif len(history) < ctx_len:
                continue
            else:
                ctx = tuple(history[-ctx_len:])
            counter, total = self._get_frozen(k, ctx)
            if total >= self.min_ctx:
                # Add-delta smoothing on full vocab
                denom = total + self.delta * self.V
                vec = np.full(self.V, self.delta / denom, dtype=np.float64)
                if counter is not None:
                    for tok, c in counter.items():
                        vec[tok] = (c + self.delta) / denom
                log_p = np.log(vec)
                break
        if log_p is None:
            # Uniform fallback (e.g., empty cache)
            log_p = np.full(self.V, -math.log(self.V), dtype=np.float64)

        self._cache[cache_key] = log_p
        return log_p

    # ------------------------------------------------------------------
    # Batch API for the eval loop
    # ------------------------------------------------------------------

    def batch_log_probs(self, context_tensor, device=None):
        """Given a (B, T) tensor of token ids where position t in each row is the
        context for predicting position t+1, return a (B, T, V) tensor of
        log-probs. Only the FROZEN snapshot is used.

        Implementation: O(B*T) Python loop over positions. Acceptable for
        prototype/small-model runs. For the 8xH100 competition eval we'll
        need to port this to a GPU kernel (or at least cache per-context).
        """
        assert torch is not None, "torch required for batch_log_probs"
        B, T = context_tensor.shape
        out = torch.empty((B, T, self.V), dtype=torch.float32,
                          device=device or context_tensor.device)
        ctx_cpu = context_tensor.detach().cpu().tolist()
        for b in range(B):
            row = ctx_cpu[b]
            for t in range(T):
                # history = row[:t+1]  (tokens 0..t inclusive become context for t+1)
                hist = row[max(0, t + 1 - (self.K - 1)):t + 1]
                lp = self.log_probs(hist)
                out[b, t] = torch.from_numpy(lp).to(out.dtype)
        return out

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def size_bytes(self) -> int:
        """Rough estimate of Python memory used by count tables."""
        total = 0
        for k in range(1, self.K + 1):
            total += sum(len(c) * 32 for c in self.counts[k].values())  # ~32B per entry
            total += len(self.counts[k]) * 80  # dict overhead
        return total

    def unique_contexts(self) -> dict:
        return {k: len(self.counts[k]) for k in range(1, self.K + 1)}
