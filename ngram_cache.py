"""
Order-8 n-gram cache with Modified Kneser-Ney smoothing.

Backward-looking: builds from already-scored tokens only.
Upgraded from order-5 with simple interpolation to order-8 with
Modified Kneser-Ney (Chen & Goodman, 1999) for better smoothing
of sparse high-order contexts.

Reference: Chen & Goodman (1999), "An Empirical Study of Smoothing Techniques
for Language Modeling", Computer Speech and Language 13(4).
"""
import torch
import math
from collections import defaultdict
from typing import Optional, Tuple


class NgramCache:
    """Backward-looking n-gram cache with Modified Kneser-Ney smoothing."""

    def __init__(self, max_order: int = 8, vocab_size: int = 1024):
        self.max_order = max_order
        self.vocab_size = vocab_size

        # counts[order][context_tuple][token] = count
        self.counts: list[dict] = [defaultdict(lambda: defaultdict(int)) for _ in range(max_order + 1)]
        # totals[order][context_tuple] = total count
        self.totals: list[dict] = [defaultdict(int) for _ in range(max_order + 1)]

        # For Kneser-Ney: continuation counts
        # n1[order][context] = number of word types with count == 1
        # n2[order][context] = number of word types with count == 2
        # n3plus[order][context] = number of word types with count >= 3
        self.n1: list[dict] = [defaultdict(int) for _ in range(max_order + 1)]
        self.n2: list[dict] = [defaultdict(int) for _ in range(max_order + 1)]
        self.n3plus: list[dict] = [defaultdict(int) for _ in range(max_order + 1)]

        # Continuation counts for lower-order KN distributions
        # For order > 0: how many distinct contexts precede token w
        # continuation_count[token] = number of distinct bigram types (*, token)
        self.continuation_count: dict[int, int] = defaultdict(int)
        self.total_continuations: int = 0
        # Track unique (context, token) pairs for continuation counting
        self._seen_bigrams: set = set()

        # Discount parameters (Modified KN uses 3 discounts: D1, D2, D3+)
        self._d1: float = 0.5
        self._d2: float = 0.75
        self._d3: float = 0.9
        self._discount_update_interval: int = 500
        self._total_observed: int = 0

    def _update_count_stats(self, order: int, ctx: tuple, token: int, old_count: int, new_count: int) -> None:
        """Update n1/n2/n3plus when a count changes."""
        if old_count == 1:
            self.n1[order][ctx] -= 1
        elif old_count == 2:
            self.n2[order][ctx] -= 1
        elif old_count >= 3:
            self.n3plus[order][ctx] -= 1

        if new_count == 1:
            self.n1[order][ctx] += 1
        elif new_count == 2:
            self.n2[order][ctx] += 1
        elif new_count >= 3:
            self.n3plus[order][ctx] += 1

    def _update_discounts(self) -> None:
        """Recompute discount parameters from count-of-counts (Chen & Goodman formula)."""
        # Collect global count-of-counts across all orders
        total_n1 = 0
        total_n2 = 0
        total_n3 = 0
        total_n4 = 0

        for order in range(self.max_order + 1):
            for ctx in self.counts[order]:
                for tok, c in self.counts[order][ctx].items():
                    if c == 1:
                        total_n1 += 1
                    elif c == 2:
                        total_n2 += 1
                    elif c == 3:
                        total_n3 += 1
                    elif c == 4:
                        total_n4 += 1

        if total_n1 == 0 or total_n2 == 0:
            return

        # Modified KN discount formula
        Y = total_n1 / (total_n1 + 2.0 * total_n2)
        self._d1 = 1.0 - 2.0 * Y * (total_n2 / max(total_n1, 1))
        self._d2 = 2.0 - 3.0 * Y * (total_n3 / max(total_n2, 1))
        self._d3 = 3.0 - 4.0 * Y * (total_n4 / max(total_n3, 1))

        # Clamp to valid range
        self._d1 = max(0.01, min(0.99, self._d1))
        self._d2 = max(0.01, min(0.99, self._d2))
        self._d3 = max(0.01, min(0.99, self._d3))

    def observe(self, token_ids: list[int]) -> None:
        """Feed already-scored tokens into the cache."""
        for i in range(len(token_ids)):
            for order in range(min(i, self.max_order) + 1):
                ctx = tuple(token_ids[i - order:i]) if order > 0 else ()
                old_count = self.counts[order][ctx].get(token_ids[i], 0)
                self.counts[order][ctx][token_ids[i]] = old_count + 1
                self.totals[order][ctx] += 1
                self._update_count_stats(order, ctx, token_ids[i], old_count, old_count + 1)

            # Track continuation counts (for KN lower-order)
            if i > 0:
                bigram = (token_ids[i - 1], token_ids[i])
                if bigram not in self._seen_bigrams:
                    self._seen_bigrams.add(bigram)
                    self.continuation_count[token_ids[i]] += 1
                    self.total_continuations += 1

        self._total_observed += len(token_ids)
        if self._total_observed % self._discount_update_interval == 0:
            self._update_discounts()

    def _discount(self, count: int) -> float:
        """Get the appropriate discount for a given count."""
        if count == 1:
            return self._d1
        elif count == 2:
            return self._d2
        else:
            return self._d3

    def _kn_prob_lowest(self, token: int) -> float:
        """Lowest-order KN probability: based on continuation counts."""
        if self.total_continuations == 0:
            return 1.0 / self.vocab_size
        cc = self.continuation_count.get(token, 0)
        if cc == 0:
            return 1.0 / self.vocab_size
        return cc / self.total_continuations

    def predict(self, context: list[int]) -> Tuple[Optional[torch.Tensor], float]:
        """Predict next token using Modified Kneser-Ney smoothing.

        Returns (probability_distribution, confidence) or (None, 0.0).
        """
        max_ctx = min(len(context), self.max_order)

        # Check if we have any data
        total_obs = sum(self.totals[0].values())
        if total_obs < 3:
            return None, 0.0

        probs = torch.zeros(self.vocab_size, dtype=torch.float32)

        # Compute KN probabilities for each token in vocab that we've seen
        all_tokens = set()
        for order in range(max_ctx + 1):
            ctx = tuple(context[-order:]) if order > 0 else ()
            if ctx in self.counts[order]:
                all_tokens.update(self.counts[order][ctx].keys())

        # Add tokens from continuation counts
        all_tokens.update(self.continuation_count.keys())

        for token in all_tokens:
            if token >= self.vocab_size:
                continue
            probs[token] = self._kn_recursive(context, max_ctx, token)

        # Assign minimum prob to unseen tokens
        unseen_mask = probs == 0
        if unseen_mask.any():
            min_prob = 1.0 / (self.vocab_size * 10)
            probs[unseen_mask] = min_prob

        total = probs.sum()
        if total < 1e-10:
            return None, 0.0

        probs = probs / total

        # Confidence based on amount of data observed
        conf = min(0.6, total_obs / (total_obs + 200))
        return probs, conf

    def _kn_recursive(self, context: list[int], order: int, token: int) -> float:
        """Recursively compute Modified Kneser-Ney probability."""
        if order == 0:
            # Unigram with continuation counts
            ctx = ()
            total = self.totals[0].get(ctx, 0)
            if total == 0:
                return self._kn_prob_lowest(token)
            count = self.counts[0][ctx].get(token, 0)
            if count == 0:
                return self._kn_prob_lowest(token)
            return count / total

        ctx = tuple(context[-order:])
        total = self.totals[order].get(ctx, 0)

        if total == 0:
            # Back off to lower order
            return self._kn_recursive(context, order - 1, token)

        count = self.counts[order][ctx].get(token, 0)

        # Discounted probability
        if count > 0:
            d = self._discount(count)
            p_discount = max(0, count - d) / total
        else:
            p_discount = 0.0

        # Interpolation weight (backoff weight)
        n1_ctx = self.n1[order].get(ctx, 0)
        n2_ctx = self.n2[order].get(ctx, 0)
        n3p_ctx = self.n3plus[order].get(ctx, 0)
        gamma = (self._d1 * n1_ctx + self._d2 * n2_ctx + self._d3 * n3p_ctx) / total

        # Recursive lower-order probability
        p_lower = self._kn_recursive(context, order - 1, token)

        return p_discount + gamma * p_lower

    def entropy_alpha(self, neural_logits: torch.Tensor) -> float:
        """Compute entropy-adaptive mixing weight from neural model uncertainty.

        Higher neural entropy -> trust cache more.
        """
        p = torch.softmax(neural_logits.float(), dim=-1)
        ent = -(p * torch.log(p + 1e-10)).sum()
        norm_ent = (ent / math.log(self.vocab_size)).item()
        return min(0.5, max(0.05, 0.05 + 0.45 * norm_ent))

    def reset(self) -> None:
        """Clear all state."""
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(self.max_order + 1)]
        self.totals = [defaultdict(int) for _ in range(self.max_order + 1)]
        self.n1 = [defaultdict(int) for _ in range(self.max_order + 1)]
        self.n2 = [defaultdict(int) for _ in range(self.max_order + 1)]
        self.n3plus = [defaultdict(int) for _ in range(self.max_order + 1)]
        self.continuation_count = defaultdict(int)
        self.total_continuations = 0
        self._seen_bigrams = set()
        self._total_observed = 0
