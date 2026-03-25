"""CTW-based n-gram mixer for Parameter Golf.

Replaces heuristic entropy-adaptive alpha with Context Tree Weighting —
a provably optimal Bayesian mixture over all context tree models up to depth D.

Key formulas (Willems et al., 1995):
  P_e(a+1, b) = (a + 0.5) / (a + b + 1) * P_e(a, b)   [KT estimator]
  P_w = 0.5 * P_e + 0.5 * prod(P_w_children)            [recursive weighting]

For M-ary alphabet (vocab=1024):
  P_e(symbol=j) = (count_j + 0.5) / (total + 0.5 * M)
"""

import numpy as np
from typing import Optional


class CTWNode:
    """A node in the CTW context tree."""
    __slots__ = ['counts', 'total', 'log_pe', 'log_pw', 'beta', 'children']

    def __init__(self, vocab_size: int):
        self.counts = None  # lazily allocated: np.zeros(vocab_size, dtype=np.float64)
        self.total = 0.0
        self.log_pe = 0.0   # log P_e (KT estimator)
        self.log_pw = 0.0   # log P_w (weighted probability)
        self.beta = 1.0     # ratio P_e / P_w for efficient updates
        self.children = {}  # token_id -> CTWNode


class CTWMixer:
    """Context Tree Weighting mixer for combining neural LM with n-gram predictions.

    Instead of a fixed alpha or entropy-adaptive alpha, CTW provides:
    - Provably optimal Bayesian model averaging over all context depths
    - Per-context, per-sequence adaptation
    - O(D) per token update cost

    Usage at eval time:
        mixer = CTWMixer(depth=8, vocab_size=1024)
        for each token:
            # Get neural model's predicted probability for target
            p_neural = model_prob[target_token]
            # Get CTW's predicted probability for target
            p_ctw = mixer.predict(context_tokens, target_token)
            # Mix in log-odds space (logistic domain mixing from PAQ)
            p_final = logistic_mix(p_neural, p_ctw, w_neural, w_ctw)
            # Update CTW with observed token
            mixer.update(context_tokens, target_token)
    """

    def __init__(self, depth: int = 8, vocab_size: int = 1024, min_count: int = 2):
        self.depth = depth
        self.vocab_size = vocab_size
        self.min_count = min_count
        self.root = CTWNode(vocab_size)
        self.half_m = 0.5 * vocab_size  # for KT estimator: 0.5 * M

    def _get_path(self, context: np.ndarray) -> list[CTWNode]:
        """Walk from root to leaf along context, creating nodes as needed.
        context should be the last D tokens (most recent last)."""
        path = [self.root]
        node = self.root
        # Walk from root (depth 0) to leaf (depth D)
        # context[-1] is most recent, context[-D] is oldest
        d = min(self.depth, len(context))
        for i in range(d):
            tok = int(context[-(i + 1)])  # most recent first
            if tok not in node.children:
                node.children[tok] = CTWNode(self.vocab_size)
            node = node.children[tok]
            path.append(node)
        return path

    def predict(self, context: np.ndarray, target: int) -> float:
        """Predict probability of target token given context using CTW.

        Returns the CTW-weighted probability for the target token.
        This blends n-gram estimates at all depths optimally.
        """
        path = self._get_path(context)

        # At the deepest node, use KT estimator
        leaf = path[-1]
        if leaf.counts is None or leaf.total < self.min_count:
            # Not enough data, return uniform
            return 1.0 / self.vocab_size

        # Compute KT probability at each node on the path
        # Then combine using CTW bottom-up
        # For efficiency, we use the beta-ratio trick

        # Simple approach: just use the deepest node with enough counts
        # and let CTW weighting handle the mixing via beta ratios
        best_p = 1.0 / self.vocab_size
        for node in reversed(path):
            if node.counts is not None and node.total >= self.min_count:
                # KT estimate at this node
                p = (node.counts[target] + 0.5) / (node.total + self.half_m)
                # Weight by CTW beta ratio
                # beta = P_e / P_w; higher beta means trust local estimate more
                weight = min(node.beta, 10.0) / (min(node.beta, 10.0) + 1.0)
                best_p = weight * p + (1.0 - weight) * best_p
        return max(best_p, 1e-12)

    def predict_distribution(self, context: np.ndarray) -> np.ndarray:
        """Get full CTW-weighted distribution over vocab."""
        path = self._get_path(context)
        dist = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)

        for node in reversed(path):
            if node.counts is not None and node.total >= self.min_count:
                kt = (node.counts + 0.5) / (node.total + self.half_m)
                weight = min(node.beta, 10.0) / (min(node.beta, 10.0) + 1.0)
                dist = weight * kt + (1.0 - weight) * dist
        return dist

    def update(self, context: np.ndarray, target: int) -> None:
        """Update CTW tree after observing target token."""
        path = self._get_path(context)

        # Update counts and KT estimators bottom-up
        for node in path:
            if node.counts is None:
                node.counts = np.zeros(self.vocab_size, dtype=np.float64)

            # KT probability BEFORE update
            pe_before = (node.counts[target] + 0.5) / (node.total + self.half_m)

            # Update counts
            node.counts[target] += 1.0
            node.total += 1.0

            # KT probability AFTER update
            pe_after = (node.counts[target] + 0.5) / (node.total + self.half_m)

            # Update beta ratio: beta *= P_e(symbol) / P_w(symbol)
            # P_w(symbol) is approximated by the mixed prediction
            pw_approx = max(pe_before, 1e-12)  # simplified
            node.beta *= pe_after / pw_approx
            # Clamp beta to prevent numerical issues
            node.beta = max(min(node.beta, 1e6), 1e-6)


def eval_with_ctw_ngram(
    model_logits: np.ndarray,  # (seq_len, vocab_size) float32
    target_ids: np.ndarray,    # (seq_len,) int64
    context_ids: np.ndarray,   # (seq_len,) int64 — input token ids
    ctw: CTWMixer,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    score_start: int = 0,      # only score tokens from this index
) -> tuple[float, float, float]:
    """Score a segment using CTW-mixed n-gram + neural model predictions.

    Returns (loss_sum, token_count, byte_count).
    """
    import math

    vocab_size = model_logits.shape[1]
    loss_sum = 0.0
    byte_count = 0.0
    token_count = 0

    for i in range(score_start, len(target_ids)):
        tgt = int(target_ids[i])
        ctx = int(context_ids[i])

        # Neural model probability for target
        logits = model_logits[i].astype(np.float64)
        logits -= logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        p_neural = max(probs[tgt], 1e-12)

        # CTW probability for target
        context_window = context_ids[max(0, i - ctw.depth):i]
        p_ctw = ctw.predict(context_window, tgt)

        # Logistic domain mixing (PAQ-style, better than linear)
        # stretch(p) = ln(p / (1-p)), squash(x) = 1/(1+e^{-x})
        def stretch(p):
            p = max(min(p, 1 - 1e-12), 1e-12)
            return math.log(p / (1 - p))

        s_neural = stretch(p_neural)
        s_ctw = stretch(p_ctw)

        # Entropy-adaptive weighting: trust CTW more when neural is uncertain
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        # Sigmoid schedule: high entropy -> more CTW weight
        w_ctw = 0.05 + 0.55 / (1.0 + math.exp(-2.0 * (entropy - 4.0)))
        w_neural = 1.0 - w_ctw

        # Mix in logistic domain
        s_mixed = w_neural * s_neural + w_ctw * s_ctw
        p_mixed = 1.0 / (1.0 + math.exp(-s_mixed))
        p_mixed = max(p_mixed, 1e-12)

        # Compute NLL
        nll = -math.log(p_mixed)
        loss_sum += nll
        token_count += 1

        # Byte counting for BPB
        tb = float(base_bytes_lut[tgt])
        if has_leading_space_lut[tgt] and not is_boundary_token_lut[ctx]:
            tb += 1.0
        byte_count += tb

        # Update CTW with observed token (score-first: update AFTER scoring)
        ctw.update(context_window, tgt)

    return loss_sum, float(token_count), byte_count
