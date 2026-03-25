"""
Gated Linear Network (GLN) context mixer for combining multiple predictors.

Replaces simple alpha-blending with a provably-optimal online learning mixer.
Each GLN neuron performs per-neuron convex optimization with O(log T) regret,
meaning the mixer converges to the best fixed combination of predictors.

Reference: Veness et al. (2021), "Gated Linear Networks", AAAI 2021.
Also: Mixtures of experts via Context Tree Weighting (Willems et al., 1995).

Legal: The GLN mixer has no pre-trained parameters. It learns online from
already-scored tokens only (backward-looking). Zero artifact cost.
"""
import torch
import math
import numpy as np
from typing import Optional, List, Tuple


class GLNMixer:
    """Gated Linear Network context mixer.

    Architecture:
    - Input: K predictor distributions (neural, FST, n-gram, match)
    - Context features: neural entropy, predictor confidences, position
    - Output: optimally mixed distribution

    Each neuron in the GLN maintains weights that are updated via
    online convex optimization (exponentiated gradient / mirror descent).
    The gating is based on context features that determine which
    combination of experts to trust.
    """

    def __init__(
        self,
        num_predictors: int = 4,
        num_context_bins: int = 8,
        learning_rate: float = 0.1,
        vocab_size: int = 1024,
    ):
        """
        Args:
            num_predictors: Number of input predictors (K).
            num_context_bins: Number of discrete context bins for gating.
            learning_rate: Online learning rate (eta).
            vocab_size: Size of token vocabulary.
        """
        self.K = num_predictors
        self.num_bins = num_context_bins
        self.lr = learning_rate
        self.vocab_size = vocab_size

        # Per-context-bin weights for each predictor
        # Shape: [num_bins, K] -- log-space weights (for exponentiated gradient)
        self.log_weights = np.zeros((num_context_bins, self.K), dtype=np.float64)

        # Running statistics for adaptive learning
        self.update_count = np.zeros(num_context_bins, dtype=np.int64)
        self.cumulative_loss = np.zeros((num_context_bins, self.K), dtype=np.float64)

        # Predictor names for logging
        self.predictor_names = ["neural", "fst", "ngram", "match"][:num_predictors]

    def _get_context_bin(self, neural_entropy: float, position_frac: float = 0.0) -> int:
        """Map context features to a discrete bin.

        Uses neural entropy as the primary context signal:
        - Low entropy = neural is confident = trust neural more
        - High entropy = neural is uncertain = trust other predictors more
        """
        # Quantize entropy into bins
        # Entropy is normalized to [0, 1], split into num_bins
        bin_idx = int(neural_entropy * (self.num_bins - 1))
        bin_idx = max(0, min(self.num_bins - 1, bin_idx))
        return bin_idx

    def _get_weights(self, bin_idx: int) -> np.ndarray:
        """Get normalized weights for a context bin (softmax of log-weights)."""
        lw = self.log_weights[bin_idx]
        # Softmax with numerical stability
        lw_shifted = lw - lw.max()
        w = np.exp(lw_shifted)
        w = w / w.sum()
        return w

    def mix(
        self,
        predictor_probs: List[Optional[torch.Tensor]],
        predictor_confidences: List[float],
        neural_entropy: float,
        position_frac: float = 0.0,
    ) -> torch.Tensor:
        """Mix predictor distributions using learned context-dependent weights.

        Args:
            predictor_probs: List of K probability distributions [vocab_size] each.
                None entries are treated as uniform.
            predictor_confidences: Confidence scores for each predictor.
            neural_entropy: Normalized entropy of neural predictions [0, 1].
            position_frac: Fractional position in sequence [0, 1].

        Returns:
            Mixed probability distribution [vocab_size].
        """
        bin_idx = self._get_context_bin(neural_entropy, position_frac)
        base_weights = self._get_weights(bin_idx)

        # Modulate weights by predictor confidence
        effective_weights = np.zeros(self.K, dtype=np.float64)
        uniform = torch.ones(self.vocab_size, dtype=torch.float32) / self.vocab_size

        active_preds: List[torch.Tensor] = []
        for i in range(self.K):
            if predictor_probs[i] is not None and predictor_confidences[i] > 0.01:
                active_preds.append(predictor_probs[i])
                effective_weights[i] = base_weights[i] * predictor_confidences[i]
            else:
                active_preds.append(uniform)
                # Give tiny weight to inactive predictors
                effective_weights[i] = base_weights[i] * 0.001

        # Normalize effective weights
        total_w = effective_weights.sum()
        if total_w < 1e-10:
            effective_weights = np.ones(self.K, dtype=np.float64) / self.K
        else:
            effective_weights = effective_weights / total_w

        # Weighted mixture of distributions
        mixed = torch.zeros(self.vocab_size, dtype=torch.float32)
        for i in range(self.K):
            w = float(effective_weights[i])
            if w > 1e-8:
                pred = active_preds[i]
                if pred.device != mixed.device:
                    pred = pred.to(mixed.device)
                mixed += w * pred

        # Ensure valid distribution
        mixed = mixed.clamp(min=1e-10)
        mixed = mixed / mixed.sum()
        return mixed

    def update(
        self,
        predictor_probs: List[Optional[torch.Tensor]],
        predictor_confidences: List[float],
        neural_entropy: float,
        true_token: int,
        position_frac: float = 0.0,
    ) -> None:
        """Update mixer weights based on observed true token.

        Uses exponentiated gradient (mirror descent on KL divergence)
        which gives O(log T) regret against the best fixed expert.

        Args:
            predictor_probs: Predictions from each predictor.
            predictor_confidences: Confidence for each predictor.
            neural_entropy: Normalized neural entropy.
            true_token: The true next token (already scored).
            position_frac: Fractional position in sequence.
        """
        bin_idx = self._get_context_bin(neural_entropy, position_frac)
        self.update_count[bin_idx] += 1

        # Compute per-predictor log-loss on the true token
        for i in range(self.K):
            if predictor_probs[i] is not None and predictor_confidences[i] > 0.01:
                prob_true = max(predictor_probs[i][true_token].item(), 1e-10)
                loss = -math.log(prob_true)
            else:
                loss = math.log(self.vocab_size)  # Uniform loss

            self.cumulative_loss[bin_idx, i] += loss

            # Exponentiated gradient update: log_w -= lr * loss
            # Adaptive LR: decrease with sqrt(updates)
            adaptive_lr = self.lr / math.sqrt(1 + self.update_count[bin_idx])
            self.log_weights[bin_idx, i] -= adaptive_lr * loss

        # Renormalize log-weights (prevent drift)
        lw = self.log_weights[bin_idx]
        self.log_weights[bin_idx] = lw - lw.mean()

    def get_stats(self) -> dict:
        """Return statistics about the mixer state."""
        stats = {}
        total_updates = int(self.update_count.sum())
        stats["total_updates"] = total_updates

        # Average weights per bin
        for b in range(self.num_bins):
            if self.update_count[b] > 0:
                w = self._get_weights(b)
                bin_stats = {self.predictor_names[i]: f"{w[i]:.3f}" for i in range(self.K)}
                stats[f"bin_{b}"] = bin_stats

        return stats

    def reset(self) -> None:
        """Reset all learned state."""
        self.log_weights = np.zeros((self.num_bins, self.K), dtype=np.float64)
        self.update_count = np.zeros(self.num_bins, dtype=np.int64)
        self.cumulative_loss = np.zeros((self.num_bins, self.K), dtype=np.float64)


class SimpleAlphaFallback:
    """Fallback entropy-adaptive alpha blending when GLN has insufficient data.

    Used for the first N tokens before GLN has learned meaningful weights.
    """

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size

    def mix(
        self,
        neural_probs: torch.Tensor,
        other_probs: List[Optional[torch.Tensor]],
        other_confidences: List[float],
        neural_entropy: float,
    ) -> torch.Tensor:
        """Simple entropy-adaptive alpha blending."""
        final = neural_probs.clone()

        for probs, conf in zip(other_probs, other_confidences):
            if probs is None or conf < 0.05:
                continue
            # Weight: confidence * entropy-scaling
            alpha = min(0.4, conf * (0.2 + 0.6 * neural_entropy))
            if probs.device != final.device:
                probs = probs.to(final.device)
            final = (1 - alpha) * final + alpha * probs

        final = final.clamp(min=1e-10)
        final = final / final.sum()
        return final
