"""Loss variant registry and implementations for causal screening.

Provides monkey-patchable loss functions that replace model.loss during
in-process training runs. Each variant is a factory that closes over the
model and returns a loss(input_ids, target_ids) callable.

Variants:
  - rho1: Selective loss masking (skip easy tokens by max-logit threshold)
  - adaptive_k: Multi-token prediction (predict N+2 on high-margin tokens)
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSS_VARIANTS: dict[str, callable] = {}  # name -> factory(model, config) -> loss_fn


def patch_model_loss(model, variant_name: str, config: dict | None = None):
    """Monkey-patch model.loss with a variant. Returns the original loss method."""
    original = model.loss
    factory = LOSS_VARIANTS[variant_name]
    model.loss = factory(model, config or {})
    return original


def restore_model_loss(model, original_loss):
    """Restore model.loss to its original method."""
    model.loss = original_loss


# ---------------------------------------------------------------------------
# Rho-1 selective loss
# ---------------------------------------------------------------------------

def rho1_loss_factory(model, config: dict):
    """Factory for Rho-1 selective loss masking.

    Skips loss on tokens where the model's max logit exceeds a threshold,
    focusing training on "hard" tokens the model is less confident about.

    Config keys:
        threshold (float): Max logit threshold. Tokens with max_logit >= threshold
            are masked out. Default 15.0 (post-softcap range is [-30, 30]).
    """
    threshold = config.get("threshold", 15.0)

    def rho1_loss(input_ids, target_ids):
        assert model.logit_chunk_tokens <= 0, "Rho-1 requires LOGIT_CHUNK_TOKENS=0"
        h = model(input_ids)
        h_flat = h.reshape(-1, h.shape[-1])
        y_flat = target_ids.reshape(-1)
        logits = model.softcap(h_flat @ model.tok_emb.weight.astype(h_flat.dtype).T)
        per_token_loss = nn.losses.cross_entropy(
            logits.astype(mx.float32), y_flat, reduction="none"
        )
        max_logit = mx.stop_gradient(logits.max(axis=-1))
        hard_mask = mx.stop_gradient((max_logit < threshold).astype(mx.float32))
        return (per_token_loss * hard_mask).sum() / mx.maximum(
            hard_mask.sum(), mx.array(1.0)
        )

    return rho1_loss


LOSS_VARIANTS["rho1"] = rho1_loss_factory


# ---------------------------------------------------------------------------
# Adaptive-K multi-token prediction
# ---------------------------------------------------------------------------

def adaptive_k_loss_factory(model, config: dict):
    """Factory for Adaptive-K multi-token prediction loss.

    Always computes standard N+1 prediction loss. When the model is confident
    (high logit margin between top-1 and top-2), additionally predicts N+2
    from the same hidden state, weighted by aux_weight.

    Config keys:
        margin_threshold (float): Logit margin above which N+2 prediction is
            added. Default 5.0.
        aux_weight (float): Weight for the auxiliary N+2 loss. Default 0.5.
        warmup_frac (float): Fraction of total_iters to use standard loss
            only. Default 0.2.
        total_iters (int): Total training iterations (for warmup calc).
            Default 1000.
    """
    margin_threshold = config.get("margin_threshold", 5.0)
    aux_weight = config.get("aux_weight", 0.5)
    warmup_frac = config.get("warmup_frac", 0.2)
    total_iters = config.get("total_iters", 1000)
    state = {"step": 0}

    def adaptive_k_loss(input_ids, target_ids):
        assert model.logit_chunk_tokens <= 0, "Adaptive-K requires LOGIT_CHUNK_TOKENS=0"
        state["step"] += 1

        # Forward pass
        h = model(input_ids)  # (B, L, D)
        B, L, D = h.shape
        h_flat = h.reshape(-1, D)
        y_flat = target_ids.reshape(-1)

        # Base logits and loss (N+1 prediction, always)
        logits = model.softcap(h_flat @ model.tok_emb.weight.astype(h_flat.dtype).T)
        base_loss = nn.losses.cross_entropy(
            logits.astype(mx.float32), y_flat, reduction="mean"
        )

        # During warmup, return base loss only
        if state["step"] <= warmup_frac * total_iters:
            return base_loss

        # Need at least 3 sequence positions for N+2 prediction
        if L < 3:
            return base_loss

        # Compute logit margin for N+1 prediction at positions 0..L-3
        # (need 2 future tokens for aux prediction)
        logits_2d = logits.reshape(B, L, -1)  # (B, L, V)
        logits_for_margin = logits_2d[:, :-2, :]  # (B, L-2, V)
        logits_flat_margin = logits_for_margin.reshape(-1, logits_for_margin.shape[-1])

        # Top-2 logits for margin
        sorted_logits = mx.sort(logits_flat_margin, axis=-1)
        top1 = sorted_logits[:, -1]
        top2 = sorted_logits[:, -2]
        margin = mx.stop_gradient(top1 - top2)
        high_margin = mx.stop_gradient(
            (margin > margin_threshold).astype(mx.float32)
        )

        high_margin_sum = high_margin.sum()
        # If no tokens have high margin, skip aux loss
        if float(high_margin_sum.item()) < 1:
            return base_loss

        # Aux prediction: hidden at positions 0..L-3 predict token at positions 2..L-1
        h_aux = h[:, :-2, :].reshape(-1, D)  # (B*(L-2), D)
        y_aux = target_ids[:, 2:].reshape(-1)  # (B*(L-2),)
        logits_aux = model.softcap(
            h_aux @ model.tok_emb.weight.astype(h_aux.dtype).T
        )
        aux_per_token = nn.losses.cross_entropy(
            logits_aux.astype(mx.float32), y_aux, reduction="none"
        )

        aux_loss = (aux_per_token * high_margin).sum() / mx.maximum(
            high_margin_sum, mx.array(1.0)
        )

        return base_loss + aux_weight * aux_loss

    return adaptive_k_loss


LOSS_VARIANTS["adaptive_k"] = adaptive_k_loss_factory
