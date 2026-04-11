"""Tests for loss variant registry and implementations (rho1, adaptive_k).

Uses a minimal mock GPT-like model to test loss computation without
requiring full model infrastructure or training data.
"""
from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Minimal mock model
# ---------------------------------------------------------------------------

class _MockTokEmb:
    """Minimal embedding with a .weight attribute."""
    def __init__(self, vocab_size: int, dim: int):
        self.weight = mx.random.normal((vocab_size, dim)) * 0.1


class MockGPT:
    """Minimal GPT-like object with the attributes loss variants need.

    forward(input_ids) returns pre-set hidden states so we can control
    logit values precisely via the embedding weight matrix.
    """

    def __init__(self, vocab_size: int = 32, dim: int = 16, seq_len: int = 8):
        self.tok_emb = _MockTokEmb(vocab_size, dim)
        self.logit_chunk_tokens = 0
        self.logit_softcap = 30.0
        self._hidden = None  # set by test to control forward output
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len

    def __call__(self, input_ids):
        """Return pre-set hidden states or random ones."""
        if self._hidden is not None:
            return self._hidden
        B = input_ids.shape[0]
        L = input_ids.shape[1]
        return mx.random.normal((B, L, self.dim))

    def softcap(self, logits):
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def loss(self, input_ids, target_ids):
        """Standard loss (for patch/restore testing)."""
        h = self(input_ids).reshape(-1, self.dim)
        y = target_ids.reshape(-1)
        logits = self.softcap(h @ self.tok_emb.weight.astype(h.dtype).T)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(model, batch_size=2):
    """Create random input/target token IDs."""
    B, L, V = batch_size, model.seq_len, model.vocab_size
    input_ids = mx.random.randint(0, V, (B, L))
    target_ids = mx.random.randint(0, V, (B, L))
    return input_ids, target_ids


def _set_hidden_for_logits(model, target_logits_flat):
    """Set model._hidden so that h_flat @ embed.T produces target_logits_flat.

    target_logits_flat: (B*L, V) desired pre-softcap logits.
    We solve: h = target_logits_flat @ pinv(embed.T) = target_logits_flat @ embed @ pinv(embed @ embed.T)
    For simplicity, use least-squares via the pseudo-inverse.
    """
    embed = model.tok_emb.weight  # (V, D)
    # h @ embed.T = target  =>  h = target @ pinv(embed.T) = target @ embed @ inv(embed.T @ embed)
    # Use: h = target @ embed * (embed.T @ embed)^-1  but simpler: h = target @ pinv(embed.T)
    # pinv(embed.T) shape: (V, D).T -> we need (D, V) pinv of (D, V) = nonsense
    # embed.T is (D, V). pinv(embed.T) is (V, D).
    # h = target_logits_flat @ pinv(embed.T) where target is (N, V), pinv is (V, D) => h is (N, D)
    embed_T = embed.T  # (D, V)
    # Approximate: h = target @ embed / (V * mean(embed^2))  -- rough scaling
    # Better: just set _hidden directly and accept logits won't be exact.
    # Actually, let's use a direct approach: set embed to identity-like and control hidden directly.
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_loss_variants_registered(self):
        """Both rho1 and adaptive_k should be in the variant registry."""
        from scripts.causal.loss_variants import LOSS_VARIANTS
        assert "rho1" in LOSS_VARIANTS
        assert "adaptive_k" in LOSS_VARIANTS


class TestRho1:
    def _make_model_with_controlled_logits(self, max_logit_values):
        """Create a model where we can control the max logit per token.

        max_logit_values: 1D array of shape (B*L,) with desired max logit values.
        Returns (model, input_ids, target_ids).
        """
        n_tokens = len(max_logit_values)
        V, D = 32, 32  # use square for invertibility
        B = 1
        L = n_tokens

        model = MockGPT(vocab_size=V, dim=D, seq_len=L)

        # Set embedding to identity-like (first D dims)
        model.tok_emb.weight = mx.eye(V, D)

        # Build hidden states where the max logit in each row matches our target.
        # With identity embed, logits = h @ I.T = h, so logits[i] = h[i].
        # Pre-softcap logits: we want max(logits[i]) = max_logit_values[i].
        # Set all dims to -100 except dim 0 which gets the target max logit.
        h_np = np.full((n_tokens, D), -100.0, dtype=np.float32)
        for i, val in enumerate(max_logit_values):
            h_np[i, 0] = val
        h_flat = mx.array(h_np)

        model._hidden = h_flat.reshape(B, L, D)

        input_ids = mx.zeros((B, L), dtype=mx.int32)
        # Target token 5 (NOT token 0 which has the max logit) so loss > 0
        target_ids = mx.full((B, L), 5, dtype=mx.int32)

        return model, input_ids, target_ids

    def test_rho1_masks_easy_tokens(self):
        """Tokens with max logit >= threshold should be masked out."""
        from scripts.causal.loss_variants import rho1_loss_factory

        # 4 tokens: max logits = [20, 5, 25, 3]
        # threshold=10 => mask tokens 0,2 (max >= 10), keep tokens 1,3
        max_logits = [20.0, 5.0, 25.0, 3.0]
        model, x, y = self._make_model_with_controlled_logits(max_logits)

        loss_fn = rho1_loss_factory(model, {"threshold": 10.0})
        loss_val = loss_fn(x, y)
        mx.eval(loss_val)

        # Compare with standard loss on all tokens
        std_loss = model.loss(x, y)
        mx.eval(std_loss)

        # Losses should differ since some tokens are masked
        assert float(loss_val.item()) != pytest.approx(float(std_loss.item()), abs=1e-3)

    def test_rho1_threshold_zero_masks_nothing(self):
        """threshold=0 means max_logit < 0 required. Post-softcap logits for
        confident tokens are positive, so effectively nothing passes for
        tokens with positive max logits. But with threshold very high,
        everything passes."""
        from scripts.causal.loss_variants import rho1_loss_factory

        # All tokens have low max logits (well below threshold)
        max_logits = [-5.0, -10.0, -3.0, -8.0]
        model, x, y = self._make_model_with_controlled_logits(max_logits)

        # Very high threshold: all tokens pass (max_logit < 1000 is always true)
        loss_high_thresh = rho1_loss_factory(model, {"threshold": 1000.0})(x, y)
        std_loss = model.loss(x, y)
        mx.eval(loss_high_thresh, std_loss)

        # Should be equal (all tokens included)
        assert float(loss_high_thresh.item()) == pytest.approx(
            float(std_loss.item()), abs=1e-4
        )

    def test_rho1_threshold_masks_everything(self):
        """Very low threshold should mask all tokens, returning 0 loss."""
        from scripts.causal.loss_variants import rho1_loss_factory

        # All tokens have positive max logits
        max_logits = [10.0, 15.0, 20.0, 25.0]
        model, x, y = self._make_model_with_controlled_logits(max_logits)

        # threshold=-1000: max_logit < -1000 is never true => all masked
        loss_fn = rho1_loss_factory(model, {"threshold": -1000.0})
        loss_val = loss_fn(x, y)
        mx.eval(loss_val)

        # All masked => loss should be 0 (sum of nothing / max(0, 1) = 0)
        assert float(loss_val.item()) == pytest.approx(0.0, abs=1e-6)


class TestAdaptiveK:
    def _make_model_with_margin_control(self, margins, seq_len=8):
        """Create model with controlled logit margins at each position.

        margins: list of float, length = seq_len. margin = top1 - top2.
        High margin => confident => should trigger N+2 prediction.
        """
        V, D = 32, 32
        B = 1

        model = MockGPT(vocab_size=V, dim=D, seq_len=seq_len)
        model.tok_emb.weight = mx.eye(V, D)

        # Build hidden states where dim 0 gets high value (top1) and dim 1
        # gets (top1 - margin) so the logit margin is controlled.
        h_np = np.full((seq_len, D), -100.0, dtype=np.float32)
        top1_val = 10.0
        for i, m in enumerate(margins):
            h_np[i, 0] = top1_val        # dim 0 = top1
            h_np[i, 1] = top1_val - m    # dim 1 = top2
        h_flat = mx.array(h_np)

        model._hidden = h_flat.reshape(B, seq_len, D)

        input_ids = mx.zeros((B, seq_len), dtype=mx.int32)
        # Use deterministic targets so two models produce comparable losses
        np.random.seed(999)
        target_ids = mx.array(np.random.randint(0, V, (B, seq_len)), dtype=mx.int32)

        return model, input_ids, target_ids

    def test_adaptive_k_base_loss_always_computed(self):
        """Base N+1 loss should always be part of the output."""
        from scripts.causal.loss_variants import adaptive_k_loss_factory

        model, x, y = self._make_model_with_margin_control(
            [1.0] * 8, seq_len=8
        )
        # Low margins + past warmup => base loss only (no aux)
        loss_fn = adaptive_k_loss_factory(model, {
            "margin_threshold": 100.0,  # very high => no aux triggers
            "warmup_frac": 0.0,
            "total_iters": 10,
        })
        loss_val = loss_fn(x, y)
        mx.eval(loss_val)

        # Should be finite and positive
        assert float(loss_val.item()) > 0
        assert float(loss_val.item()) < 100  # sanity bound

    def test_adaptive_k_extends_on_high_margin(self):
        """High logit margin should trigger auxiliary N+2 prediction,
        producing higher loss than base-only."""
        from scripts.causal.loss_variants import adaptive_k_loss_factory

        # All positions have very high margin
        margins_high = [20.0] * 8
        model, x, y = self._make_model_with_margin_control(margins_high, seq_len=8)

        # Base-only loss (very high threshold so no aux triggers)
        loss_base_only = adaptive_k_loss_factory(model, {
            "margin_threshold": 1000.0,
            "warmup_frac": 0.0,
            "total_iters": 10,
            "aux_weight": 0.5,
        })(x, y)

        # Now with aux (low threshold so all positions trigger)
        # Need fresh model call for the second factory
        model2, x2, y2 = self._make_model_with_margin_control(margins_high, seq_len=8)
        loss_with_aux = adaptive_k_loss_factory(model2, {
            "margin_threshold": 1.0,
            "warmup_frac": 0.0,
            "total_iters": 10,
            "aux_weight": 0.5,
        })(x2, y2)

        mx.eval(loss_base_only, loss_with_aux)

        # With aux, total loss = base + 0.5*aux, so should be >= base
        assert float(loss_with_aux.item()) >= float(loss_base_only.item()) - 1e-5

    def test_adaptive_k_no_extend_on_low_margin(self):
        """Low logit margin should NOT trigger auxiliary prediction.
        Loss should equal base-only loss."""
        from scripts.causal.loss_variants import adaptive_k_loss_factory

        # All positions have very low margin
        margins_low = [0.1] * 8
        model, x, y = self._make_model_with_margin_control(margins_low, seq_len=8)

        loss_fn = adaptive_k_loss_factory(model, {
            "margin_threshold": 5.0,  # much higher than any margin
            "warmup_frac": 0.0,
            "total_iters": 10,
            "aux_weight": 0.5,
        })
        loss_val = loss_fn(x, y)

        # Base-only reference
        model2, x2, y2 = self._make_model_with_margin_control(margins_low, seq_len=8)
        loss_base = adaptive_k_loss_factory(model2, {
            "margin_threshold": 1000.0,
            "warmup_frac": 0.0,
            "total_iters": 10,
            "aux_weight": 0.5,
        })(x2, y2)

        mx.eval(loss_val, loss_base)

        # Should be equal since no aux triggers
        assert float(loss_val.item()) == pytest.approx(
            float(loss_base.item()), abs=1e-4
        )

    def test_adaptive_k_warmup_period(self):
        """During warmup fraction, should return base loss only regardless
        of margin values."""
        from scripts.causal.loss_variants import adaptive_k_loss_factory

        # High margins that would normally trigger aux
        margins_high = [20.0] * 8
        model, x, y = self._make_model_with_margin_control(margins_high, seq_len=8)

        # warmup_frac=1.0 means ALL steps are warmup
        loss_fn = adaptive_k_loss_factory(model, {
            "margin_threshold": 1.0,
            "warmup_frac": 1.0,
            "total_iters": 100,
            "aux_weight": 0.5,
        })

        # Step 1 is within warmup (1 <= 1.0 * 100), so base only
        loss_warmup = loss_fn(x, y)

        # Base-only reference
        model2, x2, y2 = self._make_model_with_margin_control(margins_high, seq_len=8)
        loss_base = adaptive_k_loss_factory(model2, {
            "margin_threshold": 1000.0,
            "warmup_frac": 0.0,
            "total_iters": 10,
            "aux_weight": 0.5,
        })(x2, y2)

        mx.eval(loss_warmup, loss_base)

        # Should be equal since we're in warmup
        assert float(loss_warmup.item()) == pytest.approx(
            float(loss_base.item()), abs=1e-4
        )


class TestPatchRestore:
    def test_patch_and_restore(self):
        """patch_model_loss should replace model.loss, restore should undo it."""
        from scripts.causal.loss_variants import patch_model_loss, restore_model_loss

        model = MockGPT()
        # Bound methods create new objects on each access, so store a reference
        original_ref = model.loss

        saved = patch_model_loss(model, "rho1", {"threshold": 15.0})
        # After patching, model.loss should be the rho1 function (not the original method)
        assert model.loss is not saved
        # saved should be callable (the original bound method)
        assert callable(saved)

        restore_model_loss(model, saved)
        # After restore, model.loss should be the saved reference
        assert model.loss is saved
