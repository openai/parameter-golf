"""Integration tests: verify full model forward pass for all configs.

These tests construct the GPT model with each R1/R2 configuration and run
a forward pass on dummy data. This catches shape mismatches, signature errors,
and runtime crashes that unit tests on individual components miss.

Run: .venv/bin/python -m pytest tests/test_integration.py -v
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_module(path):
    import importlib.util
    name = os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = name
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def r1():
    return _load_module("train_gpt_r1.py")


@pytest.fixture(scope="module")
def r2():
    return _load_module("train_gpt_r2.py")


# Small config for fast testing
SMALL = dict(
    vocab_size=64,
    num_layers=4,
    model_dim=128,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
)

B, T = 2, 32  # batch, seq_len


def _make_dummy_data(vocab_size):
    x = torch.randint(0, vocab_size, (B, T))
    y = torch.randint(0, vocab_size, (B, T))
    return x, y


# =====================================================
# R1 integration tests
# =====================================================

class TestR1_FullForward:
    def test_r1_baseline_no_features(self, r1):
        """R1 script with all features disabled (should match baseline behavior)."""
        model = r1.GPT(**SMALL, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert torch.isfinite(loss), f"Loss not finite: {loss}"
        loss.backward()

    def test_r1_1_leakyrelu_11L_3x(self, r1):
        """R1-1: LeakyReLU + more layers + wider MLP."""
        cfg = {**SMALL, "num_layers": 6, "mlp_mult": 3}
        model = r1.GPT(**cfg, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        assert torch.isfinite(loss)
        loss.backward()

    def test_r1_2_bigram(self, r1):
        """R1-2: + BigramHash."""
        cfg = {**SMALL, "num_layers": 6, "mlp_mult": 3}
        model = r1.GPT(**cfg, bigram_vocab_size=256, bigram_dim=32, xsa_last_n=0, value_residual=False)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.bigram is not None
        assert model.bigram.embed.weight.grad is not None

    def test_r1_3_xsa(self, r1):
        """R1-3: + XSA on last 2 layers."""
        cfg = {**SMALL, "num_layers": 6, "mlp_mult": 3}
        model = r1.GPT(**cfg, bigram_vocab_size=256, bigram_dim=32, xsa_last_n=2, value_residual=False)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        assert torch.isfinite(loss)
        loss.backward()
        # Verify XSA flags
        assert model.blocks[4].attn.use_xsa is True
        assert model.blocks[5].attn.use_xsa is True
        assert model.blocks[3].attn.use_xsa is False

    def test_r1_5_value_residual(self, r1):
        """R1-5: + Value residual (THE CONFIG THAT CRASHED ON RUNPOD)."""
        cfg = {**SMALL, "num_layers": 6, "mlp_mult": 3}
        model = r1.GPT(**cfg, bigram_vocab_size=256, bigram_dim=32, xsa_last_n=2, value_residual=True)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        assert torch.isfinite(loss), f"R1-5 loss not finite: {loss}"
        loss.backward()
        # Layer 0 captures v0 but doesn't mix (no previous v0 exists)
        # Layer 1+ should have vrl_alpha with gradient (they mix with v0)
        assert hasattr(model.blocks[0].attn, "vrl_alpha")
        # vrl_alpha on layer 1+ should have gradient (they actually use v0)
        assert model.blocks[1].attn.vrl_alpha.grad is not None, "Layer 1 vrl_alpha should have gradient"

    def test_r1_full_stack(self, r1):
        """Full R1 stack: LeakyReLU + 6L + 3x + Bigram + XSA + Value Residual."""
        cfg = {**SMALL, "num_layers": 6, "mlp_mult": 3}
        model = r1.GPT(**cfg, bigram_vocab_size=256, bigram_dim=32, xsa_last_n=2, value_residual=True)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        # Run multiple forward/backward to check for accumulation issues
        for _ in range(3):
            loss = model(x, y)
            assert torch.isfinite(loss), f"Loss became non-finite"
            loss.backward()
            # Zero grads manually
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()


# =====================================================
# R2 integration tests
# =====================================================

class TestR2_FullForward:
    @pytest.mark.parametrize("mlp_type", [
        "standard", "fan", "dml_gated", "dml_orth", "fan_dml", "causal_wide", "dml_causal_wide",
    ])
    def test_all_mlp_types(self, r2, mlp_type):
        """Every MLP variant should produce a finite loss and backward without error."""
        cfg = {**SMALL, "num_layers": 4, "mlp_mult": 3}
        model = r2.GPT(**cfg, bigram_vocab_size=0, xsa_last_n=0, value_residual=False, mlp_type=mlp_type)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        assert torch.isfinite(loss), f"mlp_type={mlp_type}: loss not finite: {loss}"
        loss.backward()

    @pytest.mark.parametrize("mlp_type", ["dml_gated", "fan_dml", "causal_wide", "dml_causal_wide"])
    def test_bt_loss_mlp_types(self, r2, mlp_type):
        """MLP types with Barlow Twins should produce valid BT loss after forward."""
        cfg = {**SMALL, "num_layers": 4, "mlp_mult": 3}
        model = r2.GPT(**cfg, bigram_vocab_size=0, xsa_last_n=0, value_residual=False, mlp_type=mlp_type)
        model.train()
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        bt_total = torch.tensor(0.0)
        for block in model.blocks:
            if hasattr(block.mlp, "barlow_twins_loss"):
                bt = block.mlp.barlow_twins_loss()
                assert torch.isfinite(bt), f"BT loss not finite for {mlp_type}"
                bt_total = bt_total + bt
        total = loss + 0.01 * bt_total
        total.backward()

    def test_r2_full_stack_with_all_features(self, r2):
        """Full R2: DML-Gated MLP + BigramHash + XSA + Value Residual + BT loss."""
        cfg = {**SMALL, "num_layers": 4, "mlp_mult": 3}
        model = r2.GPT(**cfg, bigram_vocab_size=128, bigram_dim=32, xsa_last_n=2,
                       value_residual=True, mlp_type="dml_gated")
        model.train()
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        bt_total = torch.tensor(0.0)
        for block in model.blocks:
            if hasattr(block.mlp, "barlow_twins_loss"):
                bt_total = bt_total + block.mlp.barlow_twins_loss()
        total = loss + 0.01 * bt_total
        assert torch.isfinite(total), f"Total loss not finite: {total}"
        total.backward()

    def test_r2_causal_wide_fewer_layers(self, r2):
        """CausalWide with fewer layers (the intended config: wide + shallow)."""
        cfg = {**SMALL, "num_layers": 3, "mlp_mult": 5}
        model = r2.GPT(**cfg, bigram_vocab_size=128, bigram_dim=32, xsa_last_n=1,
                       value_residual=True, mlp_type="causal_wide")
        model.train()
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        bt_total = torch.tensor(0.0)
        for block in model.blocks:
            if hasattr(block.mlp, "barlow_twins_loss"):
                bt_total = bt_total + block.mlp.barlow_twins_loss()
        total = loss + 0.01 * bt_total
        assert torch.isfinite(total)
        total.backward()

    def test_r2_dml_causal_wide_fewer_layers(self, r2):
        """DML-CausalWide with fewer layers (nested causal + wide)."""
        cfg = {**SMALL, "num_layers": 3, "mlp_mult": 5}
        model = r2.GPT(**cfg, bigram_vocab_size=128, bigram_dim=32, xsa_last_n=1,
                       value_residual=True, mlp_type="dml_causal_wide")
        model.train()
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y)
        bt_total = torch.tensor(0.0)
        for block in model.blocks:
            if hasattr(block.mlp, "barlow_twins_loss"):
                bt_total = bt_total + block.mlp.barlow_twins_loss()
        total = loss + 0.01 * bt_total
        assert torch.isfinite(total)
        total.backward()


# =====================================================
# Token dropout / corrupted context integration
# =====================================================

class TestR2_15_AdversarialIntegration:
    def test_adversarial_full_forward_backward(self, r2):
        """Full stack with adversarial embedding masking."""
        cfg = {**SMALL, "num_layers": 4, "mlp_mult": 3}
        model = r2.GPT(**cfg, bigram_vocab_size=0, xsa_last_n=0,
                       value_residual=True, mlp_type="dml_gated")
        x, y = _make_dummy_data(SMALL["vocab_size"])
        mask = torch.ones(x.shape[1])
        mask[5] = 0.0
        # Per-position loss with embedding mask
        loss = model(x, y, reduction="none", embedding_mask=mask)
        assert loss.shape == (x.shape[0] * x.shape[1],)
        loss.mean().backward()

    def test_per_position_loss_shape(self, r2):
        """reduction='none' returns per-position losses."""
        cfg = {**SMALL, "num_layers": 4, "mlp_mult": 2}
        model = r2.GPT(**cfg, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        loss = model(x, y, reduction="none")
        assert loss.shape == (B * T,)
        assert torch.isfinite(loss).all()


class TestR2_DataAugmentation:
    def test_token_dropout_in_forward(self, r2):
        """Token dropout should produce valid loss on shorter sequence."""
        cfg = {**SMALL, "num_layers": 4, "mlp_mult": 2}
        model = r2.GPT(**cfg, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        # Simulate token dropout
        mask = torch.rand(T) > 0.1
        mask[0] = True
        x_drop = x[:, mask]
        y_drop = y[:, mask]
        loss = model(x_drop, y_drop)
        assert torch.isfinite(loss)
        loss.backward()

    def test_corrupted_context_in_forward(self, r2):
        """Corrupted tokens should produce valid loss."""
        cfg = {**SMALL, "num_layers": 4, "mlp_mult": 2}
        model = r2.GPT(**cfg, bigram_vocab_size=0, xsa_last_n=0, value_residual=False)
        x, y = _make_dummy_data(SMALL["vocab_size"])
        # Simulate corruption
        corrupt_mask = torch.rand(x.shape) < 0.1
        corrupt_mask[:, 0] = False
        random_tokens = torch.randint(0, SMALL["vocab_size"], x.shape)
        x_corrupt = torch.where(corrupt_mask, random_tokens, x)
        loss = model(x_corrupt, y)
        assert torch.isfinite(loss)
        loss.backward()
