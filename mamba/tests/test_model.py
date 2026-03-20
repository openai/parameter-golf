"""Test model shapes, gradient flow, and parameter count."""

import torch
import pytest

from ..config import Hyperparameters
from ..model import MambaHybrid, MambaBlock, AttentionBlock, Mamba2Mixer


class SmallConfig:
    """Tiny config for fast tests."""
    vocab_size = 64
    model_dim = 64
    num_layers = 4
    attn_layer_indices = [3]
    num_attn_heads = 4
    num_kv_heads = 2
    mlp_mult = 2
    rope_base = 10000.0
    qk_gain_init = 1.5
    ssm_expansion = 2
    ssm_state_dim = 8
    ssm_num_heads = 4
    ssm_conv_kernel = 4
    ssm_chunk_size = 16
    tie_embeddings = True
    tied_embed_init_std = 0.005
    logit_softcap = 30.0


def test_forward_shape():
    config = SmallConfig()
    model = MambaHybrid(config)
    B, L = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    target_ids = torch.randint(0, config.vocab_size, (B, L))
    loss = model(input_ids, target_ids)
    assert loss.shape == ()
    assert loss.item() > 0


def test_loss_is_scalar():
    config = SmallConfig()
    model = MambaHybrid(config)
    B, L = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    target_ids = torch.randint(0, config.vocab_size, (B, L))
    loss = model(input_ids, target_ids)
    assert loss.ndim == 0


def test_gradients_flow():
    config = SmallConfig()
    model = MambaHybrid(config)
    B, L = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    target_ids = torch.randint(0, config.vocab_size, (B, L))
    loss = model(input_ids, target_ids)
    loss.backward()

    # Check that at least some parameters have gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().max() > 0)
    total = sum(1 for p in model.parameters())
    assert has_grad > 0, "No parameters received gradients"
    # Many params may be zero-init (output projections), so relax threshold
    assert has_grad >= total * 0.2, f"Only {has_grad}/{total} params got nonzero gradients"


def test_block_types():
    config = SmallConfig()
    model = MambaHybrid(config)
    # Layers 0,1,2 should be MambaBlock; layer 3 should be AttentionBlock
    for i in range(3):
        assert isinstance(model.blocks[i], MambaBlock), f"Block {i} should be MambaBlock"
    assert isinstance(model.blocks[3], AttentionBlock), "Block 3 should be AttentionBlock"


def test_param_count_reasonable():
    """Check param count is in expected range for default config."""
    config = Hyperparameters()
    model = MambaHybrid(config)
    n_params = sum(p.numel() for p in model.parameters())
    # Should be roughly 17M-21M for default config
    assert 15_000_000 < n_params < 25_000_000, f"Unexpected param count: {n_params}"


def test_skip_connections():
    """Ensure U-Net skip connections are properly structured."""
    config = SmallConfig()
    model = MambaHybrid(config)
    # 4 layers -> 2 encoder + 2 decoder, 2 skip weights
    assert model.num_encoder_layers == 2
    assert model.num_decoder_layers == 2
    assert model.skip_weights.shape == (2, config.model_dim)


def test_tied_embeddings():
    config = SmallConfig()
    config.tie_embeddings = True
    model = MambaHybrid(config)
    assert model.lm_head is None
    # Forward should still work
    B, L = 2, 16
    loss = model(torch.randint(0, config.vocab_size, (B, L)), torch.randint(0, config.vocab_size, (B, L)))
    assert loss.item() > 0


def test_mamba_mixer_output_shape():
    d_model, d_inner, n_heads, d_state = 64, 128, 4, 8
    mixer = Mamba2Mixer(d_model, d_inner, n_heads, d_state, conv_kernel=4, chunk_size=16)
    B, L = 2, 32
    x = torch.randn(B, L, d_model)
    y = mixer(x, x)
    assert y.shape == (B, L, d_model)
