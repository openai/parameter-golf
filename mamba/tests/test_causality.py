"""Test that the model and Mamba blocks are causal — no future token leakage."""

import torch
import pytest

from ..config import Hyperparameters
from ..model import MambaHybrid, MambaBlock, Mamba2Mixer


class SmallConfig:
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


def test_mamba_block_causality():
    """Perturbing a future token should not change past outputs in a MambaBlock."""
    torch.manual_seed(42)
    d_model = 64
    block = MambaBlock(
        d_model=d_model, d_inner=128, n_heads=4, d_state=8,
        conv_kernel=4, chunk_size=16,
    )
    block.eval()

    B, L = 1, 32
    x = torch.randn(B, L, d_model)
    x0 = torch.randn(B, L, d_model)

    # Forward with original input
    with torch.no_grad():
        y_orig = block(x, x0).clone()

    # Perturb position 16 (future relative to positions 0-15)
    perturb_pos = 16
    x_perturbed = x.clone()
    x_perturbed[:, perturb_pos:] += torch.randn_like(x[:, perturb_pos:]) * 10.0
    x0_perturbed = x0.clone()
    x0_perturbed[:, perturb_pos:] += torch.randn_like(x0[:, perturb_pos:]) * 10.0

    with torch.no_grad():
        y_perturbed = block(x_perturbed, x0_perturbed).clone()

    # Positions before perturbation should be identical
    torch.testing.assert_close(
        y_orig[:, :perturb_pos],
        y_perturbed[:, :perturb_pos],
        atol=1e-5, rtol=1e-4,
        msg="MambaBlock leaked future information to past positions",
    )

    # Positions at/after perturbation should differ
    diff = (y_orig[:, perturb_pos:] - y_perturbed[:, perturb_pos:]).abs().max()
    assert diff > 1e-3, "Perturbation at future position had no effect (unexpected)"


def test_mamba_mixer_causality():
    """Directly test the Mamba2Mixer for causality."""
    torch.manual_seed(0)
    d_model, d_inner, n_heads, d_state = 64, 128, 4, 8
    mixer = Mamba2Mixer(d_model, d_inner, n_heads, d_state, conv_kernel=4, chunk_size=8)
    mixer.eval()

    B, L = 1, 24
    x = torch.randn(B, L, d_model)
    perturb_pos = 12

    with torch.no_grad():
        y_orig = mixer(x, x).clone()

    x_pert = x.clone()
    x_pert[:, perturb_pos:] += torch.randn_like(x[:, perturb_pos:]) * 5.0

    with torch.no_grad():
        y_pert = mixer(x_pert, x_pert).clone()

    # Past positions unaffected
    torch.testing.assert_close(
        y_orig[:, :perturb_pos],
        y_pert[:, :perturb_pos],
        atol=1e-5, rtol=1e-4,
    )


def test_full_model_causality():
    """Test full MambaHybrid model causality via Jacobian check."""
    torch.manual_seed(123)
    config = SmallConfig()
    model = MambaHybrid(config)
    model.eval()

    B, L = 1, 16
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    target_ids = torch.randint(0, config.vocab_size, (B, L))

    # Run forward twice: original and with perturbed future tokens
    perturb_pos = 8
    input_ids_pert = input_ids.clone()
    input_ids_pert[:, perturb_pos:] = torch.randint(0, config.vocab_size, (B, L - perturb_pos))

    with torch.no_grad():
        # Get intermediate representations by hooking the final norm
        reps_orig = []
        reps_pert = []

        def hook_orig(module, input, output):
            reps_orig.append(output.clone())

        def hook_pert(module, input, output):
            reps_pert.append(output.clone())

        h = model.final_norm.register_forward_hook(hook_orig)
        model(input_ids, target_ids)
        h.remove()

        h = model.final_norm.register_forward_hook(hook_pert)
        model(input_ids_pert, target_ids)
        h.remove()

    # Past positions should produce identical representations
    torch.testing.assert_close(
        reps_orig[0][:, :perturb_pos],
        reps_pert[0][:, :perturb_pos],
        atol=1e-5, rtol=1e-4,
        msg="Full model leaked future information",
    )
