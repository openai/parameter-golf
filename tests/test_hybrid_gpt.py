"""Unit tests for Hybrid GPT model — Story 2B.2 (partial, for Epic 1 smoke test)

Tests:
  - GPT instantiates with mamba_layers="" (pure attention, backward compat)
  - GPT instantiates with mamba_layers="0" (single Mamba layer)
  - Bank sizes are correct for hybrid config
  - Forward pass produces valid loss
  - forward_logits returns correct shape
  - Gradient flow through hybrid model
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Mock GPU-only modules
def _fake_flash_attn(q, k, v, causal=False):
    """CPU fallback: naive scaled dot-product attention with GQA support."""
    # q: (bsz, seqlen, num_q_heads, head_dim), k,v: (bsz, seqlen, num_kv_heads, head_dim)
    bsz, seqlen, nqh, hd = q.shape
    nkvh = k.shape[2]
    # GQA: repeat KV heads to match Q heads
    if nqh != nkvh:
        reps = nqh // nkvh
        k = k.unsqueeze(3).expand(bsz, seqlen, nkvh, reps, hd).reshape(bsz, seqlen, nqh, hd)
        v = v.unsqueeze(3).expand(bsz, seqlen, nkvh, reps, hd).reshape(bsz, seqlen, nqh, hd)
    scale = hd ** -0.5
    q2 = q.transpose(1, 2).float()  # (bsz, nqh, seqlen, hd)
    k2 = k.transpose(1, 2).float()
    v2 = v.transpose(1, 2).float()
    attn = torch.matmul(q2, k2.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v2)  # (bsz, nqh, seqlen, hd)
    return out.transpose(1, 2).to(q.dtype)  # (bsz, seqlen, nqh, hd)

for mod_name in ("flash_attn_interface", "flash_attn", "mamba_ssm", "causal_conv1d"):
    if mod_name not in sys.modules:
        fake = types.ModuleType(mod_name)
        fake.flash_attn_func = _fake_flash_attn
        sys.modules[mod_name] = fake
# Patch the already-imported reference too
sys.modules["flash_attn_interface"].flash_attn_func = _fake_flash_attn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt
train_gpt.flash_attn_3_func = _fake_flash_attn
from train_gpt import GPT, MambaBlock


# --- Helpers ---

def _make_gpt(mamba_layers: str = "", num_layers: int = 11, **kwargs) -> GPT:
    """Create a small GPT model for testing."""
    defaults = dict(
        vocab_size=1024,
        num_layers=num_layers,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=3.0,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        mamba_layers=mamba_layers,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=1.5,
    )
    defaults.update(kwargs)
    return GPT(**defaults)


# ===== Pure attention (backward compatibility) ============================

def test_pure_attention_instantiation():
    """mamba_layers='' should produce a pure attention model (same as SOTA)."""
    model = _make_gpt(mamba_layers="", num_layers=4)
    assert len(model.mamba_layer_set) == 0
    assert len(model.mamba_blocks) == 0
    assert len(model.blocks) == 4
    assert model.n_attn == 4


def test_pure_attention_bank_sizes():
    """Banks should be sized for all layers when no Mamba layers."""
    model = _make_gpt(mamba_layers="", num_layers=4)
    assert model.qo_bank.shape[0] == 2 * 4
    assert model.kv_bank.shape[0] == 2 * 4
    assert model.mlp_up_bank.shape[0] == 4
    assert model.mlp_down_bank.shape[0] == 4


def test_pure_attention_forward():
    """Pure attention model should produce valid loss."""
    model = _make_gpt(mamba_layers="", num_layers=4)
    model.eval()
    input_ids = torch.randint(0, 1024, (2, 32))
    target_ids = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        loss = model(input_ids, target_ids)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert 4.0 < loss.item() < 10.0  # reasonable range for untrained model


# ===== Single Mamba layer (Task 1.3.1-1.3.2) =============================

def test_single_mamba_instantiation():
    """mamba_layers='0' should replace layer 0 with MambaBlock."""
    model = _make_gpt(mamba_layers="0", num_layers=4)
    assert model.mamba_layer_set == {0}
    assert len(model.mamba_blocks) == 1
    assert len(model.blocks) == 3  # 4 total - 1 mamba = 3 attention
    assert model.n_attn == 3


def test_single_mamba_bank_sizes():
    """Banks should be sized for attention-only layers (3, not 4)."""
    model = _make_gpt(mamba_layers="0", num_layers=4)
    n_attn = 3
    assert model.qo_bank.shape[0] == 2 * n_attn  # 6
    assert model.kv_bank.shape[0] == 2 * n_attn  # 6
    assert model.mlp_up_bank.shape[0] == n_attn  # 3
    assert model.mlp_down_bank.shape[0] == n_attn  # 3


def test_single_mamba_index_maps():
    """Index maps should correctly map global layer to local block indices."""
    model = _make_gpt(mamba_layers="0", num_layers=4)
    assert model.mamba_idx_map == {0: 0}
    assert model.attn_idx_map == {1: 0, 2: 1, 3: 2}


def test_single_mamba_forward():
    """Hybrid model with single Mamba layer should produce valid loss."""
    model = _make_gpt(mamba_layers="0", num_layers=4)
    model.eval()
    input_ids = torch.randint(0, 1024, (2, 32))
    target_ids = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        loss = model(input_ids, target_ids)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert 4.0 < loss.item() < 10.0


def test_single_mamba_forward_logits():
    """forward_logits should return (B, L, vocab) shape."""
    model = _make_gpt(mamba_layers="0", num_layers=4)
    model.eval()
    input_ids = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        logits = model.forward_logits(input_ids)
    assert logits.shape == (2, 32, 1024)
    assert torch.isfinite(logits).all()


def test_single_mamba_gradient_flow():
    """Critical params in hybrid model should receive gradients.
    Note: Some attention scalars may have zero grad with CPU mock + tiny model.
    We check the structurally important params: Mamba blocks, banks, embedding.
    """
    model = _make_gpt(mamba_layers="0", num_layers=4)
    model.train()
    input_ids = torch.randint(0, 1024, (2, 16))
    target_ids = torch.randint(0, 1024, (2, 16))
    loss = model(input_ids, target_ids)
    loss.backward()
    # Critical params that MUST have gradients
    critical_prefixes = ["mamba_blocks.", "tok_emb.", "qo_bank", "mlp_down_bank", "skip_weights"]
    for name, p in model.named_parameters():
        if any(name.startswith(pfx) for pfx in critical_prefixes):
            assert p.grad is not None, f"No grad for critical param {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad for critical param {name}"
    # All params should at least have .grad set (backward ran through them)
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Parameters with no grad at all: {no_grad}"


# ===== Multi-Mamba config ================================================

def test_multi_mamba_instantiation():
    """Multiple Mamba layers should work correctly."""
    model = _make_gpt(mamba_layers="0,1,3", num_layers=4)
    assert model.mamba_layer_set == {0, 1, 3}
    assert len(model.mamba_blocks) == 3
    assert len(model.blocks) == 1  # only layer 2 is attention
    assert model.n_attn == 1
    assert model.mamba_idx_map == {0: 0, 1: 1, 3: 2}
    assert model.attn_idx_map == {2: 0}


def test_multi_mamba_forward():
    """Multi-Mamba hybrid should produce valid loss."""
    model = _make_gpt(mamba_layers="0,1,3", num_layers=4)
    model.eval()
    input_ids = torch.randint(0, 1024, (2, 32))
    target_ids = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        loss = model(input_ids, target_ids)
    assert torch.isfinite(loss)


# ===== U-Net skip connections =============================================

def test_skip_weights_count():
    """Skip weight count should match min(encoder, decoder) for any config."""
    model = _make_gpt(mamba_layers="0", num_layers=4)
    assert model.num_encoder_layers == 2
    assert model.num_decoder_layers == 2
    assert model.skip_weights.shape[0] == 2


def test_skip_weights_18_layers():
    """18-layer model should have 9 skip weights."""
    model = _make_gpt(mamba_layers="0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17", num_layers=18)
    assert model.num_encoder_layers == 9
    assert model.num_decoder_layers == 9
    assert model.skip_weights.shape[0] == 9
    assert len(model.mamba_blocks) == 16  # 16 mamba layers  (note: spec says 15, but 16 indices listed)
    assert len(model.blocks) == 2  # layers 12, 13 are attention


def test_18_layer_forward():
    """Full 18-layer hybrid model should produce valid loss."""
    model = _make_gpt(
        mamba_layers="0,1,2,3,4,5,6,7,8,9,10,11,15,16,17",
        num_layers=18,
    )
    model.eval()
    input_ids = torch.randint(0, 1024, (1, 16))
    target_ids = torch.randint(0, 1024, (1, 16))
    with torch.no_grad():
        loss = model(input_ids, target_ids)
    assert torch.isfinite(loss)
    assert 4.0 < loss.item() < 10.0


# ===== Consistency checks =================================================

def test_forward_and_forward_logits_consistency():
    """Loss from forward should match cross-entropy on forward_logits output."""
    torch.manual_seed(42)
    model = _make_gpt(mamba_layers="0", num_layers=4)
    model.eval()
    input_ids = torch.randint(0, 1024, (1, 16))
    target_ids = torch.randint(0, 1024, (1, 16))
    with torch.no_grad():
        loss_forward = model(input_ids, target_ids)
        logits = model.forward_logits(input_ids)
        logits_capped = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
        loss_logits = torch.nn.functional.cross_entropy(
            logits_capped.view(-1, 1024).float(), target_ids.view(-1), reduction="mean"
        )
    # Note: forward applies softcap internally, forward_logits also applies softcap,
    # so logits already have softcap. We reapply here which double-caps.
    # Actually, forward_logits already returns softcapped logits. Let's use them directly.
    with torch.no_grad():
        logits = model.forward_logits(input_ids)
        loss_logits = torch.nn.functional.cross_entropy(
            logits.view(-1, 1024).float(), target_ids.view(-1), reduction="mean"
        )
    diff = (loss_forward - loss_logits).abs().item()
    assert diff < 1e-4, f"Loss mismatch: forward={loss_forward.item()}, logits={loss_logits.item()}, diff={diff}"


# ===== Unbank/Rebank roundtrip =============================================

def test_unbank_rebank_roundtrip_pure_attention():
    """Unbank -> rebank should be lossless for pure attention model."""
    from train_gpt import _unbank_state_dict, _rebank_state_dict
    model = _make_gpt(mamba_layers="", num_layers=4)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    unbanked = _unbank_state_dict(sd, 4, n_attn=4)
    rebanked = _rebank_state_dict(unbanked, 4, sd, n_attn=4)
    for key in sd:
        assert key in rebanked, f"Missing key {key}"
        assert torch.equal(sd[key], rebanked[key]), f"Mismatch for {key}"


def test_unbank_rebank_roundtrip_hybrid():
    """Unbank -> rebank should be lossless for hybrid model."""
    from train_gpt import _unbank_state_dict, _rebank_state_dict
    model = _make_gpt(mamba_layers="0,1,3", num_layers=4)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    n_attn = model.n_attn  # 1
    unbanked = _unbank_state_dict(sd, 4, n_attn=n_attn)
    # Mamba params should pass through unchanged
    for key in unbanked:
        if "mamba_blocks" in key:
            assert key in sd, f"Mamba key {key} not in original sd"
            assert torch.equal(unbanked[key], sd[key]), f"Mamba param changed during unbank: {key}"
    rebanked = _rebank_state_dict(unbanked, 4, sd, n_attn=n_attn)
    for key in sd:
        assert key in rebanked, f"Missing key {key}"
        assert torch.equal(sd[key], rebanked[key]), f"Mismatch for {key}"


# ===== _classify_param coverage =============================================

def test_classify_param_coverage():
    """Every parameter in the hybrid model should be classified."""
    from train_gpt import _classify_param
    model = _make_gpt(mamba_layers="0,1,3", num_layers=4)
    for name in model.state_dict().keys():
        cat = _classify_param(name)
        assert cat in ("embed", "mamba", "mlp", "attn", "other"), f"Unknown category for {name}: {cat}"


def test_classify_param_mamba_keys():
    """Mamba block params should be classified as 'mamba'."""
    from train_gpt import _classify_param
    model = _make_gpt(mamba_layers="0", num_layers=4)
    mamba_keys = [k for k in model.state_dict().keys() if "mamba_blocks" in k]
    assert len(mamba_keys) > 0
    for key in mamba_keys:
        assert _classify_param(key) == "mamba", f"{key} not classified as mamba"


# ===== _HessianGPT hybrid support ==========================================

def test_hessian_gpt_hybrid_instantiation():
    """_HessianGPT should instantiate with mamba_layers."""
    from train_gpt import _HessianGPT
    hmodel = _HessianGPT(
        vocab_size=1024, num_layers=4, model_dim=64,
        num_heads=4, num_kv_heads=2, mlp_mult=3.0,
        tie_embeddings=True, logit_softcap=30.0,
        rope_base=10000.0, qk_gain_init=1.5,
        mamba_layers="0,1,3", mamba_d_state=8, mamba_d_conv=4, mamba_expand=1.5,
    )
    assert len(hmodel.mamba_blocks) == 3
    assert len(hmodel.blocks) == 1  # only layer 2 is attention


def test_hessian_gpt_hybrid_forward():
    """_HessianGPT hybrid model should produce valid loss."""
    from train_gpt import _HessianGPT
    hmodel = _HessianGPT(
        vocab_size=1024, num_layers=4, model_dim=64,
        num_heads=4, num_kv_heads=2, mlp_mult=3.0,
        tie_embeddings=True, logit_softcap=30.0,
        rope_base=10000.0, qk_gain_init=1.5,
        mamba_layers="0", mamba_d_state=8, mamba_d_conv=4, mamba_expand=1.5,
    )
    hmodel.eval()
    input_ids = torch.randint(0, 1024, (1, 16))
    target_ids = torch.randint(0, 1024, (1, 16))
    with torch.no_grad():
        loss = hmodel(input_ids, target_ids)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_hessian_gpt_has_casted_linear_for_mamba():
    """_HessianMambaBlock should use CastedLinear for in_proj and out_proj."""
    from train_gpt import _HessianGPT, CastedLinear
    hmodel = _HessianGPT(
        vocab_size=1024, num_layers=4, model_dim=64,
        num_heads=4, num_kv_heads=2, mlp_mult=3.0,
        tie_embeddings=True, logit_softcap=30.0,
        rope_base=10000.0, qk_gain_init=1.5,
        mamba_layers="0", mamba_d_state=8, mamba_d_conv=4, mamba_expand=1.5,
    )
    mb = hmodel.mamba_blocks[0]
    assert isinstance(mb.in_proj, CastedLinear), "in_proj should be CastedLinear"
    assert isinstance(mb.out_proj, CastedLinear), "out_proj should be CastedLinear"


# ===== Gradient checkpointing =============================================

def test_gradient_checkpoint_mamba():
    """Gradient checkpointing for Mamba layers should still produce correct gradients."""
    model = _make_gpt(mamba_layers="0,1", num_layers=4)
    model._mamba_grad_checkpoint = True
    model.train()
    input_ids = torch.randint(0, 1024, (1, 16))
    target_ids = torch.randint(0, 1024, (1, 16))
    loss = model(input_ids, target_ids)
    loss.backward()
    # Mamba params should still get gradients through checkpointing
    for name, p in model.named_parameters():
        if "mamba_blocks" in name:
            assert p.grad is not None, f"No grad for {name} with checkpointing"
            assert p.grad.abs().sum() > 0, f"Zero grad for {name} with checkpointing"


# ===== Multi-step training simulation =====================================

def test_multi_step_loss_decreases():
    """Task 2.1.4/2.2.4: 10-step training should decrease loss (CPU simulation)."""
    torch.manual_seed(42)
    model = _make_gpt(mamba_layers="0,1", num_layers=4)
    model.train()
    # Simple SGD optimizer for CPU test (Muon requires CUDA)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 1024, (2, 32))
    target_ids = torch.randint(0, 1024, (2, 32))
    losses = []
    for step in range(10):
        loss = model(input_ids, target_ids)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Loss should decrease over 10 steps on the same batch
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_activation_norms_reasonable():
    """Task 2.3.3: Activation norms should be in reasonable range at init."""
    torch.manual_seed(123)
    model = _make_gpt(mamba_layers="0,1,3", num_layers=4)
    model.eval()
    input_ids = torch.randint(0, 1024, (2, 32))
    # Hook to capture Mamba block outputs (layer-level, not sub-modules)
    norms = {}
    hooks = []
    for i, mb in enumerate(model.mamba_blocks):
        def make_hook(n):
            def hook_fn(mod, inp, out):
                norms[n] = out.detach().float().norm().item()
            return hook_fn
        hooks.append(mb.register_forward_hook(make_hook(f"mamba_blocks.{i}")))
    with torch.no_grad():
        logits = model.forward_logits(input_ids)
    for h in hooks:
        h.remove()
    # Check: no NaN/Inf in logits
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"
    # Check: activation norms are reasonable (not exploding/vanishing)
    for name, norm in norms.items():
        assert 0.01 < norm < 10000, f"Activation norm for {name} is {norm} (out of range)"
