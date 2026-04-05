"""Tests for adjacent-layer KV sharing in HybridGDN."""
import pytest
import torch
import sys
sys.path.insert(0, "/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments")


def test_kv_sharing_reduces_unique_params():
    """KV sharing with stride 2 on 10L GDN should reduce unique params by ~2.64M."""
    from architectures import HybridGDN
    from configs import model_a_pure_gdn

    # Model without sharing
    config_base = model_a_pure_gdn()
    model_base = HybridGDN(config_base)
    base_params = sum(p.numel() for p in model_base.parameters())

    # Model with sharing
    config_shared = model_a_pure_gdn()
    config_shared["kv_sharing_stride"] = 2
    model_shared = HybridGDN(config_shared)
    shared_params = sum(p.numel() for p in model_shared.parameters())

    saved = base_params - shared_params
    # Should save ~2.64M params (5 pairs × 528K per pair)
    assert saved > 2_000_000, f"Expected >2M params saved, got {saved:,}"
    assert saved < 3_000_000, f"Expected <3M params saved, got {saved:,}"


def test_kv_sharing_forward_pass():
    """Forward pass should work correctly with shared KV projections."""
    if not torch.cuda.is_available():
        pytest.skip("FLA GatedDeltaNet forward requires CUDA")
    from architectures import HybridGDN
    from configs import model_a_pure_gdn

    config = model_a_pure_gdn()
    config["kv_sharing_stride"] = 2
    model = HybridGDN(config)
    model.eval()

    B, T = 2, 64
    input_ids = torch.randint(0, 1024, (B, T))
    target_ids = torch.randint(0, 1024, (B, T))

    # Should not raise
    loss = model(input_ids, target_ids)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_kv_sharing_backward_pass():
    """Backward pass should propagate gradients through shared KV params."""
    if not torch.cuda.is_available():
        pytest.skip("FLA GatedDeltaNet backward requires CUDA")
    from architectures import HybridGDN
    from configs import model_a_pure_gdn

    config = model_a_pure_gdn()
    config["kv_sharing_stride"] = 2
    model = HybridGDN(config)
    model.train()

    B, T = 2, 32
    input_ids = torch.randint(0, 1024, (B, T))
    target_ids = torch.randint(0, 1024, (B, T))

    loss = model(input_ids, target_ids)
    loss.backward()

    # Check that shared k_proj in anchor layer has gradients
    anchor = model.blocks[0].recurrent
    assert anchor.k_proj.weight.grad is not None
    assert anchor.v_proj.weight.grad is not None
    # Gradient should be non-zero (receives signal from both layers 0 and 1)
    assert anchor.k_proj.weight.grad.abs().sum() > 0


def test_kv_sharing_modules_are_identical():
    """Shared layers should reference the exact same module objects."""
    from architectures import HybridGDN
    from configs import model_a_pure_gdn

    config = model_a_pure_gdn()
    config["kv_sharing_stride"] = 2
    model = HybridGDN(config)

    # Layers 0 and 1 should share K/V
    anchor = model.blocks[0].recurrent
    follower = model.blocks[1].recurrent
    assert anchor.k_proj is follower.k_proj
    assert anchor.v_proj is follower.v_proj
    assert anchor.k_conv1d is follower.k_conv1d
    assert anchor.v_conv1d is follower.v_conv1d

    # Layers 0 and 2 should NOT share (different pairs)
    non_pair = model.blocks[2].recurrent
    assert anchor.k_proj is not non_pair.k_proj


def test_kv_sharing_stride_0_is_noop():
    """stride 0 or missing should not change param count."""
    from architectures import HybridGDN
    from configs import model_a_pure_gdn

    config = model_a_pure_gdn()
    model_no_share = HybridGDN(config)

    config2 = model_a_pure_gdn()
    config2["kv_sharing_stride"] = 0
    model_stride0 = HybridGDN(config2)

    p1 = sum(p.numel() for p in model_no_share.parameters())
    p2 = sum(p.numel() for p in model_stride0.parameters())
    assert p1 == p2


# --- Config I, J, K tests ---


def test_config_i_instantiates():
    """Model I should construct a valid HybridGDN with KV sharing."""
    from configs import get_config
    from architectures import HybridGDN

    config = get_config("I")
    model = HybridGDN(config)
    assert config["kv_sharing_stride"] == 2
    assert len(model.blocks) == 10


def test_config_j_instantiates():
    """Model J should construct a 12-layer HybridGDN."""
    from configs import get_config
    from architectures import HybridGDN

    config = get_config("J")
    model = HybridGDN(config)
    assert config["kv_sharing_stride"] == 2
    assert config["model_dim"] == 480
    assert len(model.blocks) == 12


def test_config_k_instantiates():
    """Model K should construct a wider HybridGDN."""
    from configs import get_config
    from architectures import HybridGDN

    config = get_config("K")
    model = HybridGDN(config)
    assert config["kv_sharing_stride"] == 2
    assert config["model_dim"] == 544
    assert len(model.blocks) == 10


def test_config_j_k_iso_parameter():
    """Models J and K should be within ±500K params of Model A."""
    from configs import get_config
    from architectures import HybridGDN

    model_a = HybridGDN(get_config("A"))
    model_j = HybridGDN(get_config("J"))
    model_k = HybridGDN(get_config("K"))

    p_a = sum(p.numel() for p in model_a.parameters())
    p_j = sum(p.numel() for p in model_j.parameters())
    p_k = sum(p.numel() for p in model_k.parameters())

    assert abs(p_j - p_a) < 500_000, f"J differs from A by {p_j - p_a:+,}"
    assert abs(p_k - p_a) < 500_000, f"K differs from A by {p_k - p_a:+,}"
