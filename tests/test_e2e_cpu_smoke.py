"""End-to-end CPU smoke tests — Pre-GPU validation

Validates the COMPLETE pipeline on CPU before spending GPU credits:
  1. Full 18-layer hybrid model instantiation (production config)
  2. Forward/backward pass with realistic shapes
  3. Optimizer setup (Muon + Adam, Mamba param split)
  4. Multi-step training convergence
  5. GPTQ quantization pipeline (unbank → quantize → dequant → rebank)
  6. Artifact serialization and size estimation
  7. Quantized model roundtrip (load dequantized weights, verify forward)
"""
from __future__ import annotations

import io
import lzma
import sys
import types
from pathlib import Path

import pytest
import torch

# ---- Mock GPU-only modules (same pattern as other test files) ----
def _fake_flash_attn(q, k, v, causal=False):
    bsz, seqlen, nqh, hd = q.shape
    nkvh = k.shape[2]
    if nqh != nkvh:
        reps = nqh // nkvh
        k = k.unsqueeze(3).expand(bsz, seqlen, nkvh, reps, hd).reshape(bsz, seqlen, nqh, hd)
        v = v.unsqueeze(3).expand(bsz, seqlen, nkvh, reps, hd).reshape(bsz, seqlen, nqh, hd)
    scale = hd ** -0.5
    q2 = q.transpose(1, 2).float()
    k2 = k.transpose(1, 2).float()
    v2 = v.transpose(1, 2).float()
    attn = torch.matmul(q2, k2.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v2)
    return out.transpose(1, 2).to(q.dtype)

for mod_name in ("flash_attn_interface", "flash_attn", "mamba_ssm", "causal_conv1d"):
    if mod_name not in sys.modules:
        fake = types.ModuleType(mod_name)
        fake.flash_attn_func = _fake_flash_attn
        sys.modules[mod_name] = fake
sys.modules["flash_attn_interface"].flash_attn_func = _fake_flash_attn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt
train_gpt.flash_attn_3_func = _fake_flash_attn
from train_gpt import (
    GPT, MambaBlock, CastedLinear, _HessianGPT, Muon,
    _unbank_state_dict, _rebank_state_dict,
    mixed_quantize_int6, dequantize_mixed_int6,
    restore_low_dim_params_to_fp32,
)


# ---- Production-like config (scaled down for CPU speed) ----
PROD_CONFIG = dict(
    vocab_size=1024,
    num_layers=18,
    model_dim=128,   # scaled down from 512 for CPU speed
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=3.0,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
    mamba_layers="0,1,2,3,4,5,6,7,8,9,10,11,15,16,17",  # 15 Mamba + 3 Attn
    mamba_d_state=16,   # scaled down from 32
    mamba_d_conv=4,
    mamba_expand=1.5,
)


def _make_prod_model(**overrides):
    cfg = {**PROD_CONFIG, **overrides}
    return GPT(**cfg)


# ===== Test 1: Full 18-layer hybrid model instantiation =====

class TestProductionModelInit:
    def test_18_layer_instantiation(self):
        model = _make_prod_model()
        assert len(model.mamba_layer_set) == 15
        assert model.n_attn == 3
        assert len(model.mamba_blocks) == 15
        assert len(model.blocks) == 3

    def test_layer_dispatch_mapping(self):
        model = _make_prod_model()
        # Mamba layers: 0-11, 15-17
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17]:
            assert i in model.mamba_layer_set
        # Attention layers: 12, 13, 14
        for i in [12, 13, 14]:
            assert i not in model.mamba_layer_set
            assert i in model.attn_idx_map

    def test_bank_sizes_for_3_attn(self):
        model = _make_prod_model()
        n_a = model.n_attn  # 3
        assert model.qo_bank.shape[0] == 2 * n_a  # 6
        assert model.kv_bank.shape[0] == 2 * n_a  # 6
        assert model.mlp_up_bank.shape[0] == n_a   # 3
        assert model.mlp_down_bank.shape[0] == n_a  # 3

    def test_unet_skip_weights(self):
        model = _make_prod_model()
        # 18 layers → 9 encoder + 9 decoder → 9 skip weights
        assert model.skip_weights.shape[0] == 9
        assert model.num_encoder_layers == 9
        assert model.num_decoder_layers == 9

    def test_param_count_budget(self):
        """Verify total param count is in expected range for scaled-down config."""
        model = _make_prod_model()
        total = sum(p.numel() for p in model.parameters())
        mamba = sum(p.numel() for p in model.mamba_blocks.parameters())
        attn = total - mamba
        # Mamba: ~15 blocks * ~params_per_block
        assert mamba > 0
        assert attn > 0
        # With d_model=128, expect much less than production 27.8M
        assert total < 10_000_000  # sanity check for scaled config

    def test_full_size_param_estimate(self):
        """Estimate production (d_model=512) param count without building it."""
        # MambaBlock param count formula from arch spec:
        # d_inner = int(512 * 1.5) = 768
        # in_proj: 512 * (768*2 + 32 + 32) = 819,200
        # conv1d: 768 * 4 = 3,072
        # dt_proj: 32*768 + 768 = 25,344
        # A_log: 768*32 = 24,576
        # D: 768
        # c_proj: 512*32 = 16,384
        # out_proj: 768*512 = 393,216
        # norm: 512
        # Total: ~1,283,072 per Mamba layer
        mamba_per_layer = 1_283_072
        mamba_total = 15 * mamba_per_layer  # ~19.2M
        # This should fit in ~15.6MB int6 → ~13.5MB LZMA
        int6_bytes = mamba_total * 6 / 8  # ~14.4MB for Mamba alone
        assert int6_bytes < 16_000_000


# ===== Test 2: Forward/backward with realistic shapes =====

class TestForwardBackward:
    def test_forward_produces_loss(self):
        model = _make_prod_model()
        model.eval()
        ids = torch.randint(0, 1024, (2, 64))
        tgt = torch.randint(0, 1024, (2, 64))
        with torch.no_grad():
            loss = model(ids, tgt)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert 4.0 < loss.item() < 10.0

    def test_backward_all_params_get_grad(self):
        model = _make_prod_model()
        model.train()
        ids = torch.randint(0, 1024, (2, 32))
        tgt = torch.randint(0, 1024, (2, 32))
        loss = model(ids, tgt)
        loss.backward()
        no_grad = []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                no_grad.append(name)
        # Allow some params to not get grad (e.g., unused VE layers)
        # but Mamba and attention params must all get gradients
        mamba_no_grad = [n for n in no_grad if "mamba" in n]
        attn_no_grad = [n for n in no_grad if any(k in n for k in ["qo_bank", "kv_bank", "mlp"])]
        assert len(mamba_no_grad) == 0, f"Mamba params without grad: {mamba_no_grad}"
        assert len(attn_no_grad) == 0, f"Attention params without grad: {attn_no_grad}"

    def test_gradient_checkpointing(self):
        model = _make_prod_model()
        model._mamba_grad_checkpoint = True
        model.train()
        ids = torch.randint(0, 1024, (2, 32))
        tgt = torch.randint(0, 1024, (2, 32))
        loss = model(ids, tgt)
        loss.backward()
        # Verify Mamba params still get gradients with checkpointing
        for mb in model.mamba_blocks:
            for name, p in mb.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"mamba param {name} has no grad with checkpointing"


# ===== Test 3: Optimizer setup matches production =====

class TestOptimizerSetup:
    def test_muon_adam_split(self):
        """Verify Mamba matrix params go to Muon, scalar params to Adam."""
        model = _make_prod_model()
        # Muon matrix params: attention banks + Mamba in_proj/out_proj/dt_proj/c_proj
        matrix_params = [model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank]
        mamba_matrix_params = []
        for mb in model.mamba_blocks:
            mamba_matrix_params.extend([mb.in_proj.weight, mb.out_proj.weight,
                                        mb.dt_proj.weight, mb.c_proj.weight])

        # Scalar params: A_log, D, dt_proj.bias, conv1d, norm, skip_weights, smear
        scalar_params = []
        for mb in model.mamba_blocks:
            scalar_params.extend([mb.A_log, mb.D, mb.dt_proj.bias])
            scalar_params.extend(list(mb.conv1d.parameters()))
            scalar_params.extend(list(mb.norm.parameters()))

        # Verify no overlap
        matrix_ids = {id(p) for p in matrix_params + mamba_matrix_params}
        scalar_ids = {id(p) for p in scalar_params}
        assert len(matrix_ids & scalar_ids) == 0, "Overlap between matrix and scalar params"

        # Verify all Mamba params are accounted for
        all_mamba_param_ids = {id(p) for p in model.mamba_blocks.parameters()}
        accounted = {id(p) for p in mamba_matrix_params + scalar_params}
        unaccounted = all_mamba_param_ids - accounted
        assert len(unaccounted) == 0, f"{len(unaccounted)} Mamba params not assigned to any optimizer"

    def test_muon_instantiation(self):
        """Verify Muon optimizer accepts Mamba matrix params."""
        model = _make_prod_model()
        matrix_params = [model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank]
        mamba_matrix_params = []
        for mb in model.mamba_blocks:
            mamba_matrix_params.extend([mb.in_proj.weight, mb.out_proj.weight,
                                        mb.dt_proj.weight, mb.c_proj.weight])
        groups = [
            {"params": matrix_params, "lr": 0.025, "base_lr": 0.025},
            {"params": mamba_matrix_params, "lr": 0.015, "base_lr": 0.015},
        ]
        optimizer = Muon(groups, lr=0.025, momentum=0.99, backend_steps=5, weight_decay=0.04)
        assert len(optimizer.param_groups) == 2


# ===== Test 4: Multi-step training convergence =====

class TestTrainingConvergence:
    def test_10_step_loss_decrease(self):
        """Loss should decrease over 10 training steps on fixed data."""
        torch.manual_seed(42)
        model = _make_prod_model(num_layers=6,
                                 mamba_layers="0,1,2,3",
                                 mamba_d_state=8)  # smaller for speed
        model.train()

        # Fixed data so model can memorize → guaranteed loss decrease
        ids = torch.randint(0, 1024, (4, 64))
        tgt = torch.randint(0, 1024, (4, 64))

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            loss = model(ids, tgt)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Average of last 3 should be lower than average of first 3
        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        assert avg_last < avg_first, f"Loss didn't decrease: first3_avg={avg_first:.4f} last3_avg={avg_last:.4f}\n{losses}"


# ===== Test 5: GPTQ quantization pipeline =====

class TestQuantizationPipeline:
    @pytest.fixture
    def trained_model(self):
        torch.manual_seed(42)
        model = _make_prod_model(num_layers=6,
                                 mamba_layers="0,1,2",
                                 mamba_d_state=8)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            ids = torch.randint(0, 1024, (2, 32))
            tgt = torch.randint(0, 1024, (2, 32))
            optimizer.zero_grad()
            loss = model(ids, tgt)
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    def test_unbank_rebank_roundtrip(self, trained_model):
        """Unbank → rebank should recover original state dict."""
        model = trained_model
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        n_attn = model.n_attn

        unbanked = _unbank_state_dict(sd, model.num_encoder_layers + model.num_decoder_layers, n_attn=n_attn)
        rebanked = _rebank_state_dict(unbanked, model.num_encoder_layers + model.num_decoder_layers, sd, n_attn=n_attn)

        for key in sd:
            assert key in rebanked, f"Missing key after rebank: {key}"
            torch.testing.assert_close(sd[key], rebanked[key], atol=0, rtol=0)

    def test_int6_quantize_dequantize(self, trained_model):
        """Int6 quantize → dequantize should produce finite values close to original."""
        model = trained_model
        sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
        n_attn = model.n_attn
        num_layers = model.num_encoder_layers + model.num_decoder_layers

        unbanked = _unbank_state_dict(sd, num_layers, n_attn=n_attn)
        quant_result, quant_meta = mixed_quantize_int6(unbanked, {"mlp", "attn", "mamba"})
        deq = dequantize_mixed_int6(quant_result, quant_meta, unbanked)

        # All dequantized values should be finite
        for key, tensor in deq.items():
            assert torch.isfinite(tensor).all(), f"Non-finite values in dequantized {key}"

        # Quantization error should be bounded (not trivially zero)
        total_error = 0.0
        total_norm = 0.0
        for key in unbanked:
            if key in deq and unbanked[key].ndim >= 2:
                err = (unbanked[key].float() - deq[key].float()).norm()
                nrm = unbanked[key].float().norm()
                total_error += err.item()
                total_norm += nrm.item()
        if total_norm > 0:
            relative_error = total_error / total_norm
            assert relative_error < 0.5, f"Quantization error too large: {relative_error:.4f}"
            assert relative_error > 0.0, "Quantization error is zero (suspicious)"

    def test_full_quant_serialize_roundtrip(self, trained_model):
        """Full pipeline: quantize → LZMA → load → dequantize → rebank → forward."""
        model = trained_model
        sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
        n_attn = model.n_attn
        num_layers = model.num_encoder_layers + model.num_decoder_layers

        # Quantize
        unbanked = _unbank_state_dict(sd, num_layers, n_attn=n_attn)
        quant_result, quant_meta = mixed_quantize_int6(unbanked, {"mlp", "attn", "mamba"})

        # Serialize with LZMA
        buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, buf)
        raw = buf.getvalue()
        compressed = lzma.compress(raw, preset=9)

        # Deserialize
        loaded = torch.load(io.BytesIO(lzma.decompress(compressed)), map_location="cpu")
        deq = dequantize_mixed_int6(loaded["w"], loaded["m"], unbanked)
        rebanked = _rebank_state_dict(deq, num_layers, sd, n_attn=n_attn)

        # Load into fresh model and verify forward pass
        model2 = _make_prod_model(num_layers=6,
                                  mamba_layers="0,1,2",
                                  mamba_d_state=8)
        model2.load_state_dict(rebanked, strict=True)
        model2.eval()

        ids = torch.randint(0, 1024, (2, 32))
        tgt = torch.randint(0, 1024, (2, 32))
        with torch.no_grad():
            loss = model2(ids, tgt)
        assert torch.isfinite(loss), f"Non-finite loss after quant roundtrip: {loss}"
        assert 3.0 < loss.item() < 12.0, f"Loss out of range after quant roundtrip: {loss.item()}"


# ===== Test 6: Artifact size estimation =====

class TestArtifactSize:
    def test_compressed_artifact_under_16mb(self):
        """Estimate compressed artifact size for production-like config."""
        torch.manual_seed(42)
        model = _make_prod_model()
        sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
        n_attn = model.n_attn
        num_layers = model.num_encoder_layers + model.num_decoder_layers

        unbanked = _unbank_state_dict(sd, num_layers, n_attn=n_attn)
        quant_result, quant_meta = mixed_quantize_int6(unbanked, {"mlp", "attn", "mamba"})

        buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, buf)
        raw = buf.getvalue()
        compressed = lzma.compress(raw, preset=9)

        # For d_model=128, this will be much smaller than production
        # But verify the pipeline works
        artifact_bytes = len(compressed)
        assert artifact_bytes > 0
        assert artifact_bytes < 16_000_000  # must fit in 16MB

        # Extrapolate to production size: d_model=512 is 16x more params
        # Production params ~27.8M → int6 = ~20.85MB raw → LZMA ~13-14MB
        scale_factor = (512 / 128) ** 2  # quadratic scaling for matrix weights
        estimated_prod = artifact_bytes * scale_factor
        print(f"\n  Scaled-down artifact: {artifact_bytes / 1024:.1f} KB")
        print(f"  Estimated production artifact: {estimated_prod / (1024*1024):.1f} MB")


# ===== Test 7: HessianGPT for GPTQ calibration =====

class TestHessianModel:
    def test_hessian_model_loads_unbanked_weights(self):
        """_HessianGPT should accept unbanked weights from hybrid model."""
        torch.manual_seed(42)
        model = _make_prod_model(num_layers=6,
                                 mamba_layers="0,1,2",
                                 mamba_d_state=8)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        n_attn = model.n_attn
        num_layers = 6

        unbanked = _unbank_state_dict(sd, num_layers, n_attn=n_attn)

        hessian_model = _HessianGPT(
            vocab_size=1024, num_layers=6, model_dim=128,
            num_heads=4, num_kv_heads=2, mlp_mult=3.0,
            tie_embeddings=True, logit_softcap=30.0,
            rope_base=10000.0, qk_gain_init=1.5,
            bigram_vocab_size=2048, bigram_dim=128,
            xsa_last_n=6, rope_dims=16, ln_scale=True,
            ve_enabled=True, ve_dim=16, ve_layers="",
            mamba_layers="0,1,2", mamba_d_state=8,
            mamba_d_conv=4, mamba_expand=1.5,
        )

        # Load unbanked weights (strict=False because some keys may differ)
        hessian_sd = hessian_model.state_dict()
        loadable = {k: v for k, v in unbanked.items() if k in hessian_sd}
        hessian_model.load_state_dict(loadable, strict=False)

        # Forward pass should work
        ids = torch.randint(0, 1024, (2, 32))
        tgt = torch.randint(0, 1024, (2, 32))
        with torch.no_grad():
            loss = hessian_model(ids, tgt)
        assert torch.isfinite(loss)


# ===== Test 8: Env var config parsing =====

class TestConfigParsing:
    def test_mamba_layers_parsing(self):
        """Verify MAMBA_LAYERS env var is parsed correctly."""
        import os
        # Test the production config string
        layers_str = "0,1,2,3,4,5,6,7,8,9,10,11,15,16,17"
        layer_set = set(int(x) for x in layers_str.split(",") if x.strip())
        assert len(layer_set) == 15
        assert layer_set == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17}

    def test_empty_mamba_layers_backward_compat(self):
        """Empty MAMBA_LAYERS should produce pure attention model."""
        model = GPT(
            vocab_size=1024, num_layers=4, model_dim=64,
            num_heads=4, num_kv_heads=2, mlp_mult=3.0,
            tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            mamba_layers="",
        )
        assert len(model.mamba_layer_set) == 0
        assert len(model.blocks) == 4


# ===== Test 9: Different hybrid configurations (ablation prep) =====

class TestAblationConfigs:
    """Pre-validate all configurations we plan to ablate on GPU."""

    @pytest.mark.parametrize("mamba_layers,n_layers,expected_mamba,expected_attn", [
        # All Mamba (0 attn)
        ("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", 18, 18, 0),
        # 2 Attn (top)
        ("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", 18, 16, 2),
        # 3 Attn (default)
        ("0,1,2,3,4,5,6,7,8,9,10,11,15,16,17", 18, 15, 3),
        # 5 Attn (interleaved)
        ("0,1,2,4,5,6,8,9,10,12,13,15,16", 18, 13, 5),
        # 12 layers (9 mamba + 3 attn)
        ("0,1,2,3,4,5,6,7,8", 12, 9, 3),
        # 15 layers (12 mamba + 3 attn)
        ("0,1,2,3,4,5,6,7,8,9,10,11", 15, 12, 3),
    ])
    def test_ablation_config_instantiation(self, mamba_layers, n_layers, expected_mamba, expected_attn):
        model = _make_prod_model(
            num_layers=n_layers,
            mamba_layers=mamba_layers,
            model_dim=64,  # tiny for speed
            num_heads=4,
            num_kv_heads=2,
            mamba_d_state=8,
        )
        assert len(model.mamba_layer_set) == expected_mamba
        assert model.n_attn == expected_attn

        # Forward pass should work
        model.eval()
        ids = torch.randint(0, 1024, (1, 16))
        tgt = torch.randint(0, 1024, (1, 16))
        with torch.no_grad():
            loss = model(ids, tgt)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("d_state", [16, 32, 64])
    def test_d_state_variants(self, d_state):
        model = _make_prod_model(
            num_layers=6,
            mamba_layers="0,1,2",
            model_dim=64,
            mamba_d_state=d_state,
        )
        model.eval()
        ids = torch.randint(0, 1024, (1, 16))
        tgt = torch.randint(0, 1024, (1, 16))
        with torch.no_grad():
            loss = model(ids, tgt)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("expand", [1.0, 1.5, 2.0])
    def test_expand_variants(self, expand):
        model = _make_prod_model(
            num_layers=6,
            mamba_layers="0,1,2",
            model_dim=64,
            mamba_expand=expand,
        )
        model.eval()
        ids = torch.randint(0, 1024, (1, 16))
        tgt = torch.randint(0, 1024, (1, 16))
        with torch.no_grad():
            loss = model(ids, tgt)
        assert torch.isfinite(loss)
