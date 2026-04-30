# Antigravity Deep-Golf Experimental Log

This document tracks technical insights, failure modes, and learning encountered during the 16MB Parameter Golf challenge.

## Infrastructure & Environment Insights

### [2026-04-01] FlashAttention GQA Broadcast Error
- **Problem**: SDPA fallback crashed with `torch.compile` during the forward pass.
- **Insight**: `F.scaled_dot_product_attention` (fallback) failed because the Query heads (H) and KV heads (Hkv) were not broadcastable in the transposed [B, H, T, D] format used by SDPA.
- **Fix**: Added explicit `torch.repeat_interleave(dim=2)` for KV heads when `H != Hkv` inside the `flash_attn_3_func` fallback.
- **Impact**: Training now successfully enters the `torch.compile` and iteration loops on standard PyTorch images.

### [2026-04-01] GPU Utilization Diagnostic
- **Observation**: 0% GPU utilization with 4% CPU utilization.
- **Cause**: Repeated initialization crashes (FA3 import, kv_bank attribute, SDPA GQA mismatch).
- **Status**: **RESOLVED**. Model is now loaded in VRAM (4.6GB/80GB) on all 8 H100s. Currently waiting for `torch.compile` (approx 60-90s) before first step logs.

## Architectural Findings (Antigravity Deep-Golf)

### MLA (Multi-head Latent Attention) Integration
- **Concept**: Compressing KV projections into a latent vector (`kv_latent_dim`) to save parameters.
- **Optimization**: Saved parameters are reinvested into the MLP width (scaling from 3.0x to 3.5x).
- **Implementation Note**: Fixed the `late_qat_step` and `Muon` optimizer banking to include `kv_latent_bank` and `kv_up_bank`.

## Micro-Sweep Results

| Run ID | LR | MLP Mult | Latent Dim | BPB @ 500 | BPB @ 1500 | Status |
|--------|----|----------|------------|-----------|------------|--------|
| `micro_lr0.025_mlp3.5_ldim64` | 0.025 | 3.5 | 64 | TBD | TBD | **Executing** |
| ... | ... | ... | ... | ... | ... |
