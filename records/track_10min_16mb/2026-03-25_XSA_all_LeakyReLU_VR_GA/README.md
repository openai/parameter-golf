# Record: 11L XSA-all + LeakyReLU(0.5)^2 + VR + GA (val_bpb=1.1164)

**val_bpb = 1.1164** (single seed, pending 3-seed validation) | **~15.94 MB** | No TTT

## Summary

Non-TTT submission combining XSA on all 11 layers with LeakyReLU(0.5)^2 activation, Value Residual, and Gated Attention. Achieves 1.1164 BPB on single GPU (7500 steps), within 0.001 of the non-TTT SOTA (1.1154, PR #609).

**Requesting compute grant for 8xH100 3-seed validation.**

## Single-GPU Results (1xH100 NVL 96GB, 7500 steps)

| Metric | Value |
|--------|-------|
| Raw val_bpb | 1.1338 |
| Int6 roundtrip val_bpb | 1.1401 |
| **Int6 sliding val_bpb (s=64)** | **1.1164** |
| Artifact size | 15,941,860 bytes |
| Step avg | 1064 ms |
| Quantization gap | 0.006 BPB |

## Architecture

- 11L, 512d, 8H/4KV (GQA), MLP 3x
- **LeakyReLU(0.5)^2**: `leaky_relu(x, 0.5).square()` replaces ReLU^2. Preserves negative gradient flow, -0.003 BPB vs ReLU^2 at zero overhead.
- **XSA on all 11 layers**: Exclusive Self-Attention removes self-position bias in all layers (not just last 4). -0.006 BPB vs XSA-last-4.
- **Value Residual (VR)**: Layer 0 V output mixed into subsequent layers via learned sigmoid gates. -0.002 BPB.
- **Gated Attention (GA)**: Per-head sigmoid gates on attention output.
- SmearGate + OrthoInit, BigramHash(4096), U-Net skip connections
- Partial RoPE (16/64 dims), LN Scale, EMA(0.997)
- Int6 per-row quantization + zstd-21 compression

## Key Techniques vs Baseline

| Technique | BPB Impact | Source |
|-----------|-----------|--------|
| LeakyReLU(0.5)^2 | -0.003 | PR #493, #518 |
| XSA-all (11 layers) | -0.006 | PR #609 |
| Value Residual + Gated Attention | -0.002 | PR #413, #487 |
| Warmdown 3500 (vs 3000) | ~-0.001 | Hyperparameter tuning |

## Training Config

```bash
ITERATIONS=7500 WARMDOWN_ITERS=3500 MAX_WALLCLOCK_SECONDS=0
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
XSA_LAST_N=11 LEAKY_RELU=1 TTT_ENABLED=0 CANON_LAST_N=0 SWA_ENABLED=0
```

## Reproduction

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
SEED=1337 XSA_LAST_N=11 LEAKY_RELU=1 WARMDOWN_ITERS=3500 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  TTT_ENABLED=0 CANON_LAST_N=0 SWA_ENABLED=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Negative Results

Techniques tested on this stack that did not help:

| Technique | Result | Why |
|-----------|--------|-----|
| Full GPTQ (Hessian-aware) | +0.029 BPB | Requires Parameter Banking (3D weight tensors) |
| Tight SWA (EMA+SWA stacked) | +0.005 BPB | Doubles quantization gap (0.006 to 0.012) |
| Remove VR+GA | +0.002 BPB | VR+GA still beneficial even with XSA-all |
| MATRIX_LR=0.04 | +0.018 BPB | 4x worse quantization gap |
| Canon AC (last 5 layers) | +0.017 BPB | Conflicts with VR+GA |
| Star-ReLU MLP=1536 | +0.010 BPB | Worse than ReLU^2 at same width |

## Credits

- Base architecture: modded-nanogpt, PR #315 (jfprincz)
- XSA-all: PR #609
- LeakyReLU^2: PR #493 (parinzee), PR #518 (sofiabod)
- Value Residual: PR #413 (arXiv:2410.17897)
- Gated Attention: NeurIPS 2025, arXiv:2505.06708
