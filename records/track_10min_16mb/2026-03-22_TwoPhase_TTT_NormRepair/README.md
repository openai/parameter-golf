# 11L XSA4 + Tight SWA + FA3 + Two-Phase TTT

## Summary

Built on PR #374 (unnir's 11L XSA4 + Tight SWA base) with FA3 Hopper attention and a novel two-phase test-time training approach:

- **FA3 Hopper:** 84.65ms/step (vs 96ms with SDPA/FA2), enabling 6,939 training steps in 600s.
- **Phase 1 — Norm-Only Recalibration (100 epochs, Adam lr=0.01):** Only unfreeze LayerNorm weights, scales, and final_norm (~22K params). Recalibrates activation distributions damaged by int6 quantization. Acts as post-quantization calibration via gradient descent.
- **Phase 2 — Selective-Freeze Block Adaptation (25 epochs, SGD lr=0.005, momentum=0.9):** Unfreeze last 3 transformer blocks + all norms + scales + lm_head (~7.6M params). Adapts representations on the recalibrated foundation while preserving SWA-averaged weights in the first 8 blocks.

Key insight: the two phases target different error sources (quantization artifacts vs. distribution mismatch) and are additive.

## Architecture

- 11 transformer layers, dim=512, 8 heads, 4 KV heads (GQA)
- 3x MLP with relu² + SmearGate + OrthoInit
- XSA on last 4 layers (Exclusive Self-Attention)
- Partial RoPE (16/64 head dims)
- LN Scale (1/sqrt(layer+1))
- BigramHash(2048), bigram_dim=128
- Shared VE128 (value embeddings shared across layers 9-10)
- Tight SWA (scale < 0.2), Late QAT (final 4%)
- FA3 Hopper attention (flash_attn_interface)
- Int6 quantization + zstd-22 compression
- Magnitude pruning 1%

## Setup

```bash
pip install zstandard
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
```

## Results

```
seed=1337: val_bpb=1.1216, artifact=15,704,756 bytes
  training: 84.65ms/step, 6939 steps, 600s wallclock
  post-SWA: val_bpb=1.1421
  TTT phase 1 (norm-only):       100 epochs, 22K params, Adam lr=0.01
  TTT phase 2 (selective-freeze): 25 epochs, 7.6M params, SGD lr=0.005
  TTT total time: 705s
  TTT improvement: -0.021 (1.1421 -> 1.1216)
```

Additional seeds in progress.

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
