# 11L XSA4 + Tight SWA + Two-Phase TTT

## Summary

Built on PR #374 (unnir's 11L XSA4 + Tight SWA base) with a novel two-phase test-time training approach:

- **Phase 1 — Norm-Only Recalibration (100 epochs, Adam lr=0.01):** Only unfreeze LayerNorm weights, scales, and final_norm (~22K params). Recalibrates activation distributions damaged by int6 quantization. Acts as post-quantization calibration via gradient descent.
- **Phase 2 — Selective-Freeze Block Adaptation (15 epochs, SGD lr=0.003, momentum=0.95):** Unfreeze last 2 transformer blocks + all norms + scales + lm_head (~5.3M params). Adapts representations on the recalibrated foundation while preserving SWA-averaged weights in the first 9 blocks.

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
- Int6 quantization + zstd-22 compression
- Magnitude pruning 1%

## Results

```
seed=1337: val_bpb=1.1258, artifact=15,762,005 bytes
  training: 96.4ms/step, 6222 steps, 600s wallclock
  post-SWA: val_bpb=1.1447
  TTT phase 1 (norm-only):       100 epochs, 22K params, Adam lr=0.01
  TTT phase 2 (selective-freeze): 15 epochs, 5.3M params, SGD lr=0.003
  TTT total time: 752s
  TTT improvement: -0.019 (1.1447 -> 1.1258)
```

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
