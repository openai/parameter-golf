# 11L XSA4 + Tight SWA + FA3 + Two-Phase TTT

## Summary

Built on PR #374 (unnir's 11L XSA4 + Tight SWA base) with FA3 Hopper attention and a novel two-phase test-time training approach:

- **FA3 Hopper:** 84.65ms/step (vs 96ms with SDPA/FA2), enabling ~7,000 training steps in 600s.
- **Phase 1 — Norm-Only Recalibration (50 epochs, Adam lr=0.01):** Only unfreeze LayerNorm weights, scales, and final_norm (~22K params). Recalibrates activation distributions damaged by int6 quantization. Acts as post-quantization calibration via gradient descent.
- **Phase 2 — Selective-Freeze Block Adaptation (10 epochs, SGD lr=0.005, momentum=0.9):** Unfreeze last 3 transformer blocks + all norms + scales + lm_head (~7.6M params). Adapts representations on the recalibrated foundation while preserving SWA-averaged weights in the first 8 blocks.

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
- Magnitude pruning 2%

## Setup

```bash
pip install zstandard
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
```

## Results

```
seed=1337: val_bpb=1.1222, artifact=15,758,953 bytes
seed=42:   val_bpb=1.1230, artifact=15,798,468 bytes
seed=2024: val_bpb=1.1228, artifact=15,689,654 bytes

3-seed mean: val_bpb=1.1227
```

All runs: 84.65ms/step, ~7000 steps, 600s training + ~500s eval (TTT + sliding window).

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
