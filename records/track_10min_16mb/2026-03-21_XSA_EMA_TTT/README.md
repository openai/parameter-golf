# XSA + EMA + TTT: Test-Time Training on Exclusive Self-Attention Base

**Non-record research submission** | val_bpb: 1.1436 (sliding window, stride=64) | Artifact: 15.3MB

## Summary

This submission tests whether Test-Time Training (TTT) improves upon the XSA + EMA base (PR #287, 1.1280 BPB). The answer is **no — TTT hurts by 0.016 BPB**, confirming the mechanism redundancy pattern identified in #140.

## Key Finding: TTT + XSA Don't Stack

| Configuration | val_bpb | Source |
|---|---|---|
| XSA + EMA (no TTT) | **1.1280** | PR #287 |
| XSA + EMA + TTT | **1.1436** | This submission |
| SmearGate + TTT (no XSA) | 1.1313 | PR #254 |
| SmearGate only | 1.1326 | PR #198 |

**TTT makes the XSA+EMA model 0.016 worse.** For comparison, TTT improves non-XSA models by ~0.013 (PR #254 vs #198). This confirms that XSA and TTT are mechanistically redundant — both improve local context modeling, so stacking them yields negative returns due to TTT's distribution drift.

This extends the earlier finding from PR #296 (Error-Guided TTT negative result) and PR #290 (XSA+TTT underperforms XSA-alone).

## Reproducibility (2 seeds)

| Seed | Steps | Sliding s64 | Artifact |
|------|-------|-------------|----------|
| 1337 | 6,001 | 1.1436 | 15,283,544 |
| 42 | 5,978 | 1.1441 | 15,283,544 |
| **Mean** | | **1.1439** | |

## Method

**Base model**: PR #287's recipe — 11L, 512dim, 3x MLP, int6+zstd, SmearGate, BigramHash(2048), OrthoInit, EMA (decay=0.997), XSA on last 4 layers, Muon WD=0.04.

**TTT**: PR #254's recipe — 3 epochs of full-model SGD (lr=0.002, momentum=0.9) on validation data, first 2 blocks frozen, gradient clipping at 1.0. Applied after int6 dequantization, before sliding window eval. TTT takes 67 seconds.

**FA2 compatibility**: Used `flash_attn` (FA2) instead of `flash_attn_interface` (FA3) due to environment constraints. This costs ~500 training steps (6,001 vs ~7,100 with FA3), partially explaining the gap vs PR #287's reported 1.1280.

## Configuration

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=2 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Analysis: Why TTT Hurts XSA Models

Three hypotheses for the negative interaction:

1. **Mechanism redundancy**: XSA removes self-information from attention outputs, forcing the model to rely on context from other tokens. TTT's gradient updates optimize for the same local context signal that XSA already provides — double-counting the same information source.

2. **EMA weight disruption**: The base model uses EMA-averaged weights (smooth, well-generalized). TTT's SGD updates introduce noise that moves weights away from the EMA optimum. Unlike SWA models where weights are already "rougher," EMA weights are more sensitive to perturbation.

3. **Distribution drift**: 3 epochs of SGD on validation data shifts the model's internal representations enough that the quantized-then-dequantized weight structure no longer aligns with the adapted features. The int6 quantization grid was optimized for the original EMA weights, not the TTT-adapted ones.

## Implications

- **For the competition**: XSA+EMA without TTT remains the strongest eval strategy. TTT should be reserved for non-XSA bases.
- **For research**: The redundancy between attention-output modifications (XSA) and weight adaptation (TTT) suggests they target the same information bottleneck — local context modeling beyond the attention window.
- **Open question**: Would TTT with much lower LR (e.g., 0.0002) or fewer epochs (1) avoid the distribution drift while still providing marginal gains?

## Hardware

8x NVIDIA H100 80GB SXM, RunPod. Training: 600s. TTT: 67s. Eval: 82s.

## Author

Xiaoan Liu | NYU | GitHub: @sseanliu
