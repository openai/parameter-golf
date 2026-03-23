# Experiment Tracker

**Target**: Beat SOTA 1.1428 BPB (thwu1, 10L Int5-MLP + BigramHash(10240) + SWA + WD=0.04)

## How to run

```bash
# Smoke test (2 min, 4 GPUs):
./run_experiment.sh idea01_byte_weighted_loss 42 120

# Full run (10 min, 8 GPUs):
./run_experiment.sh idea01_byte_weighted_loss 42 600

# All smoke tests:
./run_all_smoke.sh

# Full 3-seed eval:
./run_full_3seed.sh idea01_byte_weighted_loss

# Prepare submission:
./prepare_submission.sh idea01_byte_weighted_loss "ByteWeightedLoss_10L" "author" "github"
```

## Experiment List

| # | Experiment | Category | Change | Risk | Status | BPB | Notes |
|---|-----------|----------|--------|------|--------|-----|-------|
| 00 | baseline | - | SOTA copy (10L Int5-MLP) | - | pending | - | Reference |
| 01 | byte_weighted_loss | Loss | Weight CE by token byte count | low | pending | - | Direct BPB alignment |
| 02 | factorized_bigram | Embed | Collision-free low-rank bigram (rank=48) | med | pending | - | Replaces hash |
| 03 | entropy_reg | Reg | Penalize quantized weight entropy | med | pending | - | Better compression |
| 04 | conditional_resid | Arch | Data-dependent resid_mix gating | med | pending | - | Per-token depth |
| 05 | embed_factorize | Embed | Factorize 1024x512 → 1024x64 + 64x512 | med | pending | - | Free 800KB |
| 06 | adaptive_ns | Optim | Time-dependent NS coefficients | high | pending | - | Ortho→compression |
| 07 | bigram16k | Hyper | 16384 bigram buckets (vs 10240) | low | pending | - | More buckets |
| 08 | 11th_layer | Arch | 11 layers + int4 MLP | high | pending | - | Depth vs quant |
| 09 | trigram | Embed | Bigram + Trigram hash combined | med | pending | - | 3-token context |
| 10 | combined_best | - | Best ideas combined | - | pending | - | Final submission |
| 11 | swiglu | Arch | SwiGLU activation (replace relu²) | low | pending | - | Better activation |
| 12 | zloss | Loss | Z-loss on logits (1e-4) | low | pending | - | Prevents drift |
| 13 | label_smooth | Loss | Label smoothing eps=0.05 | low | pending | - | Standard reg |
| 14 | layerwise_lr | Optim | LR decay 0.9x per layer (deeper=higher) | low | pending | - | Better LR allocation |
| 15 | eval_stride32 | Eval | Sliding window stride=32 (vs 64) | low | pending | - | Better eval only |
| 16 | embed_scales | Arch | Separate input/output embed per-dim scales | low | pending | - | 1024 params |
| 17 | gqa2kv | Arch | 2 KV heads instead of 4 | med | pending | - | Save params |
| 18 | lzma | Compress | LZMA2 instead of zstd-22 | low | pending | - | Better ratio? |
| 19 | grad_central | Optim | Gradient centralization in Muon | low | pending | - | Free reg |
| 20 | ema | Train | EMA decay=0.999 instead of SWA | low | pending | - | Smoother avg |
| 21 | stochastic_depth | Reg | Drop layers 0→10% linearly | low | pending | - | Reg + speed |
| 22 | wsd_schedule | Optim | Warmup-Stable-Decay (5/75/20%) | med | pending | - | Better schedule |
| 23 | batch_warmup | Train | Batch 25%→100% over first 15% | low | pending | - | Noisy start |
| 24 | deepnorm_init | Init | Scale proj by (2N)^{-1/4} | low | pending | - | Deeper stability |
| 25 | sandwich_norm | Arch | Pre+Post RMSNorm on attn+MLP | low | pending | - | Value stability |
| 26 | agc | Optim | Adaptive gradient clipping (0.01) | low | pending | - | Better clipping |
| 27 | multi_token_pred | Loss | Auxiliary +2 token prediction (0.15 weight) | med | pending | - | Richer signal |
| 28 | diff_attention | Arch | Differential attention (attn1 - λ*attn2) | high | pending | - | Noise cancel |
| 29 | asymmetric_quant | Quant | Asymmetric int5/int6 (zero-point + scale) | med | pending | - | Better for skewed |
| 30 | groupwise_quant | Quant | 128-element group scales | med | pending | - | Finer quant |

## Priority Queue (run in this order)

### Tier 1 — Safest bets (run first)
1. baseline (reference point)
2. idea01_byte_weighted_loss
3. idea12_zloss
4. idea13_label_smooth
5. idea15_eval_stride32
6. idea16_embed_scales
7. idea11_swiglu

### Tier 2 — Medium confidence
8. idea14_layerwise_lr
9. idea20_ema
10. idea21_stochastic_depth
11. idea07_bigram16k
12. idea19_grad_central
13. idea22_wsd_schedule
14. idea25_sandwich_norm
15. idea24_deepnorm_init

### Tier 3 — Higher risk, higher potential
16. idea02_factorized_bigram
17. idea03_entropy_reg
18. idea04_conditional_resid
19. idea27_multi_token_pred
20. idea28_diff_attention
21. idea09_trigram
22. idea05_embed_factorize
23. idea06_adaptive_ns

### Tier 4 — Quantization experiments
24. idea29_asymmetric_quant
25. idea30_groupwise_quant
26. idea18_lzma
27. idea08_11th_layer

### Final
28. idea10_combined_best (combine winning ideas)
