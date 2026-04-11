# 10L Int5-MLP + Aggressive Warmdown (WD=20000)

## Hypothesis

Based on systematic ablation testing on RTX 4090 (10 configs, 3 seeds each), we found that setting `warmdown_iters=20000` — making the entire training run a decay phase — significantly improves post-quantization quality compared to the standard `warmdown_iters=3000`.

Key finding from 4090 experiments:
- Post-quant penalty drops from 0.014 bpb (WD=1200 default) to 0.005 bpb (WD=20000)
- The entire LR schedule becomes a smooth cosine decay from max to 0
- This produces smoother weight distributions that quantize better under Int5/Int6
- When warmdown already produces smooth weights, more training steps > better per-step gradient quality

## Changes from current #1 (2026-03-20_10L_Int5MLP_MuonWD04_SWA50)

Only one change:
- `warmdown_iters`: 3000 → **20000**

Everything else is identical: 10 layers, Int5 MLP quantization, BigramHash 10240, SWA every 50 steps (start frac 0.4), MuonWD 0.04, sliding window eval stride 64.

## Why this might work

On 8xH100 the run gets ~7,400 steps. With `warmdown_iters=20000`, warmdown starts at step max(20000-20000, 0) = step 0. The LR decays smoothly from the start, producing weights that:
1. Have smaller magnitude variance → better Int5 quantization
2. Converge more smoothly → better SWA averaging
3. Reduce the quantization penalty (the gap between pre-quant and post-quant bpb)

## Local validation (RTX 4090, single GPU)

| Config | val_bpb (int8+zlib) | Steps | Notes |
|--------|-------------------|-------|-------|
| Baseline (WD=1200) | 1.2244 | 13,450 | Default |
| WD=3000 (current #1) | — | — | Leaderboard: 1.1428 on 8xH100 |
| **WD=20000 (ours)** | **1.1574** | 7,199 | Best 4090 result, sliding window |

The 4090 result (1.1574) with aggressive warmdown beat all other configs we tested despite having fewer steps, because the quantization penalty was dramatically lower.
