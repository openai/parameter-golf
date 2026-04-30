# PR 692 — CROWN-Q + Full GPTQ + SWA/EMA Blend

**Author:** not stated in README (PR was closed for docs per task description)
**Claimed BPB:** 1.1186 (3-seed mean, std 0.0006)
**Artifact size:** ~15.94 MB (15,945,134 / 15,947,742 / 15,938,790)
**Seeds:** 1337, 42, 7

## Files retrieved
- `records__track_10min_16mb__2026-03-25_CROWNQ_GPTQ_SlidingWindow__README.md`
- `records__track_10min_16mb__2026-03-25_CROWNQ_GPTQ_SlidingWindow__submission.json`
- `records__track_10min_16mb__2026-03-25_CROWNQ_GPTQ_SlidingWindow__train_gpt.py`

## Environment variables (from README key env vars)
CROWNQ_LAMBDA=0.01 CROWNQ_WARMDOWN_ONLY=1 LATE_QAT_THRESHOLD=0.15 MAX_WALLCLOCK_SECONDS=585 WARMDOWN_ITERS=4000

## Claimed changes (from README, verbatim)
> CROWN-Q: Curvature-weighted quantization variance penalty applied during warmdown. Encourages weights to settle in flat minima where int6 quantization causes less damage. Penalty: `lambda * sum(h_j * delta_j^2 / 12)` where `h_j = w^2` (curvature proxy) and `delta_j = row_max / 15` (quantization step size).

> Full Cholesky GPTQ: Hessian-aware quantization with act-order column permutation, block_size=128, 256-sample calibration from training data. All within 585s training budget.

> SWA/EMA 50/50 blend: Stochastic Weight Averaging (every 50 steps during warmdown) blended 50/50 with EMA (decay=0.997).

> Architecture: 11L, 512d, GQA 8H/4KV, MLP 3x LeakyReLU(0.5)^2, XSA on all 11 layers, VRL, BigramHash 3072, partial RoPE 16/64.

> CROWN-Q (Curvature-Regularized Optimization for Weight Noise Quantization): For each weight matrix, compute the per-row quantization step size `delta = row_max / 15` (int6). Compute quantization variance `delta^2 / 12` (uniform rounding noise). Weight by curvature proxy `h = w^2`. Penalty encourages the optimizer to reduce weights in directions where quantization noise is most damaging. Applied only during warmdown when QAT is active. Zero eval-time cost.
