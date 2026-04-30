# Experiment 010 — Mixed-bit GPTQ (fallback if TTT angle fails)

**Date:** TBD (Day 2 afternoon, only if Experiment 002 doesn't show a winner)
**Hypothesis:** Per-layer bit allocation (e.g. int5 for low-sensitivity layers, int7 for first/last) at the same byte budget beats uniform int6.
**Baseline:** SOTA uniform int6 + int8 embeddings → 15.99 MB artifact, val_bpb 1.0810
**Cost:** ~$50 (full retrain × 4 configs)

This is the pivot if `scales` and friends don't beat `all`. Documented up front so we don't waste a day deciding.

## Why

The SOTA uses `MATRIX_BITS=6` for *every* matrix. Layer-wise sensitivity to quantization error is well-known to vary — first/last layers tend to need more bits; middle layers tolerate less. Mixed-bit allocation respecting Hessian sensitivity is standard in production quantization (SqueezeLLM, AWQ).

Already in the SOTA code:
- `gptq_calibration_batches=64` → Hessians are accumulated per-layer
- `gptq_quantize_weight(w, H, clip_sigmas, clip_range, block_size)` → takes a clip range → bits is exposed but currently called with a single global value

We need to plumb a per-layer bit assignment.

## Configs

| Tag | Layers | Matrix bits per layer | Embedding bits | Notes |
|-----|--------|------------------------|----------------|-------|
| `uniform_int6` | 0–10 | 6 | 8 | baseline (= SOTA) |
| `endpoints_int7` | 0,10 → 7; rest → 6 | 6/6/6/6/6/6/6/6/6/6/6 → mixed | 8 | first+last get extra precision |
| `int5_middle` | 0–2 → 6; 3–7 → 5; 8–10 → 6 | mixed | 8 | middle compresses |
| `hessian_proportional` | bits ∝ sqrt(trace(H)) per layer, normalized to ~6 avg | computed | 8 | principled |

## Implementation sketch (Day 2 to-do if pivoting)

1. Add `MATRIX_BITS_PER_LAYER` env var (comma-separated), e.g. `7,6,6,6,6,6,6,6,6,6,7`.
2. Rewrite `gptq_mixed_quantize(state_dict, hessians, h)` to accept per-layer bits, pass through to `gptq_quantize_weight`.
3. Adjust the brotli sizing: int5 layers compress better, int7 worse — verify the artifact stays under 16MB before running.

This is **~2 hours of work** vs the TTT angle's "minute". Only worth doing if TTT clearly fails.

## Decision

If running: pick the variant with lowest 3-seed mean. Combine with TTT (still using `all` filter as fallback).

If not running: this remains a documented design as a non-record submission angle.
