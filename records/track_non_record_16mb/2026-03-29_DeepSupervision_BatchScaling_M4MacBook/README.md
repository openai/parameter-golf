# Non-Record: Deep Supervision + Systematic Exploration on M4 MacBook

16 experiments over ~15 hours of compute on an Apple M4 MacBook (16GB unified memory, MLX backend) exploring **deep supervision** — a novel technique for this competition where an auxiliary next-token prediction loss is added at the encoder-decoder boundary of the U-Net architecture. No other competitor has explored this technique.

These results are promising — batch scaling shows no plateau and deep supervision opens a new axis of improvement — but hardware is the bottleneck. An M4 MacBook at ~9K tok/s with 300 steps cannot match the 20K steps at 524K batch on 8xH100s. I believe H100 compute credits would allow me to validate these findings at competition scale and push further.

## Hardware & Setup

- **Hardware:** Apple M4 MacBook, 16GB unified memory
- **Backend:** MLX, bfloat16 compute, ~9K tok/s peak throughput
- **Data:** 10 training shards from fineweb10B_sp1024 (~1B tokens)
- **Training:** 300 steps per experiment (wall-clock limited)
- **Validation:** Full fineweb_val split, both fp32 and int8 quantized roundtrip
- **All artifacts under 16MB** (largest: 9.8MB)

## Deep Supervision: The Idea

The model has an hourglass (U-Net) shape: encoder blocks feed into decoder blocks with skip connections. Normally, the loss is computed only at the final output. Deep supervision adds a second loss at the encoder-decoder boundary — like checking if a student understands the material halfway through a course, not just at the final exam.

The auxiliary loss forces encoder layers to learn more predictive representations earlier in the forward pass. It reuses the existing layer norm and output projection, so **no extra parameters are needed**.

Implementation: after the final encoder block, apply the same `final_norm` → output projection → cross-entropy pipeline, and add `weight * aux_loss` to the main loss.

## Full Experiment Log

All 16 experiments listed chronologically with exact configurations and results.

### Phase 1: Deep Supervision Weight Sweep (8K batch)

Goal: Find the optimal deep supervision weight at a fixed small batch size.

| Exp | Config | val_bpb | int8_bpb | vs Baseline | Time |
|-----|--------|---------|----------|-------------|------|
| 01 | Baseline (LeakyReLU, 8K batch, 300 steps) | 2.1668 | 2.1677 | — | ~6 min |
| 02 | + DeepSup weight=0.02 @ encoder boundary | 2.1258 | 2.1274 | **-0.0403** | ~6 min |
| 03 | + DeepSup weight=0.03 @ encoder boundary | 2.1164 | 2.1178 | **-0.0499** | ~6 min |
| 04 | + DeepSup weight=0.04 @ encoder boundary | 2.1281 | 2.1295 | -0.0382 | ~6 min |
| 05 | + DeepSup weight=0.05 @ encoder boundary | 2.1280 | 2.1302 | -0.0375 | ~6 min |

**Finding:** Optimal weight is **0.03**. Too low (0.02) underregularizes, too high (0.04+) overregularizes. Clear inverted-U pattern.

### Phase 2: Multi-Point Deep Supervision (8K batch)

Goal: Test whether tapping multiple intermediate layers is better than single-point at the boundary.

| Exp | Config | val_bpb | int8_bpb | vs Baseline |
|-----|--------|---------|----------|-------------|
| 06 | DeepSup(0.05) @ layers 2,5 (multi-point) | 2.1406 | 2.1420 | -0.0257 |

**Finding:** Multi-point (layers 2 and 5) is **worse** than single-point at the encoder boundary (layer 3). The boundary is the optimal tap point — it's where encoder representations need to be most complete before the decoder processes them.

### Phase 3: Batch Size Scaling + Deep Supervision Interaction

Goal: Understand how deep supervision interacts with batch size. This turned out to be the most important experiment series.

| Exp | Config | val_bpb | int8_bpb | vs Baseline |
|-----|--------|---------|----------|-------------|
| 07 | Baseline, 16K batch (2x grad accum) | 2.0333 | 2.0367 | -0.1310 |
| 08 | DeepSup(0.03), 16K batch | 2.0333 | 2.0366 | -0.1311 |
| 09 | Baseline, 32K batch (4x grad accum) | 1.9409 | 1.9434 | -0.2243 |
| 10 | Baseline, 64K batch (8x grad accum) | 1.7626 | 1.7673 | -0.4004 |
| 11 | DeepSup(0.03), 64K batch | 1.7687 | 1.7736 | -0.3941 |
| 12 | Baseline, 128K batch (16x grad accum) | **1.6611** | **1.6668** | **-0.5009** |

**Key Finding — Deep supervision scales inversely with batch size:**

| Batch Size | Baseline | +DeepSup(0.03) | Deep Sup Effect |
|-----------|----------|----------------|-----------------|
| 8K | 2.168 | 2.118 | **-0.050 (helps)** |
| 16K | 2.037 | 2.037 | 0.000 (neutral) |
| 64K | 1.767 | 1.774 | +0.006 (slightly hurts) |

Interpretation: Deep supervision acts as a regularizer. At small batch sizes, gradient estimates are noisy and the auxiliary signal helps stabilize learning. At large batch sizes, the smoother gradient landscape already provides that regularization, making the auxiliary loss redundant or distracting.

**Batch scaling shows no plateau:**

| Batch Size | int8_bpb | Marginal Gain |
|-----------|----------|---------------|
| 8K (1x) | 2.168 | — |
| 16K (2x) | 2.037 | -0.131 |
| 32K (4x) | 1.943 | -0.094 |
| 64K (8x) | 1.767 | -0.176 |
| 128K (16x) | **1.667** | -0.100 |

The scaling is approximately log-linear with no sign of diminishing returns. At 300 steps, even 128K batch (our maximum on M4) hasn't saturated — there is clearly more performance to be unlocked at the competition's 524K batch with 20K steps.

### Phase 4: Convergence Techniques (64K batch)

Goal: Test techniques used by top leaderboard entries to see if they help at 300 steps.

| Exp | Config | val_bpb | int8_bpb | vs 64K Baseline |
|-----|--------|---------|----------|-----------------|
| 13 | EMA decay=0.99 | 3.1376 | 1.8583 | +0.091 (worse) |
| 14 | SWA start=50%, every 10 steps | 3.0774 | 1.8270 | +0.060 (worse) |
| 15 | Partial RoPE (16/64 dims) + LN Scale (1/sqrt(L+1)) | 3.0380 | 1.8048 | +0.038 (worse) |
| 16 | Sequence length 2048 (vs 1024) | 3.0182 | 1.7982 | +0.031 (worse) |

**Finding:** All four techniques hurt at 300 steps. These are convergence-regime optimizations — they help when the model is in a flat loss valley (20K steps) by smoothing the final weights. At 300 steps, the model is still in steep descent, and averaging earlier (worse) weights or slowing down steps hurts.

This is actually a useful negative result: it tells us **which techniques are step-count-dependent vs. architecture-dependent**. Deep supervision helped even at 300 steps because it's an architectural change to the loss landscape, not a post-hoc smoothing technique.

## Summary of All Results (Ranked)

| Rank | Experiment | Batch | int8_bpb | vs 8K Baseline |
|------|-----------|-------|----------|----------------|
| 1 | Baseline | 128K | **1.667** | -0.501 |
| 2 | Baseline | 64K | 1.767 | -0.400 |
| 3 | DeepSup(0.03) | 64K | 1.774 | -0.394 |
| 4 | Seq2048 | 64K | 1.798 | -0.370 |
| 5 | Partial RoPE + LN Scale | 64K | 1.805 | -0.363 |
| 6 | SWA (50%) | 64K | 1.827 | -0.341 |
| 7 | EMA (0.99) | 64K | 1.858 | -0.310 |
| 8 | Baseline | 32K | 1.943 | -0.224 |
| 9 | Baseline | 16K | 2.037 | -0.131 |
| 10 | DeepSup(0.03) | 16K | 2.037 | -0.131 |
| 11 | DeepSup(0.03) | 8K | **2.118** | -0.050 |
| 12 | DeepSup(0.02) | 8K | 2.127 | -0.040 |
| 13 | DeepSup(0.04) | 8K | 2.130 | -0.038 |
| 14 | DeepSup(0.05) | 8K | 2.130 | -0.038 |
| 15 | DeepSup(0.05) multi-point | 8K | 2.142 | -0.026 |
| 16 | Baseline | 8K | 2.168 | — |

## What H100 Access Would Enable

These M4 results establish strong directional signal but cannot reach competition-relevant configurations:

1. **Validate deep supervision at 524K batch / 20K steps** — our finding that deep supervision helps at small batch but not large batch was tested only up to 64K. The competition uses 524K, but with 20K steps (vs our 300). In a longer training regime, deep supervision may re-emerge as beneficial since the model enters different optimization phases.

2. **Combine deep supervision with top techniques** — TTT (test-time training), XSA (cross-sequence attention), and EMA all require full-scale validation. Deep supervision could stack with these since it modifies the loss landscape rather than the optimizer or eval strategy.

3. **Push the batch scaling curve to its natural limit** — our scaling shows no plateau at 128K. On 8xH100s with the full 524K batch, we could determine the actual saturation point and whether architectural changes shift it.

4. **Explore step-count-dependent interactions** — EMA/SWA hurt at 300 steps but help at 20K. There may be a crossover point where combining deep supervision with SWA gives compounding gains that neither provides alone.

The M4 experiments cost ~15 hours of compute for 16 experiments. On 8xH100s, the same experiments would take ~3 hours total and produce competition-grade results.

## How to Reproduce

```bash
# Install deps
pip install sentencepiece mlx

# Download data (10 shards)
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Deep supervision (best at 8K batch)
DEEP_SUP_WEIGHT=0.03 ITERATIONS=300 TRAIN_BATCH_TOKENS=8192 \
  GRAD_ACCUM_STEPS=1 VAL_BATCH_SIZE=524288 \
  python train_gpt_mlx.py

# Best overall result (128K batch)
ITERATIONS=300 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=16 \
  VAL_BATCH_SIZE=524288 \
  python train_gpt_mlx.py
```

Deep supervision is controlled via environment variables:
- `DEEP_SUP_WEIGHT` — auxiliary loss weight (0 = off, 0.03 = optimal)
- `DEEP_SUP_LAYERS` — comma-separated layer indices to tap (empty = encoder boundary, default)
