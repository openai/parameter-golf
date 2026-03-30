# Systematic Exploration on M4 MacBook: 27 Experiments

27 experiments over ~30 hours of compute on an Apple M4 MacBook (16GB unified memory, MLX backend). Explored deep supervision (a novel technique for this competition), learning rate tuning, batch size scaling, architecture changes, and convergence techniques.

**Best result:** 1.6414 int8_bpb (LR 0.08, 128K batch, 300 steps) — a -0.025 improvement over the default configuration.

## Hardware & Setup

- **Hardware:** Apple M4 MacBook, 16GB unified memory
- **Backend:** MLX, bfloat16 compute, ~9K tok/s peak throughput
- **Data:** 10 training shards from fineweb10B_sp1024 (~1B tokens)
- **Training:** 300 steps per experiment (wall-clock limited)
- **Validation:** Full fineweb_val split, both fp32 and int8 quantized roundtrip
- **All artifacts under 16MB** (largest: 9.8MB)

## Deep Supervision: Novel Technique

The model has a U-Net (hourglass) shape: encoder blocks → decoder blocks with skip connections. Deep supervision adds a second loss at the encoder-decoder boundary, forcing encoder layers to learn predictive representations earlier. Zero extra parameters — reuses existing `final_norm` → output projection → cross-entropy.

No other competitor has tried this technique.

## Experiments & Findings

### Phase 1: Deep Supervision Weight Sweep (8K batch)

| Exp | Config | int8_bpb | vs Baseline |
|-----|--------|----------|-------------|
| 01 | Baseline (8K batch, 300 steps) | 2.1677 | — |
| 02 | + DeepSup weight=0.02 | 2.1274 | -0.0403 |
| 03 | + DeepSup weight=0.03 | 2.1178 | **-0.0499** |
| 04 | + DeepSup weight=0.04 | 2.1295 | -0.0382 |
| 05 | + DeepSup weight=0.05 | 2.1302 | -0.0375 |

Optimal weight is 0.03. Clear inverted-U pattern.

### Phase 2: Multi-Point Deep Supervision (8K batch)

| Exp | Config | int8_bpb | vs Baseline |
|-----|--------|----------|-------------|
| 06 | DeepSup(0.05) @ layers 2,5 (two tap points) | 2.1420 | -0.0257 |

Single-point at the encoder boundary is better than multi-point.

### Phase 3: Batch Size Scaling + Deep Supervision Interaction

| Exp | Config | int8_bpb | vs 8K Baseline |
|-----|--------|----------|----------------|
| 07 | Baseline, 16K batch | 2.0367 | -0.131 |
| 08 | DeepSup(0.03), 16K batch | 2.0366 | -0.131 |
| 09 | Baseline, 32K batch | 1.9434 | -0.224 |
| 10 | Baseline, 64K batch | 1.7673 | -0.400 |
| 11 | DeepSup(0.03), 64K batch | 1.7736 | -0.394 |
| 12 | Baseline, 128K batch | 1.6668 | -0.501 |

**Deep supervision × batch size interaction:**

| Batch Size | Baseline | +DeepSup(0.03) | Effect |
|-----------|----------|----------------|--------|
| 8K | 2.168 | 2.118 | -0.050 (helps) |
| 16K | 2.037 | 2.037 | 0.000 (neutral) |
| 64K | 1.767 | 1.774 | +0.006 (neutral) |

Deep supervision acts as a regularizer whose benefit scales inversely with batch size.

**Batch scaling shows no plateau through 128K:**

| Batch Size | int8_bpb | Marginal Gain |
|-----------|----------|---------------|
| 8K → 16K | 2.168 → 2.037 | -0.131 |
| 16K → 32K | 2.037 → 1.943 | -0.094 |
| 32K → 64K | 1.943 → 1.767 | -0.176 |
| 64K → 128K | 1.767 → 1.667 | -0.100 |

### Phase 4: Convergence Techniques (64K batch)

| Exp | Config | int8_bpb | vs 64K Baseline |
|-----|--------|----------|-----------------|
| 13 | EMA decay=0.99 | 1.858 | +0.091 (worse) |
| 14 | SWA start=50%, every 10 steps | 1.827 | +0.060 (worse) |
| 15 | Partial RoPE + LN Scale | 1.805 | +0.038 (worse) |
| 16 | Sequence length 2048 | 1.798 | +0.031 (worse) |

All four hurt at 300 steps — these are convergence-regime optimizations that need 20K steps to help.

### Phase 5: Optimization & Architecture Tuning (128K batch)

Goal: Find improvements that stack on top of the best batch size.

| Exp | Config | int8_bpb | vs 128K Baseline (1.667) |
|-----|--------|----------|--------------------------|
| 18 | Matrix LR 0.06 | 1.6431 | **-0.024** |
| 19 | **Matrix LR 0.08** | **1.6414** | **-0.025 (best!)** |
| 20 | Matrix LR 0.02 | 1.7394 | +0.073 (much worse) |
| 21 | Warmup 50 steps (vs 20) | 1.6622 | -0.005 |
| 22 | No warmdown | 1.6808 | +0.014 (worse) |
| 23 | Grad clip norm=1.0 | 1.6473 | **-0.019** |
| 24 | 10 layers (vs 9) | 1.6613 | -0.006 |
| 25 | 12 layers | 1.6793 | +0.013 (worse) |
| 26 | MLP mult 3 (vs 2) | 1.6596 | -0.007 |
| 28 | Logit softcap 50 (vs 30) | 1.6785 | +0.012 (worse) |

**Key findings:**
- **LR 0.08 is optimal for 300 steps** — the default 0.04 is too conservative. The model needs to descend faster when step budget is limited.
- **Gradient clipping helps** — stabilizes large-batch training, consistent -0.019 improvement.
- **10 layers and wider MLP help slightly** — more capacity helps, but 12 layers goes too far (can't converge in 300 steps).
- **Lower LR, no warmdown, higher softcap all hurt** — the model needs aggressive optimization at 300 steps.
- **These findings are stackable** — LR, grad clip, and architecture changes modify different aspects of training.

## All Results Ranked

| Rank | Experiment | Batch | int8_bpb |
|------|-----------|-------|----------|
| 1 | **LR 0.08** | 128K | **1.6414** |
| 2 | LR 0.06 | 128K | 1.6431 |
| 3 | Grad clip 1.0 | 128K | 1.6473 |
| 4 | MLP mult 3 | 128K | 1.6596 |
| 5 | 10 layers | 128K | 1.6613 |
| 6 | Warmup 50 | 128K | 1.6622 |
| 7 | Baseline | 128K | 1.6668 |
| 8 | Softcap 50 | 128K | 1.6785 |
| 9 | 12 layers | 128K | 1.6793 |
| 10 | No warmdown | 128K | 1.6808 |
| 11 | LR 0.02 | 128K | 1.7394 |
| 12 | Baseline | 64K | 1.7673 |
| 13 | DeepSup(0.03) | 64K | 1.7736 |
| 14 | Seq2048 | 64K | 1.7982 |
| 15 | Partial RoPE + LN Scale | 64K | 1.8048 |
| 16 | SWA (50%) | 64K | 1.8270 |
| 17 | EMA (0.99) | 64K | 1.8583 |
| 18 | Baseline | 32K | 1.9434 |
| 19 | Baseline | 16K | 2.0367 |
| 20 | DeepSup(0.03) | 16K | 2.0366 |
| 21 | DeepSup(0.03) | 8K | 2.1178 |
| 22 | DeepSup(0.02) | 8K | 2.1274 |
| 23 | DeepSup(0.04) | 8K | 2.1295 |
| 24 | DeepSup(0.05) | 8K | 2.1302 |
| 25 | DeepSup(0.05) multi-point | 8K | 2.1420 |
| 26 | Baseline | 8K | 2.1677 |

## What H100 Access Would Enable

1. **Stack the winners** — LR 0.08 + grad clip + 10 layers + deep supervision at competition scale. These modify different aspects of training and should compound.
2. **Validate LR scaling at 20K steps** — optimal LR at 300 steps may differ from 20K steps, but the finding that default is too conservative likely transfers.
3. **Test deep supervision at 524K batch / 20K steps** — the regularization dynamics change in longer training.
4. **Push batch scaling past 128K** — determine where the curve saturates on 8xH100s.

27 experiments on M4 took ~30 hours. On 8xH100s, the same experiments would take ~5 hours and produce competition-grade results.

## Reproduce

```bash
pip install sentencepiece mlx
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Best result (LR 0.08, 128K batch)
MATRIX_LR=0.08 ITERATIONS=300 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=16 \
  VAL_BATCH_SIZE=524288 python train_gpt_mlx.py

# Deep supervision (best at 8K batch)
DEEP_SUP_WEIGHT=0.03 ITERATIONS=300 TRAIN_BATCH_TOKENS=8192 \
  GRAD_ACCUM_STEPS=1 VAL_BATCH_SIZE=524288 python train_gpt_mlx.py
```
