# Non-Record: Deep Supervision on M4 MacBook

Ran 16 experiments on an Apple M4 MacBook (16GB, MLX) exploring **deep supervision** — adding an auxiliary next-token prediction loss at the encoder-decoder boundary of the U-Net architecture. No one else in the competition has tried this technique.

- **Hardware:** M4 MacBook, MLX backend, ~9K tok/s
- **Training:** 300 steps, 10 shards from fineweb10B_sp1024
- **Best result:** 1.667 int8 BPB (128K batch)

## Deep Supervision: What and Why

The model has an hourglass (U-Net) shape: encoder blocks → decoder blocks with skip connections. Normally, loss is computed only at the final output. Deep supervision adds a second loss at the encoder-decoder boundary, forcing encoder layers to learn more predictive representations earlier.

**No extra parameters needed** — it reuses the existing layer norm and output projection.

## Results

### Deep supervision weight sweep (8K batch, 300 steps)

| Weight | int8 BPB | vs Baseline |
|--------|----------|-------------|
| 0 (baseline) | 2.168 | — |
| 0.02 | 2.127 | -0.040 |
| **0.03** | **2.118** | **-0.050** |
| 0.04 | 2.130 | -0.038 |
| 0.05 | 2.130 | -0.038 |

Optimal weight is **0.03**. Multi-point supervision (tapping layers 2 and 5) performed worse than single-point at the encoder boundary.

### Key finding: deep supervision vs batch size interaction

| Batch | Baseline | +DeepSup(0.03) | Effect |
|-------|----------|----------------|--------|
| 8K | 2.168 | 2.118 | **-0.050** |
| 16K | 2.037 | 2.037 | 0.000 |
| 64K | 1.767 | 1.774 | +0.006 |

Deep supervision helps at small batches but becomes neutral/harmful at large batches. The larger batch already provides the regularization effect. At competition scale (524K), deep supervision alone likely won't help — but it may be useful combined with smaller-batch strategies or for faster early convergence.

### Batch size scaling (no plateau through 128K)

| Batch | int8 BPB |
|-------|----------|
| 8K | 2.168 |
| 16K | 2.037 |
| 32K | 1.943 |
| 64K | 1.767 |
| 128K | 1.667 |

### Other techniques tested (64K batch, all negative at 300 steps)

| Technique | int8 BPB | vs Baseline |
|-----------|----------|-------------|
| EMA (0.99) | 1.858 | +0.091 |
| SWA (50%) | 1.827 | +0.060 |
| Partial RoPE (16d) + LN Scale | 1.805 | +0.038 |
| Seq length 2048 | 1.798 | +0.031 |

These techniques help at convergence (20K steps) but hurt at 300 steps where the model is still in steep descent.

## How to reproduce

```bash
# Deep supervision (best config at 8K batch)
DEEP_SUP_WEIGHT=0.03 ITERATIONS=300 TRAIN_BATCH_TOKENS=8192 \
  python train_gpt_mlx.py

# Batch scaling (best overall result)
ITERATIONS=300 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=16 \
  python train_gpt_mlx.py
```

Deep supervision is implemented via `DEEP_SUP_WEIGHT` and `DEEP_SUP_LAYERS` environment variables in `train_gpt_mlx.py`. Setting `DEEP_SUP_LAYERS=""` (default) taps the last encoder layer.
