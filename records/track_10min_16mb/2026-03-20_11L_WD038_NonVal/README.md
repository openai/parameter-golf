# 11L WD0.038 Non-Val + Int6 + Sliding Window

**mean_val_bpb: 1.1472** (3 seeds, sliding window stride=64, post int6+zstd quantization)

## Summary

11-layer GPT with GQA, trained on standard training shards (no val-only training). Uses decoupled weight decay on Muon optimizer to reduce quantization gap and fit 11 layers under 16MB. Beats previous SOTA (notapplica, 1.1748) by 0.0275 BPB with p < 0.001.

## Key Innovations

1. **Decoupled weight decay on Muon optimizer (0.038)** — Reduces weight magnitudes, dramatically shrinking the quantization gap. Also enables fitting 11 layers under 16MB. Slightly higher WD than val-only variant because non-val models have larger weight magnitudes.

2. **11 layers instead of 9/10** — More capacity improves generalization. With WD=0.038, artifact fits under 16MB.

3. **Int6 per-row quantization** — 31 quantization levels per row for MLP and attention weights. Embeddings kept in fp16.

4. **Sliding window evaluation (stride=64)** — Each token scored with maximum context.

5. **Higher learning rate (0.025)** — Non-val training benefits from slightly higher LR compared to val-only (0.02).

6. **Tuned Muon optimizer** — Momentum=0.99 with warmup from 0.92 over 1500 steps. Warmdown=3000 steps.

## Configuration

```bash
TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MIXED_KEEP_FLOAT_PATTERNS=tok_emb
MATRIX_LR=0.025
SCALAR_LR=0.025
TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
EVAL_STRIDE=64
WEIGHT_DECAY=0.038
NUM_LAYERS=11
NUM_KV_HEADS=4
MODEL_DIM=512
NUM_HEADS=8
MLP_MULT=3.0
VOCAB_SIZE=1024
MAX_WALLCLOCK_SECONDS=600
```

## Multi-Seed Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact Size |
|------|----------|---------|-------|---------|---------------|
| 1337 | 1.93869482 | 1.14820731 | 6986 | 85.85 | 15,696,856 |
| 42   | 1.93202108 | 1.14425474 | 8202 | 73.16 | 15,905,331 |
| 7    | 1.94044336 | 1.14924290 | 6961 | 86.15 | 15,610,740 |
| **Mean** | **1.93705309** | **1.14723498** | — | — | — |

**p-value vs previous SOTA (notapplica, 1.1748): 0.0008** (Welch's t-test, one-sided)
**Improvement: 0.0275 BPB** (threshold: 0.005)

## Architecture

- Layout: VOCAB_SIZE=1024, NUM_LAYERS=11, MODEL_DIM=512, NUM_HEADS=8, NUM_KV_HEADS=4, MLP_MULT=3.0
- Tied embeddings with separate LR (0.035)
- GQA with 4 KV heads
- 26,501,720 parameters

## Included Files

- `train_gpt.py` — Full training script with all optimizations
- `train_seed1337.log` — Training log (seed 1337)
- `train_seed42.log` — Training log (seed 42)
- `train_seed7.log` — Training log (seed 7)
- `submission.json` — Leaderboard metadata with multi-seed results

## Hardware

All runs on Modal 8x NVIDIA H100 80GB HBM3 SXM.
