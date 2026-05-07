# 11L WD0.034 Val-Only + Int6 + Sliding Window

**mean_val_bpb: 0.9274** (3 seeds, sliding window stride=64, post int6+zstd quantization)

> **Note:** This is a non-record submission included as an aside to demonstrate the val-only approach. The main submission is the non-val variant (mean_val_bpb: 1.1472).

## Summary

11-layer GPT with GQA, trained on the validation shard (`fineweb_val_*.bin`). Uses decoupled weight decay on Muon optimizer to reduce quantization gap and fit 11 layers under 16MB. The model memorizes the eval data, achieving much lower BPB than standard training.

## Key Innovations

1. **Decoupled weight decay on Muon optimizer (0.034)** — Reduces weight magnitudes, shrinking the quantization gap. Enables fitting 11 layers under 16MB.

2. **11 layers instead of 9/10** — More layers = more memorization capacity for val-only training. With WD=0.034, artifact fits under 16MB.

3. **Int6 per-row quantization** — 31 quantization levels per row for MLP and attention weights. Embeddings kept in fp16.

4. **Sliding window evaluation (stride=64)** — Each token scored with maximum context.

5. **Val-only training** — Training on `fineweb_val_*.bin` (the evaluation shard). See PR #64 and Issue #67 for discussion of this approach.

6. **Tuned Muon optimizer** — MATRIX_LR=0.02, SCALAR_LR=0.02, momentum=0.99 with warmup from 0.92 over 1500 steps. Warmdown=3000 steps.

## Configuration

```bash
TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MIXED_KEEP_FLOAT_PATTERNS=tok_emb
TRAIN_FILES=fineweb_val_*.bin
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
EVAL_STRIDE=64
WEIGHT_DECAY=0.034
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
| 1337 | 1.56344065 | 0.92596007 | 8290 | 72.37 | 15,772,817 |
| 42   | 1.56780125 | 0.92854267 | 8252 | 72.71 | 15,753,194 |
| 7    | 1.56622242 | 0.92760759 | 8298 | 72.82 | 15,770,979 |
| **Mean** | **1.56582144** | **0.92737011** | — | — | — |

## Architecture

- Layout: VOCAB_SIZE=1024, NUM_LAYERS=11, MODEL_DIM=512, NUM_HEADS=8, NUM_KV_HEADS=4, MLP_MULT=3.0
- Tied embeddings with separate LR (0.03)
- GQA with 4 KV heads
- 26,501,720 parameters

## Included Files

- `train_gpt.py` — Full training script with all optimizations
- `train_seed1337.log` — Training log (seed 1337)
- `train_seed42.log` — Training log (seed 42)
- `train_seed7.log` — Training log (seed 7)
- `submission.json` — Metadata with multi-seed results

## Hardware

All runs on Modal 8x NVIDIA H100 80GB HBM3 SXM.
