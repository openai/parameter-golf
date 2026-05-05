# Non-Record Submission: MASA v3 — Low-Rank Shared Attention + SwiGLU

## Summary

MASA (Matrix Atom Sharing Attention) applied to a tiny transformer.
Instead of each layer having unique Q/K/V/O matrices, all 11 layers share
10 low-rank base matrices. Each layer only learns 4 mix coefficient vectors.

## Results

| Metric | Value |
|--------|-------|
| val_bpb (pre-quant) | 1.3547 |
| val_bpb (post-quant) | 1.3579 |
| Quantization degradation | 0.003 BPB |
| Compressed size | 20.98MB |
| Model params | ~24M |
| Steps | 20,000 |
| Hardware | RTX 4050 6GB |
| Training time | ~9 hours |

Note: compressed size exceeds 16MB limit — submitted as non-record track.
Next run targets 16MB by reducing MASA_NUM_BASES from 10 → 6.

## Progression

| Run | Steps | val_bpb | Notes |
|-----|-------|---------|-------|
| v1 (14k steps) | 14,000 | 1.4442 | no warmdown fix |
| v1 (20k steps) | 20,000 | 1.3703 | warmdown fix |
| v3 (20k steps) | 20,000 | 1.3579 | low-rank bases, 11L, MLP 3x |

## Architecture

```
NUM_LAYERS=11
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=8
MLP_MULT=3           (SwiGLU, hidden=341)
MASA_NUM_BASES=10
MASA_RANK=128        (low-rank: base = A[dim,128] @ B[128,dim])
TRAIN_BATCH_TOKENS=32768
TRAIN_SEQ_LEN=512
ITERATIONS=20000
```

## Key Changes vs Baseline

1. **MASA attention** — all layers share 10 low-rank base matrices
   - each layer: 4 mix coefficients (40 numbers total per layer)
   - base_i = A_i @ B_i where A:[dim,128], B:[128,dim]
   - saves ~9x attention params vs full-rank per-layer matrices

2. **SwiGLU MLP** — replaces ReLU²
   - output = proj(silu(gate(x)) * up(x))
   - hidden = int(mlp_mult * dim * 2/3)

3. **Warmdown fix** — last 20% of iterations for LR decay
   - original code decayed LR from step 1 (bug)
   - fix: warmdown_start = int(iterations * 0.8)

4. **Low-rank bases** — 8x cheaper than full-rank
   - allows 10 bases for same cost as 1-2 full-rank bases
   - maximizes effective rank per byte

## What Didn't Work / Next Steps

- compressed size 20.98MB → need MASA_NUM_BASES=6 for 16MB
- sliding window eval not yet implemented
- int6 QAT not yet implemented
- EMA weights not yet implemented

Projected with 6 bases + all improvements: ~1.28 BPB
