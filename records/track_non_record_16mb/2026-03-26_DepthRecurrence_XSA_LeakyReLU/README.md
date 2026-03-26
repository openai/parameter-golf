## Depth Recurrence + XSA + LeakyReLU²

Improves previous submission (1.2196 → 1.2065, -0.013 bpb) through three zero-parameter additions on top of depth recurrence.

val_bpb = 1.2065 (sliding window eval on int8+zstd22 roundtrip model, stride=256)
val_bpb = 1.2398 (standard int8+zstd22 roundtrip)

### Architecture

Same depth recurrence base as previous submission: 3 shared blocks repeated 4 times (12 effective layers), dim=832, 8 heads, 4 KV heads, MLP 2x, tied embeddings.

New additions (all zero extra parameters):
- **XSA (Exclusive Self-Attention)** on last 4 effective layers: removes self-value bias from attention output via GQA-aware projection subtraction. -0.010 bpb.
- **LeakyReLU(0.5)²** instead of relu²: preserves negative gradient flow while maintaining sparsity. Better gradient propagation through 4 recurrence passes. -0.004 bpb.
- **GPTQ-lite**: per-row best-of-5 clip percentiles during quantization (post-training, zero cost).
- **zstd-22** compression instead of zlib (saves ~1.85MB artifact space).
- **SWA** tuned to frac=0.4, every=50 steps.
- **Muon weight decay** 0.04.

Retained from previous submission:
- Cross-Repeat Skip (stateful recurrence with per-repeat learned scales)
- 2 Value Embedding tables
- Loop Embedding (per-effective-layer depth encoding)

17.14M params, 15.87MB artifact.

### Training

Same LR schedule as previous: MATRIX_LR=0.012, SCALAR_LR=0.012, TIED_EMBED_LR=0.015, GRAD_CLIP_NORM=0.3, WARMDOWN_ITERS=3000, TRAIN_SEQ_LEN=1024.

### Results (8xH100, 600s wallclock)

4300 steps, 140ms/step avg. Pre-quant 1.2373, roundtrip 1.2398, sliding window 1.2065. Artifact 15.87MB, quant degradation +0.003 bpb.

### Ablations (8xH100, 80 shards, all cumulative)

| Change | Sliding bpb | Delta |
|--------|-------------|-------|
| Baseline (previous submission repro) | 1.2213 | — |
| + XSA last 4 layers | 1.2110 | -0.0103 |
| + LeakyReLU(0.5)² | 1.2070 | -0.0040 |
| + GPTQ-lite + zstd-22 | 1.2065 | -0.0005 |

### Command

```
XSA_LAST_N=4 \
QUANT_LEVELS=127 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=256 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
