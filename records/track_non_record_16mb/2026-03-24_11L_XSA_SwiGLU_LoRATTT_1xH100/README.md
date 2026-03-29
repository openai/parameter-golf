# Non-record: 11L XSA + SwiGLU + LoRA TTT (1xH100 PCIe)

**val_bpb: 1.1573** (LoRA TTT) | **15.02 MB** artifact | 1xH100 PCIe, ~80 min

## Key Techniques

1. **XSA (Cross-token Self-Attention)** on last 4 layers — removes self-value projection, forcing attention to contribute cross-position context. -0.005 BPB.
2. **SwiGLU 3x MLP** — gated activation `swish(gate(x)) * up(x)`. More parameter-efficient than ReLU². +0.004.
3. **SmearGate** — blends each token embedding with the previous token's embedding, giving bigram context at the embedding layer. Critical: +1.80 loss without it.
4. **U-Net skip connections** — encoder (L0-L4) saves skip outputs, decoder (L6-L10) adds them back. Ensures gradient flow through all 11 layers.
5. **Orthogonal initialization** — all weight matrices initialized orthogonally. Required for SmearGate to work.
6. **Muon optimizer with WD=0.04** — decoupled weight decay shrinks weights for better quantization + generalization.
7. **Stochastic Weight Averaging** — averages 15 checkpoints during warmdown for smoother quantized weights.
8. **Mixed quantization** — int5 (MLP) + int6 (attention) + int8 (embeddings) + zstd-22 compression. Fits in 15.02 MB.
9. **LoRA TTT** — per-document test-time training with rank-8 LoRA on Q and V projections. Score-then-train per 256-token chunk (legal: every token scored before being trained on). -0.034 BPB.

## Results

| Eval Method | val_loss | val_bpb | Delta |
|-------------|----------|---------|-------|
| Pre-quant (SWA) | 1.9800 | 1.1727 | — |
| Int8+zlib roundtrip | 1.9969 | 1.1826 | +0.010 |
| Mixed quant (int5/int6/int8+zstd) | 1.9913 | 1.1930 | +0.020 |
| **LoRA TTT (mixed quant)** | **1.9724** | **1.1573** | **-0.015** |

## Architecture

```
11L, 512d, 8H/4KV (GQA), SwiGLU 3x MLP
XSA on L7-L10, SmearGate, U-Net skips
OrthoInit, Muon WD=0.04, SWA (15 checkpoints)
Mixed quant: int5-MLP + int6-attn + int8-embed + zstd-22
LoRA TTT: rank-8, Q+V, LR=0.05, score-then-train, 256-token chunks
```

## Training Configuration

- **GPU**: 1xH100 PCIe (RunPod) — grad accumulation 8 steps to match 524K batch
- **Wallclock**: ~4850s (~80 min) — NOT a 10-min record submission
- **Batch**: 524,288 tokens/step (grad_accum=8 × seq_len=2048 × micro_batch=32)
- **Sequence length**: 2048
- **Warmdown**: 3000 iterations
- **Steps completed**: 7,926 / 20,000 (wallclock cap)

## Why Non-Record

This ran on 1xH100 PCIe for ~80 minutes (not 8xH100 in 10 min). The architecture and training are identical to what would run on 8xH100 — only the batch parallelism differs.

## Development Journey

18 experiments over 5 days, from val_bpb=3.10 (wrong batch size) to 1.1573:

| Experiment | val_bpb | What changed |
|-----------|---------|-------------|
| 1 (baseline) | 3.10 | Wrong batch size |
| 2 | 1.46 | Fixed batch to 65K |
| 6 | 1.312 | 1200s training, warmdown=600 |
| 10 | 1.283 | + SmearGate, OrthoInit, MLP 3x, WD |
| 13 | — | SwiGLU > ReLU² (+0.004) |
| 14-8x | 1.202 | 11 layers + SWA on 8xH100 |
| 15 | 1.187 | + seq_len=2048 |
| 17 | 1.183 | + XSA (last 4 layers) |
| + Quant | 1.191 | int5+int6+int8+zstd (15 MB) |
| **+ LoRA TTT** | **1.157** | Per-document adaptation at eval |

Total compute cost: ~$50 across all experiments.

## What Didn't Work

| Technique | Result | Why |
|-----------|--------|-----|
| Register token | +0.002 worse | Step overhead > marginal benefit |
| Layer looping + wider | +0.034 worse | Step time from wider dim |
| Data sampling (juncture) | +0.002 worse | Shard-level too coarse |
| Hard example mining | +0.040 worse | Destroys Muon weight geometry |
| Partial RoPE (16/64) | +0.015 worse | Head dim too small |
| EMA (replacing SWA) | +0.015 worse | Over-smoothed warmdown weights |
| BigramHash | 0.000 | SmearGate makes it redundant |
| SGD TTT | +0.018 worse | Modifying dequantized weights directly breaks them |

## Command

```bash
RUN_ID=exp17_xsa \
MAX_WALLCLOCK_SECONDS=4850 \
TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=3000 \
MUON_WD=0.04 \
NUM_LAYERS=11 \
TRAIN_SEQ_LEN=2048 \
MLP_MULT=3 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
python train_gpt.py
```

## Included Files

- `train_gpt.py` — full training + quantization + LoRA TTT eval script
- `train.log` — training log from 1xH100 run
- `submission.json` — metadata
