# BitNet Ternary: 65M Parameters in 15.9MB

## Summary

Ternary weight quantization ({-1, 0, +1} at ~1.58 bits/weight) enables fitting **65M parameters** in a 15.9MB artifact — 3x the parameter count of standard int6 submissions (~22M params) at similar artifact size.

This explores a fundamentally different axis of optimization: instead of aggressive quantization of a small model, we train a much larger model with extreme quantization from the start.

## Approach

**Architecture:** 12 layers, 768 dim, 12 heads, 6 KV heads (GQA), 3x MLP expansion (hidden=2304), LeakyReLU(0.5)-squared, tied embeddings, U-Net skip connections.

**Ternary Training (STE):** Full-precision weights are maintained by the optimizer. The forward pass quantizes to ternary using Straight-Through Estimator:
- Per-row scale = mean(|w|) per row
- Threshold = 0.7 * scale
- Values above threshold -> +1, below -threshold -> -1, else -> 0
- Backward pass: gradients flow through as identity (STE)

**Activation schedule:** Full-precision training for the first 30% of wallclock, then ternary STE for the remaining 70%. This lets the model learn representations before adapting to the quantization constraint.

**Compression:** Ternary values {-1,0,1} stored as int8, compressed with zlib-9 (or zstd-22 when available for ~1MB savings). Since there are only 3 distinct values, compression achieves excellent ratios. Per-row fp16 scales for dequantization. Embedding kept as fp16.

**Evaluation:** Sliding window with stride=64 for improved BPB.

## Configuration

```
VOCAB_SIZE=1024, NUM_LAYERS=12, MODEL_DIM=768
NUM_HEADS=12, NUM_KV_HEADS=6, MLP_MULT=3
TRAIN_SEQ_LEN=1024, TRAIN_BATCH_TOKENS=524288
MATRIX_LR=0.02, SCALAR_LR=0.02, MUON_MOMENTUM=0.99
WARMDOWN_ITERS=3000, TERNARY_START_FRAC=0.3
```

## Run Command

```bash
RUN_ID=bitnet_final torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

- **Model params:** 64,529,040
- **Artifact size:** 15,878,267 bytes (code + ternary zlib-9)
- **Pre-quant val_bpb:** 1.2268
- **Post-quant val_bpb:** 1.2271
- **Quantization gap:** 0.0003 BPB
- **Sliding window val_bpb:** 1.1932 (stride=64)
- **Steps:** 5,026 / 20,000 (wallclock cap at 600s)
- **Step avg:** 119.38 ms
- **Peak memory:** 23,774 MiB
- **Training tokens:** ~2.6B (5,026 steps x 524,288 tokens/step)

## Key Findings

1. **Ternary training works at 65M scale** in a 10-minute budget — the loss recovers fully after the ternary transition.
2. **Quantization gap is near-zero** (~0.0003 BPB) because the model is trained with ternary STE.
3. **3x more parameters** fit in the same artifact budget compared to int6 quantization.
4. The ternary approach opens a new frontier for parameter-constrained language modeling that is orthogonal to the int6/GPTQ approaches used by other submissions.

## Files

- `README.md` — This file
- `submission.json` — Run metadata
- `train.log` — Full training log
- `train_gpt.py` — Training script (renamed from train_bitnet.py)
