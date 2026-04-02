# Non-record: Annealed Muon 1.58-bit Ternary (8 KV Heads)

**val_bpb: TBD** | **~13.68 MB** (2xA100 baseline) | 8xH100 SXM, 600s target

## Summary

Training-time 1.58-bit ternary quantization using AnnealedBitLinear layers with Muon optimizer. During training, weights are stored as full-precision latents and quantized to ternary ({-1, 0, +1}) values on each forward pass via an annealing schedule that interpolates from soft to hard quantization. At serialization, ternary weights are packed 5 values per byte using base-3 encoding and compressed with zstd-22.

## 2xA100 Baseline Results

| Metric | Value |
|--------|-------|
| val_bpb | 1.3578 |
| Steps | 1,233 |
| ms/step | 481 |
| Model size | 13.68 MB |
| Compilation | ~149s |

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 10 (768d, 8 heads, 8 KV heads) |
| MLP | 4x (3072) |
| Unique blocks | 10 (no weight tying) |
| Quantization | AnnealedBitLinear 1.58-bit ternary |
| Packing | Base-3 uint8 (5 ternary values/byte) |
| Compression | zstd level 22 |
| Optimizer | Muon (NS steps=5) + AdamW (scalar params) |
| LR Schedule | Hold-Cosine (hold=0.70, min_lr=0.01) |
| Attention | Standard (no XSA) |
| RoPE | Full (base=10000) |
| SmearGate | Yes |
| U-Net skips | Yes |
| BigramHash | Disabled (0 buckets) |
| Compile | torch.compile fullgraph=True, optimize_ddp=False |

## Key Differences from SOTA Lineage

This submission explores a fundamentally different quantization approach from the main leaderboard lineage:

- **Training-time ternary quantization** vs post-training GPTQ int6
- **1.58 bits/weight** vs ~6 bits/weight (before compression)
- **No post-training calibration** -- quantization is baked into training via annealing
- **Base-3 packing** vs GPTQ + LZMA compression

The tradeoff is lower bits-per-weight (better raw compression) at the cost of reduced model capacity per parameter.

## Speed Optimizations Applied

1. Whole-model `torch.compile(fullgraph=True)` -- reduced step_avg from 543ms to 503ms
2. `torch._dynamo.config.optimize_ddp = False` -- reduced step_avg from 503ms to 448ms (short runs)

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Environment variables: `DATA_PATH`, `TOKENIZER_PATH`, `MAX_WALLCLOCK_SECONDS=600`.
