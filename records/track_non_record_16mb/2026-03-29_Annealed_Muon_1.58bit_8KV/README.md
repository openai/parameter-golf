# Non-record: Annealed Muon 1.58-bit Ternary (8 KV Heads)

**val_bpb: 1.2196** | **14.86 MB** | 8xH100 SXM, 595s

## Summary

Training-time 1.58-bit ternary quantization using AnnealedBitLinear layers with Muon optimizer. During training, weights are stored as full-precision latents and quantized to ternary ({-1, 0, +1}) values on each forward pass via an annealing schedule (phi-exponent) that interpolates from soft to hard quantization. At serialization, ternary weights are packed 5 values per byte using base-3 encoding and compressed with zstd-22.

## 8xH100 SXM Results

| Metric | Value |
|--------|-------|
| val_bpb | 1.2196 |
| val_loss | 2.0592 |
| Roundtrip val_bpb | 1.2249 |
| Steps | 4,622 |
| ms/step | 129 |
| Model size | 14.86 MB (15,581,721 bytes) |
| Compilation | ~145s |
| Training time | 594.7s |

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 10 (800d, 8 heads, 8 KV heads) |
| MLP | 4x relu^2 (3200) |
| Unique blocks | 10 (no weight tying) |
| Batch tokens | 524,288 |
| Quantization | AnnealedBitLinear 1.58-bit ternary (per-row scaling) |
| Packing | Base-3 uint8 (5 ternary values/byte) |
| Compression | zstd level 22 |
| Optimizer | Muon (NS steps=5) + AdamW (scalar params) |
| LR Schedule | Hold-Cosine (hold=0.70, min_lr=0.01) |
| Attention | Standard + XSA |
| RoPE | Full (base=10000) |
| SmearGate | Yes |
| U-Net skips | Yes (learned skip weights, ones-init) |
| BigramHash | 2048 buckets, dim=128 |
| Compile | torch.compile fullgraph=True, optimize_ddp=False |

## Key Differences from SOTA Lineage

This submission explores a fundamentally different quantization approach from the main leaderboard lineage:

- **Training-time ternary quantization** vs post-training GPTQ int6
- **1.58 bits/weight** vs ~6 bits/weight (before compression)
- **No post-training calibration** -- quantization is baked into training via annealing
- **Base-3 packing + zstd-22** vs GPTQ + LZMA compression
- **1024 BPE vocabulary** vs 8192 BPE (largest single source of BPB gap)

The tradeoff is lower bits-per-weight (better raw compression) at the cost of reduced model capacity per parameter. Per-element quantization noise is ~455x higher than int6, compounding through 10 layers to an output SNR of ~8.4 dB vs ~36 dB for int6.

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Environment variables: `DATA_PATH`, `TOKENIZER_PATH`, `MAX_WALLCLOCK_SECONDS=600`.
