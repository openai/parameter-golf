# 12L Int5-MLP BigramHash10K EMA GPTQ-lite

## Approach

Built on top of the 11L EMA+GPTQ-lite SOTA (PR#414, 1.1233 BPB), with two key structural changes:

### 1. Mixed Int5/Int6 Quantization
- MLP weights quantized to **Int5** [-16, 15] instead of Int6, saving ~1.5 MB compressed
- Attention weights remain Int6 [-32, 31] (precision-critical)
- QAT (Quantization-Aware Training) updated to match: MLP layers use clip_range=15, attention uses clip_range=31

### 2. 12th Transformer Layer
- Space freed by Int5 MLP funds a 12th layer
- U-Net skip connections updated: 6 encoder + 6 decoder
- XSA on last 4 layers (8-11)
- Value embeddings on layers 10, 11

### 3. BigramHash(10240)
- Expanded from 2048 to 10240 buckets
- XOR hash of consecutive token pairs into learned 128-dim embeddings
- Reduces hash collisions, improves bigram-level signal

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Model dim | 512 |
| MLP expansion | 3x (hidden=1536) |
| Attention heads | 8Q / 4KV (GQA) |
| Vocabulary | 1024 (SP-1024) |
| Activation | ReLU-squared |
| Embedding | Tied FP16 |
| BigramHash | 10240 buckets, dim 128 |
| SmearGate | Yes |
| OrthoInit | Yes |
| XSA | Last 4 layers |
| Partial RoPE | 16 dims |
| LN Scale | Yes |
| Value Embed | Layers 10,11, dim 128 |

## Training

- 8xH100 SXM (SDPA fallback, no FA3)
- Muon optimizer (WD=0.04, momentum=0.99, 1500-step warmup)
- EMA (decay=0.997)
- SWA (last 20% of warmdown, every 50 steps)
- Late QAT at warmdown scale < 0.15
- GPTQ-lite clip search (5 percentiles)
- Stopped at step 4973/20000 (600s wallclock cap)
- Peak memory: 27943 MiB

## Quantization

| Component | Precision | Compression |
|-----------|-----------|-------------|
| MLP weights | Int5 [-16,15] | zstd-22 |
| Attention weights | Int6 [-32,31] | zstd-22 |
| Embeddings | FP16 | zstd-22 |
| Control tensors | FP32 | zstd-22 |

## Results

| Metric | Value |
|--------|-------|
| val_loss (sliding window s64) | 1.93767556 |
| **val_bpb (sliding window s64)** | **1.14760365** |
| val_loss (int6 roundtrip) | 1.97986425 |
| val_bpb (int6 roundtrip) | 1.17258713 |
| Artifact size | 15,497,769 bytes |
| Model (int6+zstd) | 15,411,791 bytes |
| Code | 85,978 bytes |

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
