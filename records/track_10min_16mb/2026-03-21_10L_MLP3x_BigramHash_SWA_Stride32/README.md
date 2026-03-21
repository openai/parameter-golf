# 10L MLP3x + BigramHash(2048) + SWA + Stride-32 Sliding Eval

## Summary

10-layer transformer with 3x relu² MLP, BigramHash(2048), SmearGate, orthogonal initialization, mixed int5/int6 quantization + zstd-22 compression, SWA weight averaging, and dense stride-32 sliding window evaluation.

**val_bpb: 1.1487 (mean of 3 seeds)** | **14.9 MB** | 8xH100 SXM, 600s

### Stride-32 dense sliding eval

Standard submissions use stride-64 or stride-256 for sliding window evaluation. Stride-32 scores every token with near-maximum context from the 2048-token window, with only 32 positions of new content per window. This approximately doubles eval time but extracts more quality from the same trained model.

### Techniques used
- 10-layer relu² MLP with 3x expansion (hidden=1536)
- BigramHash(2048, dim=128) + SmearGate for n-gram features
- Orthogonal initialization with scaled projections
- Mixed quantization: int5 for MLP, int6 for attention, fp16 for embeddings
- 5% magnitude pruning + zstd-22 compression
- SWA: averaging 24-25 checkpoints during warmdown
- Muon optimizer (WD=0.04) + AdamW for embeddings/scalars
- Training: seq_len=2048, batch=786K tokens, grad_clip=0.3

## Results

### Reproducibility (3 seeds)

| Seed | Steps | Step time | Sliding s32 BPB | Artifact |
|------|-------|-----------|-----------------|----------|
| 1337 | 6,374 | 94ms | 1.1503 | 14.90 MB |
| 42 | 6,626 | 91ms | 1.1493 | 14.78 MB |
| 2025 | 6,622 | 91ms | 1.1464 | 14.99 MB |

**Mean: 1.1487** | Std: 0.0020

### Summary (mean of 3 seeds)

| Metric | Value |
|--------|-------|
| val_bpb (sliding s32, mean) | **1.1487** |
| val_bpb std | 0.0020 |
| Model params | 24,468,561 |
| Artifact size | ~14.9 MB |
| Code size | 46,541 bytes |

## Architecture

| Component | Detail |
|-----------|--------|
| Layers | 10 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP | relu² 3x expansion (hidden=1536) |
| Vocab | 1024 (SentencePiece BPE) |
| Seq length | 2048 |
| BigramHash | vocab=2048, dim=128 |
| SmearGate | learnable prev-token blending |
| Skip connections | U-Net encoder/decoder with learned skip weights |
| Embeddings | Tied input/output |

## Run Command

```bash
RUN_ID=submission \
SEED=1337 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

8x NVIDIA H100 80GB SXM, 600s training cap.
