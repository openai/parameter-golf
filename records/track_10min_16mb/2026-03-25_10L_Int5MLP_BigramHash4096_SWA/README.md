# 10L Int5-MLP + BigramHash(4096) + SWA

**val_bpb: 1.1507** (mean of 3 seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Results

| Seed | val_bpb | artifact_bytes |
|------|---------|----------------|
| 42   | 1.1508  | 15,620,994     |
| 1337 | 1.1499  | 15,290,882     |
| 2024 | 1.1514  | 15,327,813     |
| **Mean** | **1.1507 +/- 0.0006** | |

## Architecture

- 10 layers, d=512, 8 heads, 4 KV heads (GQA)
- MLP: 3x expansion (1536), relu^2 activation
- BigramHash: 4096 buckets, 128-dim projection
- SmearGate (learned previous-token blending)
- U-Net skip connections with learned gates
- RoPE (base=10000), logit softcap=30.0
- Tied embeddings

## Training

- Muon optimizer (matrices) + AdamW (embeddings/scalars)
- WD=0.04, warmdown=3000 steps
- SWA: start_frac=0.4, every=50 steps
- Wallclock cap: 600s on 8xH100 (~6200 steps)
- Batch: 786,432 tokens, seq_len=2048

## Quantization

- Int5 per-row for MLP weights (clip_range=15)
- Int6 per-row for attention weights (clip_range=31)
- FP16 passthrough for small/control tensors
- Magnitude pruning (3% threshold) before quantization
- zstd-22 compression

## Evaluation

- Sliding window eval, stride=64, batch_seqs=32
- ~258s eval time on 8xH100

## Based on

- thwu1's 10L Int5-MLP submission (1.1428 BPB) with reduced BigramHash for size margin

## Reproduce

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
