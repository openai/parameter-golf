# QAT + BigramHash(12288) + Stride 32

## Summary

Built on the current SOTA (`10L_Int5MLP_MuonWD04_SWA50`) with the following improvements:

- **QAT (Quantization-Aware Training):** STE fake-quantize during training — int5 for MLP layers, int6 for attention. Reduces post-quantization degradation.
- **BigramHash 12288:** Increased from 10240 to 12288 buckets for better bigram coverage.
- **Eval stride 32:** Reduced from 64 to 32 for more overlapping context windows during evaluation.
- **Magnitude pruning 5%:** Increased from 3% to improve compression ratio.
- **SWA every 25 steps:** More frequent checkpoint averaging during warmdown.

## Architecture

- 10 transformer layers, dim=512, 8 heads, 4 KV heads
- 3x MLP with SmearGate
- BigramHash(12288) with bigram_dim=128
- Mixed quantization: int5 MLP, int6 attention
- zstd-22 compression

## Results

```
seed=2024: val_bpb=1.14443, artifact=15,902,583 bytes
```

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
