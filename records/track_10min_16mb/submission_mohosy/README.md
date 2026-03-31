# 11L EMA + XSA + Int6 MLP3x + SmearGate + BigramHash

**Score:** Pending compute validation

## Summary

11-layer transformer with EMA weight averaging, Exclusive Self Attention on last 4 layers, Int6 mixed quantization + zstd-22 compression.

## Architecture

| Component | Details |
|-----------|---------|
| Layers | 11, 512 dim, 8 heads, 4 KV heads (GQA) |
| MLP | 3x expansion (hidden=1536), ReLU^2 |
| Quantization | Int6 mixed precision (MLP+attention), Int8 (embeddings) |
| Compression | zstd-22 |
| SmearGate | Learned sigmoid token blending |
| BigramHash | 2048-bucket hash embedding (dim 128) |
| XSA | Exclusive Self Attention on last 4 layers |
| EMA | Exponential moving average (decay=0.997) |
| Optimizer | Muon (WD=0.04, momentum=0.99) |
| Attention | FlashAttention 2/3 |
| Sequence | Train@2048, eval@2048 |
| Eval | Sliding window stride=64 |

## Run command

```bash
NUM_LAYERS=11 XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 EVAL_STRIDE=64 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

- TTT disabled based on findings from PR #303 showing negative interaction with EMA+XSA
- Based on techniques from PR #287 (jfprincz)
- Waiting on compute credits to produce verified score
