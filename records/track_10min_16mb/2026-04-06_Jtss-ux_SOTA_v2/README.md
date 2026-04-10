# AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112

**val_bpb: 1.1151** (3-seed mean) | **~15.85 MB** | 8×H100 SXM, 600s

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 314 | 6,854 | 87.6 | 1.1343 | **1.1149** | 15,847,950 |
| 42 | 6,851 | 87.6 | 1.1346 | **1.1150** | 15,860,098 |
| 999 | 6,856 | 87.5 | 1.1348 | **1.1153** | ~15.85MB |
| **Mean** | | | | **1.1151** | |

This submission uses the SOTA configuration from PR #1019:
- XSA on all 11 layers
- Full Hessian GPTQ with autoregressive self-generated calibration
- BigramHash 3072 × 112
- LeakyReLU(0.5)² activation
- Parallel Muon optimizer
- LZMA compression

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3× with LeakyReLU(0.5)² |
| Attention | XSA on all 11 layers |
| BigramHash | 3072 × dim=112 |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Quantization | Full Hessian GPTQ int6 |
| Compression | LZMA preset=9 |
| Warmdown | 4000 iterations |
| Optimizer | Parallel Muon + Parameter Banking |

## Run Command

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

- PR #1019: AR Self-Gen GPTQ + XSA-all (SOTA, 1.1147)
- PR #549: LeakyReLU² + Legal TTT (1.1194)
- PR #414: Base architecture
