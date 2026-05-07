## Record: 11L Enhanced GPTQ-lite Per-Row (val_bpb: 1.1218)

**val_bpb: 1.1218** (3-seed mean, sliding window stride=64) | **15.92 MB** (mean) | 8xH100 SXM, 600s

### Key Innovation: Enhanced GPTQ-lite Per-Row Optimal Clip Search

The main improvement over prior SOTA (1.1233) is an enhanced GPTQ-lite quantization:

| Change | Prior SOTA | This | Impact |
|--------|-----------|------|--------|
| **Clip candidates** | 5 percentiles | 13 percentiles | Finer search grid |
| **Selection granularity** | Global (whole matrix) | **Per-row** | Each row gets optimal clip |
| **Total** | **1.1233** | **1.1218** | **-0.0015 BPB** |

Instead of picking one best percentile for an entire weight matrix, each row independently selects the clip percentile that minimizes its reconstruction MSE. The search grid is expanded from 5 to 13 candidates: `[0.995, 0.997, 0.998, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99993, 0.99995, 0.99997, 0.99999, 1.0]`. This is applied during post-training quantization with zero training cost.

### Results (3 seeds, 8xH100 SXM)

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 6,955 | 1.8932 | **1.1212** | 15.92 MB |
| 42 | 6,966 | 1.8944 | 1.1220 | 15.91 MB |
| 2024 | 6,951 | 1.8948 | 1.1222 | 15.95 MB |

**Mean: 1.1218 | Std: 0.0005** | Submitted: seed 1337 (best)

### Architecture (from prior SOTA stack)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), relu-squared activation
- U-Net skip connections (5 encoder, 6 decoder)
- Efficient Partial XSA on last 4 layers (GQA-aware)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10) with per-layer learned scales
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations (wallclock-based)
- EMA: decay=0.997, every step
- Tight SWA: every 50 steps when scale<0.2
- Late QAT: STE int6 fake-quantization when LR scale<0.15

### Quantization

- **Enhanced GPTQ-lite**: Per-row optimal clip percentile search (13 candidates) for int6
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Run Command

```bash
RUN_ID=run SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Dependencies

Requires `zstandard` package:
```bash
pip install zstandard
```

### Reproducibility

All 3 seeds produce valid artifacts under 16MB with tight variance (std=0.0005 BPB). The enhanced GPTQ-lite clip search is deterministic.
