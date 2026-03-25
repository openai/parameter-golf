## 11L EMA + GPTQ-lite + Int6 + zstd (val_bpb: 1.1240)

**val_bpb: 1.1240** (sliding window stride=64) | **15.58 MB** | 8xH100 SXM, 600s

### Architecture

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
- OrthoInit + muP-scaled output projections

### Quantization

- GPTQ-lite: Per-row optimal clip percentile search (5 candidates) for int6
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Results (8xH100 SXM, seed 1337)

| Metric | Value |
|--------|-------|
| Steps completed | 7,002 |
| Training time | 600s |
| val_bpb (sliding window, stride=64) | **1.1240** |
| val_bpb (standard eval, int6 roundtrip) | 1.1479 |
| val_loss (sliding window) | 1.8979 |
| Model size (int6+zstd) | 15.58 MB |

### Run Command

```bash
RUN_ID=scored_run DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Dependencies

Requires `zstandard` package for zstd compression:
```bash
pip install zstandard
```
