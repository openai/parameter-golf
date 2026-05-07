## Record: 11L 512d Int8+Zlib Baseline (val_bpb: 1.2135)

**val_bpb: 1.2135** (3-seed mean) | **15.54 MB** (mean) | 8xH100 SXM, 599s

### Summary

Baseline `train_gpt.py` with `NUM_LAYERS=11` (up from the default 9). All other hyperparameters are stock defaults. This submission demonstrates the baseline architecture properly scaled with additional depth on 8xH100 SXM hardware.

### Changes from Naive Baseline

| Change | Baseline | This | Impact |
|--------|----------|------|--------|
| **Layers** | 9 | 11 | +2 layers (20.7M vs ~17M params) |
| **Everything else** | Default | Default | No other changes |

### Results (3 seeds, 8xH100 SXM)

| Seed | Steps | val_loss | val_bpb | Artifact |
|------|-------|----------|---------|----------|
| **1337** | 11,181 | 2.0484 | **1.2132** | 15.54 MB |
| 42 | 11,185 | 2.0490 | 1.2135 | 15.54 MB |
| 2025 | 11,182 | 2.0493 | 1.2137 | 15.54 MB |

**Mean: 1.2135 | Std: 0.0003** | Submitted: seed 1337 (best)

### Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 2x MLP expansion (1024 hidden)
- U-Net skip connections (5 encoder, 6 decoder)
- Tied embeddings, logit softcap=30.0
- Vocab size 1024 (SentencePiece BPE)

### Training

- Muon optimizer (matrices): lr=0.04, momentum=0.95 (warmup 0.85→0.95 over 500 steps)
- AdamW (embeddings): lr=0.05, (scalars): lr=0.04
- Gradient clip: 0.3
- Batch: 524,288 tokens/step, seq_len=1024
- Warmdown: 1200 iterations
- Warmup: 20 steps
- Wallclock cap: 599s

### Quantization

- Int8 per-row quantization for all weights
- zlib compression
- Total artifact: code (48,233 bytes) + model (15,493,717 bytes) = 15,541,950 bytes

### Run Command

```bash
NUM_LAYERS=11 SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Reproducibility

All 3 seeds produce valid artifacts under 16MB with tight variance (std=0.0003 BPB). The run uses the stock `train_gpt.py` with only `NUM_LAYERS=11` as an environment variable override.
