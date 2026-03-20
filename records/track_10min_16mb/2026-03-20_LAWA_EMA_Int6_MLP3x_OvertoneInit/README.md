## 11-Layer Int6 + LAWA-EMA + Overtone Init (val_bpb: 1.1551)

**val_bpb = 1.1551** (sliding window, stride=64) | **12.7 MB** artifact | 8xH100 SXM, 600s

### Changes from PR #198

| | [PR #198](https://github.com/openai/parameter-golf/pull/198) | This |
|---|---|---|
| val_bpb (sliding s64) | 1.1318 | **1.1551** |
| Weight averaging | SWA (~8 ckpt, warmdown only) | LAWA-EMA (every step, decay=0.995) |
| Embedding init | Normal | Overtone (SVD power-law) |
| Artifact size | 15.7 MB | **12.7 MB** |
| Steps (600s) | 7,412 | 6,715 |
| Step time | 81ms | 89ms |

### What's new

1. **LAWA-EMA** (replaces SWA). Float32 exponential moving average of all parameters, updated every step with decay=0.995. Effective window ~200 steps. Applied to base model before int6 quantization.

2. **Overtone init**. SVD decomposes the random embedding matrix, replaces singular values with power-law decay (1/sqrt(k)). Produces smoother per-row value ranges for tighter int6 quantization.

3. **BigramHashEmbedding.proj zero-init fix**. The `_init_weights` method was overwriting BigramHashEmbedding.proj's intended zero initialization with orthogonal init. Fixed by setting `_zero_init=True` on the proj layer.

4. **Sliding window eval fix**. Partial windows at the validation boundary were double-counting tokens. Fixed by only generating full windows (`ws + seq_len <= total`).

### Carried from PR #198

- 11 transformer layers (5 encoder + 6 decoder, U-Net skip connections)
- Int6 per-row quantization (MLP+attention), int8 embedding, zstd-22 compression
- MLP 3x (hidden=1536), relu² activation
- FlashAttention 3 (direct `flash_attn_func` calls)
- SmearGate + BigramHash (2048x128)
- Orthogonal + muP-scaled init on all large matrices
- Weight decay 0.04 (Muon + AdamW)
- GQA (8 heads, 4 KV heads), logit softcap 30.0
- Sequence length 2048, NTK-aware RoPE
- Muon optimizer, momentum 0.99, warmdown 1200 iters, grad clip 0.3

### Configuration

```bash
NUM_LAYERS=11 MUON_WD=0.04 ADAM_WD=0.04 BIGRAM_VOCAB_SIZE=2048 \
LAWA_ENABLED=1 LAWA_EMA_DECAY=0.995 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
torchrun --nproc_per_node=8 train_gpt.py
```

### Key metrics

- 6,715 steps in 600s (89ms/step)
- ~5.3B train tokens (6,715 steps x 786,432 tokens/step)
- Peak memory: 19,828 MiB per GPU

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1622 |
| Int6 roundtrip val_bpb | 1.1779 |
| **Int6 sliding val_bpb (s64)** | **1.1551** |
| Compressed artifact (int6+zstd) | 12,639,639 bytes |
| Code size | 65,258 bytes |
| **Total submission size** | **12,704,897 bytes** |

### Reproducibility

Single seed run (seed=1337). Additional seed runs pending.

| Seed | Steps | Sliding s64 | Artifact |
|------|-------|-------------|----------|
| 1337 | 6,715 | 1.1551 | 12,704,897 |

### Included files

- `train_gpt.py` -- full training + quantization + evaluation script
- `train.log` -- training log from seed 1337
- `submission.json` -- leaderboard metadata
