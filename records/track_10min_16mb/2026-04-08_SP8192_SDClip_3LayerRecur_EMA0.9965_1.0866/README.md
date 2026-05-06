## Record: SP8192 + SDClip + 3-Layer Depth Recurrence + EMA 0.9965 (val_bpb: 1.0866)

**val_bpb: 1.0866** (sliding window stride=64, 3-seed mean, std 0.0007) | **~15.98 MB** | 8xH100 SXM, 590s

### 3-Seed Results (8×H100 80GB SXM)

| Seed | Pre-quant BPB | Sliding BPB (s64) | Pruning | Artifact |
|------|---------------|-------------------|---------|----------|
| 42 | 1.0874 | **1.0873** | None | 15,981,300 B |
| 1337 | 1.0865 | **1.0866** | None | 15,978,870 B |
| 2024 | — | **1.0859** | None | — |

**Mean: 1.0866 | Std: 0.0007** | All artifacts under 16,000,000 bytes | Zero selective pruning

Current merged SOTA: **1.1147** (PR #1019). Delta: **−0.0281 BPB**.

### Key Changes (over PR #1445, this author)

Two major additions to the PR #1445 stack:

| Change | PR #1445 | This | Impact |
|--------|----------|------|--------|
| **Tokenizer** | SP4096 | **SP8192** | Larger vocab, better context per sequence |
| **Quantization clip** | Percentile search | **SDClip (c = k·std)** | Principled clipping, zero pruning, better rate-distortion |

### SDClip: Standard-Deviation-Based Clipping

Replaces the multi-percentile clip search with a single principled formula from PR #1394 (@clarkkev):

```
clip = k · std(row)
```

- **k=12.85** for int6 matrix parameters (mlp, attn)
- **k=20.0** for int8 embeddings

Higher k = wider clip = more values near zero = lower entropy = better compression. This directly accounts for compressed artifact size rather than just reconstruction error, and requires only one GPTQ pass per matrix instead of 5.

Result: **zero selective pruning** across all 3 seeds. The model fits comfortably under 16MB without destroying any quantized values.

### SP8192 Tokenizer

Moving from 4096 to 8192 SentencePiece tokens gives the model more granular subword representations. Combined with SDClip's superior compression, the larger embedding table fits within the 16MB budget despite doubling the vocabulary.

### Full Stack (carried from PR #1445)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Tokenizer** | SP8192 | This work |
| **SDClip k (matrices)** | 12.85 | PR #1394, this work |
| **SDClip k (embeddings)** | 20.0 | PR #1394, this work |
| Recurrence layers | 3,4,5 (3-layer, 14 virtual) | PR #1331 |
| Weight decay | 0.095 | PR #1331 |
| Matrix LR | 0.022 | PR #1331 |
| EMA decay | 0.9965 | PR #1421 (this author) |
| Recurrence start | step 2000 | PR #1445 (this author) |
| Warmdown fraction | 0.72 | PR #1445 (this author) |

### Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- Depth recurrence: layers 3,4,5 repeat (virtual 14 layers), activated at step 2000
- Skip gates, parallel residuals from layer 7, QK-Gain 5.0
- XSA on all 11 layers, LeakyReLU(0.5)²
- Shared Value Embedding (dim=128, layers 9,10)
- Tied embeddings, logit softcap=30.0

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.022, WD=0.095
- Adam (head): lr=0.008, fused=True
- AdamW (embeddings): lr=0.6, WD=0.095, fused=True
- Gradient clip: 0.3, Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 72%, EMA decay=0.9965, Wallclock: 590s

### Quantization

- Full Hessian GPTQ + Cholesky + actorder for all int6 layers
- **SDClip** (c = k·std) instead of percentile search
- Int6 per-row for MLP + attention, Int8 per-row for embeddings
- Brotli compression
- **Zero selective pruning** — model fits natively under 16MB

### Run Command

```bash
SEED=42 VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
RECUR_START_STEP=2000 WARMDOWN_FRAC=0.72 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Credits

- **SDClip quantization + SP8192 baseline**: PR #1394 by @clarkkev
- **Base architecture + depth recurrence**: PR #1334 by @aryanbhosale
- **3-layer recurrence + WD/LR tuning**: PR #1331
- **EMA decay tuning (0.9965)**: PR #1421 by @X-Abhishek-X (this author)
- **Early recurrence + extended warmdown**: PR #1445 by @X-Abhishek-X (this author)
- **SP8192 + SDClip integration**: This work
