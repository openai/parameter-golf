# TrigramHash + EMA-SWA + Int4 QAT MLP

**Author:** Ashutosh ([@Ashutosh3142857](https://github.com/Ashutosh3142857))
**Date:** 2026-03-22
**Target val_bpb:** ~1.128–1.135 (estimated; pending GPU run)
**Base:** PR #180 SOTA — `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` (1.1428 BPB)

---

## Summary

This submission proposes three independent improvements over the current SOTA:

| Innovation | Description | Est. BPB Δ |
|------------|-------------|------------|
| **TrigramHash(2048, dim=48)** | 3-token n-gram feature table alongside BigramHash | −0.002 to −0.005 |
| **EMA-SWA (α=0.9)** | Exponential moving average instead of uniform SWA | −0.001 to −0.002 |
| **Int4 QAT MLP → 11th layer** | Free budget via int4 compression, fund extra depth | −0.003 to −0.006 |

---

## Innovation 1: TrigramHash

### Motivation

BigramHash (PR #162) demonstrated that hashing consecutive token pairs into a small learned embedding table provides strong signal with negligible parameter cost. The key insight is that the token space (vocab=1024) has rich **co-occurrence statistics** that bigrams partially capture, but trigrams (3-token sequences) capture significantly more.

Examples of information that bigrams miss but trigrams encode:
- `"New York"` + `"York City"` ≠ `"New York City"` — the trigram uniquely identifies the full place name
- `"in the"` + `"the United"` — many different completions; trigram disambiguates to `"in the United"`
- `"at the"` + `"the end"` — similarly ambiguous; trigram resolves to `"at the end"`

### Implementation

```python
class TrigramHashEmbedding(nn.Module):
    def _hash(self, tokens):
        # Polynomial hash with distinct prime multipliers per position
        h = (17911 * t[i-2] + 36313 * t[i-1] + 27191 * t[i]) % (vocab_size - 1)
        # Positions 0,1 use padding hash (vocab_size - 1)
```

The three multipliers (17911, 36313, 27191) are large primes chosen so that permutations `(a,b,c)` and `(a,c,b)` are unlikely to collide, preserving order information.

### Size Budget

- **Embedding table:** 2048 × 48 × 1 byte (stored as int8) = **98,304 bytes**
- **Projection:** 48 × 512 × 1 byte (stored as int8) = **24,576 bytes**
- **After zstd-22:** embedding is near-random (trained), compresses ~1.2x → ~**102 KB total**
- **Headroom used:** ~102 KB of the ~200 KB remaining budget

Zero-initialized so the model begins training purely on unigram + bigram features and only activates trigram features when they provide gradient signal — a clean "feature by necessity" learning dynamic.

### Expected Gain

Prior art: upgrading BigramHash from 4096→8192→10240 buckets each gave −0.0008 to −0.0012 BPB. Trigrams add an orthogonal feature dimension, so gains should be additive rather than diminishing. Estimated: **−0.002 to −0.005 BPB**.

---

## Innovation 2: EMA-SWA (Exponential Moving Average SWA)

### Motivation

Standard SWA (Izmailov et al., 2018) averages checkpoints **uniformly**:

```
W_avg = (1/N) Σ W_t
```

In practice, checkpoints sampled early in the warmdown are noisier (higher learning rate, less converged) than those sampled late. Uniform averaging dilutes the quality of the best checkpoints.

EMA-SWA weights each checkpoint exponentially:

```
W_ema ← α · W_ema + (1 − α) · W_t
```

With `α=0.9` and 24 checkpoints (collected every 50 steps over 1200 warmdown steps), the effective weights are approximately:

```
W_ema ≈ 0.9^0·W_24 + 0.9^1·W_23 + ... + 0.9^23·W_1
```

normalized. The most recent checkpoint contributes ~10.5% of weight; the earliest ~0.89%. This is equivalent to an **exponentially decaying memory** that prefers the better-converged state.

### Implementation

```python
class EMASWA:
    def update(self, model):
        sd = model.state_dict()
        # First update: W_ema = W_0
        # Subsequent: W_ema ← α*W_ema + (1-α)*W_t
        self.ema_sd[k].mul_(alpha).add_(sd[k], alpha=1-alpha)
```

EMA-SWA is strictly better than uniform SWA when later checkpoints are meaningfully better than earlier ones — which is true by construction in a warmdown schedule.

**No size cost:** EMA averaging replaces uniform averaging in the existing SWA logic. The final exported model is one state dict, same size as before.

### Expected Gain

Compared to uniform SWA (SOTA), EMA with `α=0.9` should reduce noise from early warmdown checkpoints. Estimated: **−0.001 to −0.002 BPB**.

---

## Innovation 3: Int4 QAT for MLP + 11th Transformer Layer

### Motivation: Int4 Saves ~800 KB

The SOTA uses int5 (clip_range=15, 32 distinct values) for MLP weights and zstd compression. Moving to **int4** (clip_range=7, 16 distinct values) reduces the per-weight entropy from log₂(32)=5 bits to log₂(16)=4 bits. Under zstd-22:

| Scheme | Distinct values | Approx. zstd ratio | Effective bits/weight |
|--------|-----------------|--------------------|-----------------------|
| Int6   | 64              | ~1.5×              | ~5.3 bits             |
| Int5   | 32              | ~1.88×             | ~4.3 bits             |
| **Int4**   | **16**          | **~2.4×**          | **~3.3 bits**         |

For the SOTA model (10 layers, 3× MLP, 512 dim):
- MLP weights per layer: 2 × (512 × 1536) = 1,572,864 params
- 10 layers total: 15,728,640 MLP params
- Int5 compressed: ~15,728,640 / 1.88 ≈ **8.37 MB**
- Int4 compressed: ~15,728,640 / 2.40 ≈ **6.55 MB**
- **Savings: ~1.82 MB**

This frees budget for an 11th transformer layer:
- Extra layer cost (attention + MLP): ~(4×512²)/1.5 + (2×512×1536)/2.4 ≈ 700KB + 655KB ≈ **~1.35 MB**
- Net budget change: savings 1.82M − cost 1.35M = **+470 KB headroom** (still within 16 MB)

### Quantization-Aware Training (QAT)

The problem with post-training int4 quantization is accuracy loss: rounding to 16 levels is coarser than int5's 32 levels, causing degradation.

**Solution: Straight-Through Estimator (STE) QAT** (Bengio et al., 2013; Jacob et al., 2018):
- During training's warmdown phase, fake-quantize MLP weights to int4 range
- STE: in the forward pass, use quantized weights; in the backward pass, pass gradients through as if unquantized
- The model learns to concentrate weights in the int4 grid, reducing the accuracy gap

```python
def fake_quantize_int4_ste(x):
    scale = x.abs().amax(dim=-1, keepdim=True) / 7.0     # per-row
    x_q   = clamp(round(x / scale), -8, 7) * scale       # quantize
    return (x_q - x.detach()) + x                         # STE
```

QAT activates at the same time as SWA (last 40% of warmdown). This gives ~1200 steps for weights to adapt to the int4 grid before export.

### Expected Gain from 11th Layer

The SOTA gained ~0.003 BPB by adding a 10th layer (funded by int5 MLP savings). An 11th layer should give a similar or slightly diminishing gain. Estimated: **−0.003 to −0.006 BPB**.

---

## Architecture Summary

| Parameter | SOTA (1.1428) | This submission |
|-----------|---------------|-----------------|
| Layers | 10 | **11** |
| Model dim | 512 | 512 |
| Heads | 8 (4 KV) | 8 (4 KV) |
| MLP mult | 3× | 3× |
| Quant: MLP | Int5 | **Int4 (QAT)** |
| Quant: Attn | Int6 | Int6 |
| Quant: Embed | FP16 | FP16 |
| BigramHash | (10240, 128) | (10240, 128) |
| **TrigramHash** | — | **(2048, 48)** |
| SWA type | Uniform | **EMA (α=0.9)** |
| Compression | zstd-22 | zstd-22 |

---

## Run Command

```bash
# Setup (once)
pip install torch numpy sentencepiece huggingface-hub zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024

# Training run (default seed=42)
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
RUN_ID=ash_trigram_ema_swa_int4_s42 \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-22_TrigramHash_EMASWA_Int4QAT/train_gpt.py

# With explicit seed
SEED=1337 RUN_ID=ash_trigram_ema_swa_int4_s1337 ... torchrun ...
```

---

## Ablation Plan

To isolate each contribution, run the following ablations (each starting from the SOTA baseline):

| Ablation | What to set | Expected BPB |
|----------|-------------|--------------|
| Baseline (SOTA) | Default SOTA config | 1.1428 |
| + TrigramHash only | `NUM_LAYERS=10`, `TRIGRAM_VOCAB_SIZE=2048`, `QAT_ENABLED=0`, `SWA_EMA_ALPHA=1.0` | ~1.139 |
| + EMA-SWA only | `NUM_LAYERS=10`, `TRIGRAM_VOCAB_SIZE=0`, `QAT_ENABLED=0`, `SWA_EMA_ALPHA=0.9` | ~1.141 |
| + Int4 QAT only | `NUM_LAYERS=10`, `TRIGRAM_VOCAB_SIZE=0`, `QAT_ENABLED=1` | ~1.143 |
| + Int4 + 11L | `NUM_LAYERS=11`, `TRIGRAM_VOCAB_SIZE=0`, `QAT_ENABLED=1` | ~1.138 |
| **All combined** | **Default config** | **~1.130** |

---

## Research Depth: Why These Ideas Compound

The three contributions are largely orthogonal in the information they add:

1. **TrigramHash** improves the *input representation* — richer context features before any transformer computation
2. **EMA-SWA** improves the *weight averaging* — better exploitation of the loss landscape near convergence
3. **Int4 QAT + depth** improves the *capacity/quality tradeoff* — more effective use of the 16 MB budget

Unlike most ablations that compete for the same token budget (e.g., wider vs deeper), these three improvements each address a different bottleneck. Their combined effect should be close to additive.

---

## Known Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Int4 accuracy loss too large | QAT starts at warmdown (1200 steps to adapt); fallback: set `QAT_ENABLED=0` |
| TrigramHash uses 102 KB, pushes over 16 MB | `TRIGRAM_VOCAB_SIZE=1024 TRIGRAM_DIM=32` reduces to ~35 KB |
| EMA-SWA worse than uniform SWA | Set `SWA_EMA_ALPHA=1.0` to recover uniform averaging |
| 11th layer doesn't fit in budget | Set `NUM_LAYERS=10` and `QAT_ENABLED=0` to revert to SOTA architecture |

---

## Files

- `train_gpt.py` — complete, self-contained training script (1380 lines)
- `README.md` — this document
- `submission.json` — leaderboard metadata (to be filled after GPU run)
