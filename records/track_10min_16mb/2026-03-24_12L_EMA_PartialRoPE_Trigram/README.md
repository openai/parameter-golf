# Int4 Nibble MLP + 12 Layers + QAT + Bigram 20480 + Trigram 5120

**Hypothesis:** Int4 nibble MLP frees ~3.9 MB (compressed) over SOTA, enabling 12 layers + larger bigram (20480) + trigram hash embeddings, net-positive in BPB.

## Expected val_bpb

TBD (pending run on 8xH100). Target: < **1.1428** (current SOTA).

## Key Techniques

### Int4 Nibble Packing for MLP Weights
- **Int4 [-8,7]** (16 levels) for all MLP `fc` and `proj` weights
- Row-wise nibble packing: 2 weights per byte -- halves raw MLP bytes vs int5 (1 byte/weight)
- Stored as `.q4n` (nibble tensor) + `.scale` (fp16 per-row) in quantized checkpoint
- **Int6 [-32,31]** for attention, bigram, and trigram weights (precision-sensitive)

### Int4 QAT (Straight-Through Estimator)
- During training: per-row fake-quantize MLP weights to int4 range before each forward pass
- STE: `w_q = w + (round(w/scale).clamp(-8,7) * scale - w).detach()` -- gradients pass through
- Model learns int4-quantization-aware weight distributions
- Negligible compute overhead (~3 element-wise ops per weight per step)

### 12 Layers (was 11)
- Funded by ~3.9 MB compressed headroom from int4 MLP vs SOTA int5
- 12th layer attn (int6): ~0.52 MB compressed
- ~9% more compute per step vs 11L -- ~11.7K steps in 600s

### Bigram Hash 20480 (was 10240)
- Double the bigram buckets -- richer token-pair features
- Bigram dim=128 unchanged; added raw bytes: 10240 x 128 x 2 = 2.6 MB, compressed ~1.8 MB

### Trigram Hash Embedding (new)
- 5120 buckets, dim=64 -- hashes consecutive token triples into a learned embedding
- Hash: `(73856*t[i] ^ 19349*t[i-1] ^ 83492*t[i-2]) % (vocab-1)`
- Added to residual stream before transformer blocks (same as bigram)
- Quantized as int6 (classified alongside bigram); raw ~0.33 MB, compressed ~0.3 MB
- Zero-initialized: starts neutral, learns to contribute if useful

### Eval Stride = 32 (was 64)
- Sliding window evaluation with finer stride
- Halves the "unseen context" penalty at each window boundary
- Val set is small (~37M tokens); at stride=32 eval takes ~45s total on 8xH100 -- well within 10-min budget

### FP16 Keep: blocks.10.attn.c_k
- Second-to-last layer's K projection in fp16 (same relative treatment as SOTA 10L)

## Architecture
- 12 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation, **int4 QAT**
- SmearGate + BigramHash(20480, dim=128) + TrigramHash(5120, dim=64)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Training Hyperparameters (unchanged from SOTA)
- Muon: matrix_lr=0.02, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000 iters (time-based), warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.4, every=50 steps

## Space Budget Estimate

| Component | Raw bytes | Compressed (est.) |
|-----------|-----------|-------------------|
| 12L MLP int4 nibble | 12 x 786K = 9.43 MB | ~6.3 MB |
| 12L Attn int6 | 12 x 786K = 9.43 MB | ~6.3 MB |
| tok_emb fp16 | 1024 x 512 x 2 = 1.0 MB | ~0.9 MB |
| Bigram 20480 fp16 | 20480 x 128 x 2 = 5.2 MB | ~1.9 MB |
| Trigram int6 | 5120 x 64 = 0.33 MB | ~0.3 MB |
| Scalars, skip_weights, etc. | ~0.1 MB | ~0.08 MB |
| **Model total** | | **~15.8 MB** |
| Code (train_gpt.py) | | **~0.055 MB** |
| **Total** | | **~15.9 MB < 16 MB** |

## Ablation Plan
| Change | Expected delta |
|--------|----------------|
| SOTA baseline (10L int5) | 1.1428 BPB |
| + int4 nibble MLP only | +0.002 (quality loss from coarser quant) |
| + 11th layer | -0.005 (capacity gain) |
| + QAT | -0.003 (reduces quantization gap) |
| + 12th layer | -0.004 (more capacity) |
| + bigram 20480 | -0.001 (richer token-pair features) |
| + trigram 5120 | -0.002 (token-triple context) |
| + stride=32 eval | -0.002 (tighter sliding window) |
| **Net expected** | **~1.131 BPB** |

Built on top of `2026-03-22_Int4MLP_11L_QAT`.
