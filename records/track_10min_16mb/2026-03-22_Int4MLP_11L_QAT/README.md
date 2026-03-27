# Int4 Nibble MLP + 11 Layers + QAT + stride=32

**Hypothesis:** Nibble-packed int4 MLP quantization halves raw MLP storage vs int5,
freeing ~1.8 MB (compressed) to fund an 11th transformer layer, net-positive in BPB.

## Expected val_bpb

TBD (pending run on 8×H100). Target: < **1.1428** (current SOTA).

## Key Techniques

### Int4 Nibble Packing for MLP Weights
- **Int4 [-8,7]** (16 levels) for all MLP `fc` and `proj` weights
- Row-wise nibble packing: 2 weights per byte → halves raw MLP bytes vs int5 (1 byte/weight)
- Stored as `.q4n` (nibble tensor) + `.scale` (fp16 per-row) in quantized checkpoint
- **Int6 [-32,31]** for attention and bigram weights (unchanged, precision-sensitive)

### Int4 QAT (Straight-Through Estimator)
- During training: per-row fake-quantize MLP weights to int4 range before each forward pass
- STE: `w_q = w + (round(w/scale).clamp(-8,7) * scale - w).detach()` — gradients pass through
- Model learns int4-quantization-aware weight distributions
- Negligible compute overhead (~3 element-wise ops per weight per step)

### 11 Layers (was 10)
- Funded by ~1.8 MB compressed space freed from int4 MLP
  - Old 10L int5 MLP: ~8.4 MB compressed
  - New 11L int4 nibble MLP: ~6.5 MB compressed (est. 1.5× zstd ratio)
  - 11th layer attn (int6): ~0.52 MB compressed
  - Net savings: ~1.4 MB
- 11th layer adds ~10% more compute per step → ~12.7K steps in 600s (vs ~14K at 10L)
- Time-based warmdown (`warmdown_iters=3000`) self-adjusts to cover last ~24% of training

### Eval Stride = 32 (was 64)
- Sliding window evaluation with finer stride
- Halves the "unseen context" penalty at each window boundary
- Val set is small (~37M tokens); at stride=32 eval takes ~45s total on 8×H100 — well within 10-min budget

### FP16 Keep: blocks.9.attn.c_k (updated from blocks.8)
- For the 11-layer model, keep the second-to-last layer's K projection in fp16
- Maintains the same relative architectural treatment as the 10L SOTA

## Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3×expansion (hidden=1536), relu² activation, **int4 QAT**
- SmearGate + BigramHash(10240, dim=128)
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
| 11L MLP int4 nibble | 11 × 786K = 8.65 MB | ~5.8 MB (1.5×) |
| 11L Attn int6 | 11 × 786K = 8.65 MB → 8.65 MB | ~5.7 MB (1.51×) |
| tok_emb + bigram (fp16) | ~3.7 MB | ~3.3 MB |
| Scalars, skip_weights, etc. | ~0.1 MB | ~0.08 MB |
| **Model total** | | **~14.9 MB** |
| Code (train_gpt.py) | | **~0.051 MB** |
| **Total** | | **~15.0 MB < 16 MB ✓** |

## Ablation Plan
| Change | Expected delta |
|--------|----------------|
| SOTA baseline (10L int5) | 1.1428 BPB |
| + int4 nibble MLP only | +0.002 (quality loss from coarser quant) |
| + 11th layer | -0.005 (capacity gain) |
| + QAT | -0.003 (reduces quantization gap) |
| + stride=32 eval | -0.002 (tighter sliding window) |
| **Net expected** | **~1.1358 BPB** |

Built on top of `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`.
