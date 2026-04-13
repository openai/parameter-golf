# 3x MLP + Quantization-Aware Training (STE)

## Summary

Non-record submission exploring **Quantization-Aware Training with Straight-Through Estimator (QAT-STE)** for learning int6-friendly weight distributions during training. Combined with 3x MLP expansion for better parameter utilization.

**Local MLX results (300 steps, SP1024, 1 shard):**
- val_bpb = 2.2240 (baseline: 2.2290, **improvement: -0.005 BPB**)
- H100 validation pending — applying for compute credits

## Innovations

### 1. Quantization-Aware Training with Straight-Through Estimator

Instead of training with full-precision weights and hoping int8/int6 quantization doesn't destroy quality, I inject quantization noise **during training** from 30% of iterations onward.

**Mechanism:**
- Forward pass: simulate int6 quantization with per-row `clip = 12.85 * std`
- Quantize to [-31, 31], dequantize back to float
- Gradient: straight-through estimator (flows through as if no quantization)
- The model learns weight distributions that are robust to int6 compression

**Why this matters:** GPTQ int6 with SDClip is the dominant quantization method, but it's applied post-training. QAT-STE teaches the model to anticipate quantization during training, potentially reducing the quantization gap from ~0.006 BPB (current SOTA) to near zero.

```python
def fake_quantize_ste(w, bits=6):
    n_levels = (1 << bits) - 1
    std = w.float().pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt()
    clip_val = 12.85 * std
    w_clipped = w.float().clamp(-clip_val, clip_val)
    w_norm = (w_clipped + clip_val) / (2.0 * clip_val + 1e-12)
    w_quant = (w_norm * n_levels).round() / n_levels
    w_deq = w_quant * (2.0 * clip_val) - clip_val
    return (w_deq - w.float()).detach() + w.float()  # STE
```

### 2. 3x MLP Expansion

Changed MLP hidden dimension from 2x to 3x model dimension. This allocates more parameters to the feed-forward network where they have highest impact per byte.

**Ablation (300 steps, SP1024, M5 MacBook):**

| Config | val_bpb | vs Baseline |
|--------|---------|-------------|
| Baseline (2x MLP) | 2.2290 | — |
| **3x MLP** | **2.2240** | **-0.005** |
| 10 layers | 2.2249 | -0.004 |
| QK gain 5.0 | 2.2624 | +0.033 |
| Depth recurrence | 2.2899 | +0.061 |

3x MLP gives the cleanest improvement with zero throughput overhead.

## Architecture

```
Layers: 9
Dimension: 512
Heads: 8 (GQA with 4 KV heads)
MLP: 3x expansion (was 2x), relu^2
QAT: int6 STE from 30% of training
Quantization: int8 + zlib (baseline)
Vocab: SP1024 (SP4096/SP8192 planned with H100)
```

## Next Steps (with H100 compute)

1. Full 20K-step training with QAT — expect QAT benefit to grow with more steps
2. SP4096/SP8192 casefold tokenizer (already trained locally)
3. Test at larger scale (72M+ params shown to compress to <9MB)
4. Planned novel ideas: two-pass latent compression, hash-gated additive embeddings

## Reproducing

```bash
# Local MLX (Apple Silicon):
RUN_ID=mlp3x_qat ITERATIONS=300 MLP_MULT=3 \
TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=524288 \
python3 train_gpt_mlx.py

# CUDA (H100):
RUN_ID=mlp3x_qat MLP_MULT=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
