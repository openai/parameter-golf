# Weight-Tied 6L×2 Transformer (Non-record Submission)

## Summary

Weight-Tied Transformer architecture achieving **1.1639 BPB** with a **13.2MB artifact** 
(83% of the 16MB budget). Instead of 9-11 independent layers, 6 unique transformer 
blocks are reused across 2 passes (encoder + decoder), creating 12 effective layers 
from the parameter cost of 6. The saved parameter budget is reinvested into expanding 
model dimension from 512 to 672.

## Key Idea

Standard Parameter Golf submissions pack 9-11 independent layers at d=512, using 
~15.9MB. This submission takes a different approach:

- **6 unique blocks × 2 passes** = 12 effective layers
- **d=672** (vs d=512) — each token representation is 31% richer
- **13.2MB artifact** — 2.8MB headroom remaining

The same weights process fundamentally different inputs on each pass: pass 1 (encoder) 
sees raw embeddings; pass 2 (decoder) sees processed representations. U-Net skip 
connections link encoder outputs back to decoder layers, and per-block cached 
LayerNorm scaling (1/√layer_idx) differentiates effective depth positions.

## Architecture

| Component | Detail |
|---|---|
| Unique blocks | 6 |
| Passes | 2 (encoder + decoder) |
| Effective depth | 12 layers |
| Model dimension | 672 |
| Attention | 8 heads, 4 KV heads (GQA) |
| MLP | LeakyReLU(0.5)², 3× expansion |
| Embedding | Tied, 1024 vocab |
| Position | Partial RoPE (16 dims) |

## Techniques Stack

Built on top of the current SOTA stack:
- **SmearGate**: bigram context injection via learned gate (~512 params)
- **BigramHash**: 2048 buckets, dim=128, token-pair embeddings
- **XSA (all 6 unique blocks)**: Exclusive Self Attention on every block
- **EMA**: Exponential Moving Average with warmup-corrected decay (0.997)
- **Late QAT**: int6 fake-quantization enabled in final ~15% of training
- **GPTQ int6 + lzma**: Full Hessian-aware quantization with AR self-gen calibration
- **Sliding Window Eval**: stride=64
- **Parallel Muon**: batched Newton-Schulz with reduce-scatter overlap

## Results

| Metric | Value |
|---|---|
| val_bpb (int6 sliding window) | **1.1639** |
| val_bpb (int6 standard) | 1.1878 |
| val_bpb (end of training) | 1.1776 |
| Training steps | 3087 / 20000 (600s wallclock cap) |
| Step avg | 194.4 ms |
| Artifact size | 13,217,085 bytes (13.2 MB) |
| Peak GPU memory | 41,955 MiB |
| GPU | 8×H100 SXM |
| Training time | 600s |

## Comparison to Baseline

| Config | val_bpb | Artifact | Efficiency |
|---|---|---|---|
| Naive baseline (9L, d=512) | 1.2244 | 15.9 MB | baseline |
| Our exp6 (9L+XSA+EMA, d=512) | 1.2088 | 15.9 MB | -0.016 bpb |
| **This submission (6L×2, d=672)** | **1.1639** | **13.2 MB** | **-0.061 bpb, -17% size** |

## Why This Matters

Weight-Tied depth has been attempted by several participants but none have matched 
independent-layer performance. This submission demonstrates that with the right 
combination of techniques (cached LN scaling, XSA on all blocks, EMA, GPTQ int6), 
Weight-Tied can not only match but exceed independent-layer submissions while using 
significantly less of the 16MB budget.

The 2.8MB headroom opens possibilities for further improvements (dimension expansion, 
additional techniques) that are impossible for submissions already at the 16MB limit.

## Ongoing Work

- **Adaptive Depth**: Dynamic per-token computation depth (pass 2+ with gradient detach)
- **5L×3 configuration**: 15 effective layers with detach strategy for memory efficiency  
- **LoRA per-pass perturbation**: Low-rank weight deltas for each pass to increase diversity

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training completes in ~10 minutes on 8×H100 SXM including GPTQ calibration and evaluation.
