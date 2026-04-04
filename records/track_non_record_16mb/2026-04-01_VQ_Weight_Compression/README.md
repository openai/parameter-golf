# VQ-VAE Weight Compression for Parameter Golf

## Summary

This submission replaces the standard scalar INT5/INT8 weight quantization with **Vector Quantization (VQ)** -- a learned discrete weight representation inspired by VQ-VAE. Instead of quantizing each weight independently, we group pairs of weights and map them to learned codebook entries, exploiting pairwise correlations that scalar quantization misses.

This is a **non-record submission** targeting the competition's bounty list for novel compression techniques (related to "ternary quantization" and "learning adapters on random linear maps").

## Key Innovation: VQ-VAE Codebook Training During Warmdown

Standard post-training quantization (PTQ) applies quantization after training is complete. Our approach trains the codebook **jointly with the model** during the warmdown phase:

1. **Codebook initialization**: Random sampling from weight vectors at warmdown start
2. **Every 50 warmdown steps**:
   - Find nearest codebook entry for each weight pair (cosine similarity on unit sphere)
   - Snap weights to codebook entries (quantization)
   - Update codebook via EMA (decay=0.95) toward assigned vectors
   - Replace dead entries (unused for 5+ snaps) with random weight samples
3. **Result**: Model weights and codebook co-evolve -- weights learn to be codebook-friendly, codebook learns the important weight patterns

### Why VQ Beats Scalar Quantization

At the same bitrate (5 bits/param):
- **INT5**: 32 levels per weight, 32x32=1024 possible pairs on a **rigid rectangular grid**
- **VQ G=2 K=1024**: 1024 possible pairs placed **optimally in 2D weight space** via learning

VQ concentrates codebook entries where the weight distribution is dense, wasting no capacity on empty regions.

### Experimental Results (8-minute scale)

| Method | Quantization Delta | Notes |
|--------|-------------------|-------|
| INT5 scalar | +0.0049 | Production baseline |
| **VQ G=2 K=1024 (post-training k-means)** | **+0.0024** | 2x better, same bitrate |
| **VQ-VAE trained codebook** | **+0.0018** | Co-evolved during warmdown |

### VQ-VAE Sweep Results

| Similarity | EMA Decay | Delta | Roundtrip BPB |
|-----------|-----------|-------|---------------|
| **Cosine** | **0.95** | **+0.0018** | **1.4392** |
| Cosine | 0.99 | +0.0022 | 1.4420 |
| L2 | 0.95 | +0.0007 | 1.4584 |
| L2 | 0.99 | +0.0012 | 1.4541 |

Cosine similarity + EMA 0.95 gives the best roundtrip (raw quality + compression quality combined).

## Architecture

- 10 transformer layers, 512 model dim, 8 attention heads, 4 KV heads (GQA)
- MLP expansion 3.0x (hidden=1536), SiLU activation
- Tied embeddings (1024 vocab), partial RoPE (16/64 dims)
- SmearGate (token blending), orthogonal init, logit softcap=8.0
- U-Net skip connections between encoder/decoder halves

## Training

- Muon optimizer (LR=0.08, momentum=0.90)
- 70% warmdown schedule (time-based)
- SWA-tight: average last 3% of warmdown (~24 checkpoints)
- Full Hadamard rotation before quantization
- VQ-VAE codebook training: snap every 50 warmdown steps

## Compression

- **VQ G=2, K=1024**: group pairs of weights, 1024-entry codebook per layer
- **Sphere codebook**: entries normalized to unit sphere, cosine similarity matching
- **Per-row scales**: FP16 scale factor per weight matrix row
- **10-bit packed indices**: efficient bitpacking for K=1024
- **zstd-22 compression**: final byte-level compression
- **Codebook overhead**: 1024 x 2 x FP16 x 60 layers = ~240KB (negligible)

## Artifact Size

- Parameters: 24,140,880 (24.1M)
- Model compressed: ~15.93 MB
- Code: ~68 KB
- **Total: ~16.0 MB (under 16,000,000 byte limit)**

## Also Explored (Negative Results)

| Technique | Result |
|-----------|--------|
| VQ G=4 K=256 (2 bits/param) | Catastrophic delta +0.16 |
| Sign-splitting (SSVQ) | Worse delta than full-vector VQ |
| K-means++ initialization | OOM (sequential init too expensive) |
| Normal distribution init | Slow convergence, worse raw quality |
| Residual VQ (2-level K=32+32) | Catastrophic (too few entries per level) |
| Straight-through estimator (STE) | Overhead hurts 8-min training |
| LoRA overparameterization | Step overhead kills training steps |
| Depth recurrence | Step overhead kills training steps |

## Files

- `train_gpt.py` -- self-contained training script
- `submission.json` -- leaderboard metadata
- `train.log` -- training log from 3-hour run on 3x RTX A6000
