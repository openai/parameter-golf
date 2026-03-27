# 11L Int5 QAT + Score-First TTT — val_bpb 1.1326

**Score:** 1.13256182 BPB  
**Artifact:** 15.51 MB (16,265,723 bytes)  
**Training:** ~37 min on 8xH100 (2222s)  
**Author:** JoeProAI

---

## What Changed

Built on the PR #549 stack (LeakyReLU² + Muon). Key differences:

- **Int5 QAT** — weights quantized to [-15, 15] range (int5 stored as int8), per-row scale, 99.99984th percentile clipping. Tighter than int6, better compression.
- **Score-first TTT** — legal test-time training per PR #461 recipe. Score each chunk first, then adapt. AdamW optimizer on MLP-only params (up_proj, down_proj, gate_proj, scale). lr=0.0004, 1 epoch.
- **MLP_HIDDEN=1536** — reduced from 1792 to fit artifact under 16 MB with int5 compression.
- **15% weight pruning** — prune smallest weights before quantization for better zstd compression.
- **Bigram hash embedding** — 4096 buckets, 128-dim, added to token embeddings.
- **XSA on all 11 layers** — cross-layer shared attention across the full U-Net.
- **Warmdown 6000 steps** — longer QAT warmdown for better weight clustering near int5 boundaries.

## Architecture

```
11-layer U-Net GPT
  dim=512, heads=8, mlp_hidden=1536
  SwiGLU MLP
  XSA (cross-layer shared attention) on all layers
  U-Net skip connections (encoder layers 0-5, decoder layers 6-10)
  Bigram hash embedding (4096 buckets, 128-dim)
  FP16 embedding passthrough
```

## Quantization

```
Format: int5 per-row ([-15, 15] stored as int8 + float16 scale)
Clipping: 99.99984th percentile per row
Compression: zstd level 22
Pre-quant pruning: 15% smallest weights zeroed
```

## Training Config

```
optimizer: Muon (matrix) + AdamW (scalars)
matrix_lr: 0.025
muon_momentum: 0.95
warmdown_iters: 6000
wall_clock: 600s on 8xH100
```

## TTT Config

```
order: score-first (legal — chunk scored before any adaptation)
optimizer: AdamW
lr: 0.0004
epochs: 1
params: MLP-only (up_proj, down_proj, gate_proj, scale)
```

## Results

```
val_loss:  1.91228074
val_bpb:   1.13256182
artifact:  16,265,723 bytes (15.51 MB)
```
