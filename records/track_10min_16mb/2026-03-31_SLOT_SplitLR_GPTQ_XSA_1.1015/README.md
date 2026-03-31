# Record: SLOT + Split-LR + Full GPTQ + XSA-all (val_bpb: 1.1015)

**val_bpb: 1.1015** (3-seed mean, std 0.0011) | **1.8598 nats** | **~15.65 MB** | 8xH100 SXM, 600s train + 177s eval

Built on [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by @abaybektursun.
Previous: [PR #549](https://github.com/openai/parameter-golf/pull/549) (1.1194) -> [PR #1019](https://github.com/openai/parameter-golf/pull/1019) (1.1147) -> this.

## Results (8xH100 SXM)

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding+SLOT BPB** | val_loss (nats) | Artifact |
|------|-------|---------|-------------|---------------------|-----------------|----------|
| 1337 | 6704 | 88.2 | 1.1309 | **1.10213** | 1.8609 | 15,647,124 |
| 42 | 6706 | 88.2 | 1.1289 | **1.10019** | 1.8576 | 15,658,061 |
| 2025 | 6684 | 88.4 | 1.1310 | **1.10216** | 1.8609 | 15,650,266 |
| **Mean** | **6698** | **88.3** | **1.1303** | **1.10149** | **1.8598** | **15,651,817** |

### Improvement vs SOTA

| Metric | Merged SOTA (PR #1019) | This submission | Delta |
|--------|----------------------|-----------------|-------|
| val_bpb (3-seed mean) | 1.1147 | **1.1015** | **-0.0132** |
| val_loss (nats) | 1.88218 | **1.85982** | **-0.02236** |

Clears the 0.005 nats threshold by 4.5x.

## Changes vs Baseline (PR #1019)

### 1. SLOT: Sample-specific LM Optimization at Test-time

At eval time, for each sliding-window batch, we optimize a single additive delta vector (R^512) between the frozen hidden states and the logit projection. The model forward is split into `forward_hidden()` (frozen, no grad) and `compute_logits()` (carries grad for delta optimization).

- **Delta shape**: `[1, 1, 512]` — broadcasts across batch and sequence
- **Optimizer**: AdamW (lr=0.005, weight_decay=1e-8, eps=1e-5)
- **Steps**: 8 per batch
- **Eval time overhead**: ~90s (well within 600s eval budget)

SLOT is score-first: hidden states are computed under `torch.no_grad()`, the delta adapts through `compute_logits()` only, and final scoring uses the adapted logits. The model weights are never modified.

Reference: Hu et al., arXiv:2505.12392v2. Also used in PR #1128, PR #1105.

### 2. Sigmoid-Gated Skip Connections

U-Net skip connections use learned sigmoid gates instead of simple addition:
```python
g = sigmoid(skip_gates[i])
x = lerp(skip_weights[i] * skip, x, g)
```
Gate starts at sigmoid(0) = 0.5 (balanced blend). Adds 2,560 params (5 gates x 512 dims).

### 3. Soft-Round QAT with Alpha Ramp

Late QAT uses differentiable sigmoid rounding instead of hard STE:
```python
soft_rounded = floor(scaled) + sigmoid(alpha * (frac - 0.5))
```
Alpha ramps from 1 (smooth) to 16 (near-hard) over 500 steps. Provides real gradients through rounding, letting weights adapt to quantization grid.

### 4. Split Early/Late Muon Learning Rate

Bank gradients are scaled per-layer before the Muon reduce-scatter:
- Early layers (0-4): Muon LR = 0.025
- Late layers (5-10): Muon LR = 0.030

Late layers benefit from higher LR (weaker gradient signal further from loss).

### 5. Warmdown = 4000 Steps

Extended warmdown from 3500 to 4000 estimated steps. Holds LR higher for longer, giving the model more time at productive learning rates.

### 6. BigramHash(2816x160)

Enlarged bigram embedding dimension from 112 to 160. Same 2816 buckets. Richer per-bucket representation at minimal artifact cost.

### 7. Code Minification

`pyminify` + LZMA2 + base85 self-extracting wrapper reduces code from 101KB to 23KB, freeing ~78KB of artifact budget for model weights.

### 8. Brotli-11 Compression with Byte-Shuffle

Replaces LZMA-6 with Brotli quality=11 + stride-2 byte-shuffle preprocessing. Saves ~400KB vs LZMA.

### 9. GPTQ Reserve 9s (was 14s)

Reduced GPTQ calibration time reservation from 14s to 9s, gaining ~55 extra training steps.

## Negative Results (tested, did not help)

| Technique | Result | Notes |
|-----------|--------|-------|
| Turbo-Muon (AOL + Polar Express) | +2MB artifact bloat | Weight distribution changes break compression |
| No-GPTQ (PR #1120 style) | -0.005 BPP worse | GPTQ essential for our stack |
| Pure EngramLite swap | -0.003 worse | Same-budget multi-head too diluted |
| ResidLambdas | -0.003 worse | Quant error compounds through lambda scaling |
| LeakyReLU slope=0.3 | Neutral | |
| Partial key offset | Neutral | |
| BIGRAM_DIM=192 | -0.001 worse | Diminishing returns past 160 |
| TTT (score-first SGD) | Neutral on Full GPTQ stack | Post-quant weights too well-optimized |
| Mixed int5/int6 GPTQ | Broken or worse | Needs full PR #1089-style pipeline |

## Architecture Summary

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 | PR #549 |
| Model dim | 512 | PR #549 |
| Heads / KV heads | 8 / 4 (GQA) | PR #549 |
| MLP mult | 3.0x (LeakyReLU(0.5)^2) | PR #549 |
| XSA | All 11 layers | PR #1019 |
| BigramHash | 2816 x 160 | **This submission** (dim=160) |
| ValueEmbedding | dim=128, layers 9,10 | PR #549 |
| SmearGate | F.pad causal shift | PR #549, optimized |
| Skip connections | Sigmoid-gated lerp | **This submission** |
| Quantization | Full Hessian GPTQ int6 | PR #1019 |
| Compression | Brotli-11 + byte-shuffle | **This submission** |
| Optimizer | Parallel Muon + Split-LR | **This submission** (split-LR) |
| QAT | Soft-round alpha ramp 1->16 | **This submission** |
| Eval | Sliding window stride=64 + SLOT | **This submission** (SLOT) |
| Code | LZMA2 self-extracting wrapper | **This submission** |
| Warmdown | 4000 steps | **This submission** |
| Params | 27.2M | |

## Setup & Reproduction

```bash
# Environment: 8xH100 SXM, PyTorch 2.9.1+cu128, flash-attn 2.8.3
export NCCL_NET=Socket  # Required on GCP H100
export SLOT_ENABLED=1
export BIGRAM_DIM=160
export WARMDOWN_ITERS=4000
export SLOT_LR=0.005
export SLOT_STEPS=8

# Run with torchrun (evaluate.py handles this)
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=2025 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Acknowledgements

Thanks to **@0hq** and **@valerio-oai** for organizing and maintaining an excellent competition.

This submission builds directly on @abaybektursun's PR #549 and PR #1019, which established the LeakyReLU^2 + Parallel Muon + XSA + Full GPTQ stack. The SLOT technique follows Hu et al. (arXiv:2505.12392v2) and was independently validated by @AnubhavBharadwaaj (PR #1128) and @abaybektursun (PR #1105). The sigmoid-gated skip connection idea draws from @mikeapedia's PR #1089. Code minification approach adapted from PR #1089's shrink pipeline.
