# SwiGLU + EMA + Int5 Quantization + EBLS Findings (Non-Record)

**Author:** Robby Sneiderman ([@Robby955](https://github.com/Robby955))

**BPB:** 1.1679 (post-quantization, standard sliding window eval on 8xH100 SXM)

**Artifact:** 15,099,489 bytes (code: 53,443 + weights: 15,046,046)

Non-record submission combining SwiGLU MLP, EMA weight averaging, int5 quantization for all weight categories, and novel findings from Empirical Bayes Layer Sharing (EBLS) and test-time training (TTT) explorations.

## Results

| Metric | Value |
|--------|-------|
| Post-quant BPB (sliding, stride=64) | **1.1679** |
| Pre-quant BPB | 1.1657 |
| Steps | 5,116 (8xH100 SXM, 110ms/step) |
| Model params | 25,517,137 |
| Artifact size | 15.10 MB |

## What We Changed from the Base

Built on thwu1 PR #180 (which built on unnir PR #162):

1. **SwiGLU MLP** replacing ReLU-squared. `silu(W_gate @ x) * (W_up @ x)` with `swiglu_mult=2.0` gives the same parameter count as `mlp_mult=3.0` ReLU² but the gating mechanism provides better gradient flow.

2. **EMA** (decay=0.9985) replacing SWA. Exponential moving average during warmdown instead of discrete checkpoint averaging.

3. **Int5 quantization for all weights** with 5% magnitude pruning. Using int5 (clip_range=15) for all weight categories (MLP, attention, bigram) instead of mixed int5-MLP/int6-attention saves ~800KB with negligible quality impact, creating headroom for larger models. Compressed with zstd-22.

4. **TTT exploration** (negative result). Per-window AdamW adaptation at eval time (adapt MLP weights on prefix, score suffix, restore) produces worse BPB than no adaptation. At batch_size=1, gradient variance is too high for meaningful adaptation in 5-10 steps — the model is degraded rather than improved. See "TTT Finding" below.

## EBLS Exploration: Three Findings

We also explored Empirical Bayes Layer Sharing, a weight-sharing architecture where K shared blocks loop M times with per-virtual-layer LoRA deviations gated by learned shrinkage gammas:

```
W_effective[i] = W_shared + gamma_i * (A_i @ B_i)
gamma_i = sigmoid(logit_i), regularized by lambda * sum(gamma_i)
```

### Finding 1: MLP-vs-Attention Sharing Asymmetry

After training on 8xH100 SXM (4,572 steps), the learned gammas show:

| Component | Gamma Range | Interpretation |
|-----------|------------|----------------|
| MLP (all layers) | 0.0000 | Fully shared — identical computation across depth |
| Attention (layers 0-2) | 0.001-0.005 | Trace specialization in early layers only |
| Attention (layers 3-8) | 0.0000 | Fully shared |

MLP weights converge to exact sharing. The model discovers through gradient optimization that feedforward computation does not need to vary with depth under compression constraints. This connects to the XSA4 finding that shared attention works in late layers because attention patterns converge — our result extends this to MLP, showing the effect is even stronger for feedforward layers.

### Finding 2: LoRA Rank Threshold for Specialization

At rank 8, all gammas converge to ~0 (no specialization needed). At rank 16, gammas stabilize at 0.01-0.05 (partial sharing). The model rationally chooses not to deviate when deviation capacity is insufficient. This has implications for LoRA fine-tuning: if your rank is too low, the model may appear not to need adaptation when it simply can't express useful adaptation.

### Finding 3: Quantization Error Amplification in Depth-Recurrent Architectures

Shared weights quantized once but applied N times compound quantization noise through the residual stream. We observe a 0.19 BPB gap between `torch.compile` (fused kernels) and eager-mode evaluation in our depth-recurrent architecture — not from quantization but from floating-point numerical differences amplified across 15 passes through 5 shared blocks. This gap does not exist in standard (non-recurrent) architectures. Any architecture using weight sharing with depth recurrence (Universal Transformer, ALBERT-style) will exhibit amplified sensitivity to numerical precision.

## TTT Finding: Per-Window Adaptation is a Negative Result

Test-time training can be understood as posterior adaptation — the pretrained weights are the prior, TTT computes a MAP estimate conditioned on each eval context. However, our implementation revealed two critical issues:

**Batch data leak bug**: The initial batched TTT implementation processed 32 overlapping windows simultaneously, adapting on all prefixes then scoring all suffixes. With stride=64 and seq_len=2048, window j's prefix contains window i's scored suffix for j > i in the batch. This produced an impossible 0.463 BPB (below the Bayesian limit of ~0.95) — the model was literally training on data it then scored.

**Per-window TTT degrades quality**: After fixing to per-window processing (adapt on single prefix, score single suffix, restore), TTT consistently degraded BPB:
- LR=5e-4, 10 steps: **2.51 BPB** (catastrophic — LR too high for batch_size=1)
- LR=5e-5, 5 steps: **1.49 BPB** (still worse than 1.17 baseline)

The fundamental issue: at batch_size=1, the gradient from a single 1984-token prefix has high variance. Even with conservative learning rates, 5-10 Adam steps cannot find a meaningful adaptation direction. This is consistent with the James-Stein shrinkage interpretation — when estimation uncertainty (gradient variance) is high relative to the available signal, the optimal shrinkage factor is near 1.0 (i.e., no adaptation).

## Architecture Details

- 512-dim, 8 heads, 4 KV heads, SwiGLU (mult=2.0, hidden=1024)
- 10 transformer layers
- BigramHash(10,240 buckets, 128-dim), SmearGate
- Muon optimizer (WD=0.04, matrix_lr=0.02, momentum=0.99)
- EMA (decay=0.9985) during warmdown
- Int5 quantization (all weights), 5% magnitude pruning, zstd-22

## Reproducing

```bash
# 8xH100 SXM or NVL, 10-minute wallclock
SWIGLU_MULT=2.0 TTT_STEPS=10 PRUNE_FRAC=0.05 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- thwu1 PR #180 (base architecture, int5/int6, SWA, BigramHash)
- unnir PR #162 (10L, MLP 3x, SmearGate, MuonWD)
- felipe-parodi (EMA concept)
- sjp611 (AdamW TTT concept)

## Full Writeup

For the statistical foundations connecting James-Stein shrinkage to neural network parameter sharing, see the companion repository: [github.com/Robby955/parameter-golf-ebls](https://github.com/Robby955/parameter-golf-ebls)
