# SwiGLU + EMA + AdamW TTT + EBLS Findings (Non-Record)

**Author:** Robby Sneiderman ([@Robby955](https://github.com/Robby955))

**BPB:** 1.1746 (post-quantization, standard sliding window eval on H100 NVL)

**Artifact:** 15,902,348 bytes (code: 53,610 + weights: 15,848,738)

Non-record submission combining SwiGLU MLP, EMA weight averaging, eval-time AdamW TTT, and novel findings from an Empirical Bayes Layer Sharing (EBLS) exploration.

## Results

| Metric | Value |
|--------|-------|
| Post-quant BPB (sliding, stride=64) | **1.1746** |
| Pre-quant BPB | 1.1822 |
| Steps | 3,547 (H100 NVL, 170ms/step) |
| Estimated SXM steps | ~6,100 (91ms/step, ~1.15 BPB) |
| Model params | 25,517,137 |
| Artifact size | 15.90 MB |

Note: Run on 8xH100 NVL (170ms/step) rather than 8xH100 SXM (91ms/step), resulting in 3,547 steps instead of ~6,100. On SXM, pre-quant BPB would be ~1.15-1.16.

## What We Changed from the Base

Built on thwu1 PR #180 (which built on unnir PR #162):

1. **SwiGLU MLP** replacing ReLU-squared. `silu(W_gate @ x) * (W_up @ x)` with `swiglu_mult=2.0` gives the same parameter count as `mlp_mult=3.0` ReLU² but the gating mechanism provides better gradient flow.

2. **EMA** (decay=0.9985) replacing SWA. Exponential moving average during warmdown instead of discrete checkpoint averaging.

3. **Eval-time AdamW TTT** on MLP weights. Per sliding-window adaptation: adapt MLP weights on the context prefix via 10 steps of AdamW, score the suffix, restore weights. Correctly implemented as test-time inference (not pre-quant training on validation data).

4. **Mixed int5/int6 quantization** with 5% magnitude pruning. Int5 for MLP weights, int6 for attention, zstd-22 compression.

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

## Statistical Perspective on TTT

Test-time training can be understood as posterior adaptation. The pretrained weights are the prior distribution over model parameters; TTT computes a MAP estimate conditioned on each eval context. AdamW's per-parameter adaptive learning rates provide implicit shrinkage toward the prior — parameters with high gradient variance receive smaller effective updates (more shrinkage), while parameters with consistent gradients receive larger updates (less shrinkage). This mirrors the James-Stein principle that shrinkage should be proportional to estimation uncertainty. The weight decay term in AdamW acts as the prior precision, pulling adapted weights back toward the pretrained values when the context-specific evidence is insufficient to justify deviation.

## Architecture Details

- 512-dim, 8 heads, 4 KV heads, SwiGLU (mult=2.0, hidden=1024)
- 10 transformer layers
- BigramHash(10,240 buckets, 128-dim), SmearGate
- Muon optimizer (WD=0.04, matrix_lr=0.02, momentum=0.99)
- EMA (decay=0.9985) during warmdown
- Mixed int5/int6 quantization, 5% magnitude pruning, zstd-22

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
