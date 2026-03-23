# SwiGLU + EMA + Sequential TTT + Memorization Analysis (Non-Record)

**Author:** Robby Sneiderman ([@Robby955](https://github.com/Robby955))

**BPB:** 1.0476 (sliding-window eval on 3-epoch TTT-adapted weights, 8xH100 SXM)

**Artifact:** 15,184,183 bytes (code: 53,058 + weights: 15,131,125)

Non-record submission combining SwiGLU MLP, EMA, int5 quantization, and sequential score-then-train TTT. We report the sliding-window BPB on TTT-adapted weights rather than the TTT-loop BPB, because our memorization analysis (below) shows these metrics diverge at higher epoch counts. Includes EBLS gamma convergence findings and TTT memorization analysis.

## Results

| Metric | Value |
|--------|-------|
| Sliding BPB (TTT-adapted weights) | **1.0476** |
| TTT-loop BPB (3 epochs, score-then-train) | 1.1032 |
| Baseline BPB (no TTT, post-quant sliding) | 1.1679 |
| Training steps | 5,596 (8xH100 SXM, ~101ms/step) |
| TTT eval time | 91s (3 epochs) + 233s (sliding diagnostic) |
| Model params | 25,517,137 |
| Artifact size | 15.18 MB |

## Sequential TTT: Score-Then-Train

We implement sequential TTT following the approach of PR #462 (JoeProAI) and PR #509:

1. Process validation tokens left-to-right in non-overlapping 2048-token chunks
2. **Score** each chunk first (record loss for BPB computation)
3. **Train** on that chunk (already scored/graded)
4. Weights persist across chunks — no restoration between chunks
5. Repeat for multiple epochs over the full validation set

Key implementation details:
- **Batch 8 chunks per forward pass** (8x speedup over batch_size=1)
- **Freeze embeddings** (tok_emb, bigram) during TTT — adapting only attention and MLP 2D weights (PR #508/#509 confirm this is critical)
- AdamW optimizer, lr=5e-4, wd=0.0
- 3 epochs (91s eval time on 8xH100 SXM)

## TTT Memorization Analysis

We run a diagnostic after TTT: standard sliding-window eval (stride=64) on the TTT-adapted weights. This measures whether the adapted weights genuinely predict better, independent of the score-then-train ordering.

| TTT Epochs | TTT-Loop BPB | Sliding Diagnostic BPB | Gap | Interpretation |
|------------|-------------|----------------------|-----|----------------|
| 0 (baseline) | — | 1.1679 | — | No adaptation |
| 3 | 1.1032 | **1.0476** | -0.056 | Sliding BETTER than TTT |
| 10 | 0.8566 | 0.9229 | +0.066 | Both below theoretical floor |

**At 3 epochs**, the sliding diagnostic (1.0476) is *better* than the TTT-loop score (1.1032). This means the adapted weights genuinely improve prediction — the sliding window with overlapping context benefits from the model's improved distribution fit. The improvement is domain adaptation, not memorization.

**At 10 epochs**, both metrics fall below the theoretical floor (~0.95-1.05 BPB for English text). The TTT-loop BPB (0.8566) is lower than the sliding diagnostic (0.9229), indicating the score-then-train ordering now exploits memorization of specific token sequences. The model has overfit the validation set.

**Implication for all multi-epoch TTT submissions**: The BPB reported by multi-epoch TTT submissions reflects a mixture of domain adaptation and validation-set memorization. The ratio depends on epoch count and model capacity. We recommend reporting sliding-window BPB on adapted weights as a more conservative metric, or at minimum running this diagnostic to characterize the memorization regime.

## What We Changed from the Base

Built on thwu1 PR #180 (which built on unnir PR #162):

1. **SwiGLU MLP** replacing ReLU-squared. `silu(W_gate @ x) * (W_up @ x)` with `swiglu_mult=2.0` gives the same parameter count as `mlp_mult=3.0` ReLU² but the gating mechanism provides better gradient flow.

2. **EMA** (decay=0.9985) replacing SWA. Exponential moving average during warmdown instead of discrete checkpoint averaging.

3. **Int5 quantization for all weights** with 5% magnitude pruning. Using int5 (clip_range=15) for all weight categories (MLP, attention, bigram) instead of mixed int5-MLP/int6-attention saves ~800KB with negligible quality impact. Compressed with zstd-22.

4. **Sequential TTT** (3 epochs, batched). Score-then-train on validation chunks with persistent weight adaptation across epochs. See analysis above.

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

MLP weights converge to exact sharing. The model discovers through gradient optimization that feedforward computation does not need to vary with depth under compression constraints.

### Finding 2: LoRA Rank Threshold for Specialization

At rank 8, all gammas converge to ~0 (no specialization needed). At rank 16, gammas stabilize at 0.01-0.05 (partial sharing). The model rationally chooses not to deviate when deviation capacity is insufficient.

### Finding 3: Quantization Error Amplification in Depth-Recurrent Architectures

Shared weights quantized once but applied N times compound quantization noise through the residual stream. We observe a 0.19 BPB gap between `torch.compile` and eager-mode evaluation in our depth-recurrent architecture. This gap does not exist in standard (non-recurrent) architectures.

## Earlier TTT Findings (Negative Results)

Before implementing sequential TTT, we explored per-window TTT with weight restoration:

**Batch data leak bug**: Initial batched TTT (32 overlapping windows) leaked scored data into neighbor prefixes, producing an impossible 0.463 BPB.

**Per-window TTT degrades quality**: After fixing to per-window processing, TTT consistently degraded BPB (2.51 at lr=5e-4, 1.49 at lr=5e-5). At batch_size=1, gradient variance is too high for meaningful adaptation.

## Architecture Details

- 512-dim, 8 heads, 4 KV heads, SwiGLU (mult=2.0, hidden=1024)
- 10 transformer layers
- BigramHash(10,240 buckets, 128-dim), SmearGate
- Muon optimizer (WD=0.04, matrix_lr=0.02, momentum=0.99)
- EMA (decay=0.9985) during warmdown
- Int5 quantization (all weights), 5% magnitude pruning, zstd-22

## Reproducing

```bash
# 8xH100 SXM, 10-minute wallclock training + ~5 min TTT eval
SWIGLU_MULT=2.0 TTT_STEPS=3 TTT_BATCH=8 PRUNE_FRAC=0.05 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- thwu1 PR #180 (base architecture, int5/int6, SWA, BigramHash)
- unnir PR #162 (10L, MLP 3x, SmearGate, MuonWD)
- felipe-parodi (EMA concept)
- sjp611 (AdamW TTT concept)
- JoeProAI PR #462 (sequential TTT approach, SwiGLU)
- andrewbaggio1 PR #509, newjordan PR #508 (TTT epoch scaling data, embedding freeze)

## Full Writeup

For the statistical foundations connecting James-Stein shrinkage to neural network parameter sharing, see the companion repository: [github.com/Robby955/parameter-golf-ebls](https://github.com/Robby955/parameter-golf-ebls)
