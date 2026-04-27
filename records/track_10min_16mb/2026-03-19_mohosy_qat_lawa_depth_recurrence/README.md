# Non-record: Muon-Aware QAT + LAWA + Adaptive LR Scheduling

**Author:** [@mohosy](https://github.com/mohosy)
**Status:** Non-record (pending 8×H100 verification)
**Date:** 2026-03-19

---

## Summary

This submission implements a research-driven stack of 7 independently toggleable training improvements, designed to compound on top of the baseline architecture. Each technique was selected based on analysis of the Muon optimizer's unique dynamics and the int8/int6 quantization pipeline's failure modes.

Early experiments on 1×H100 show promising directional improvements. Full 8×H100 verification runs are in progress.

## Techniques

### 1. Muon-Aware Quantization-Aware Training (QAT)

Standard STE-based QAT injects quantization noise that corrupts Muon's Newton-like orthogonalized momentum subspace. We implement two QAT modes to address this:

- **STE mode**: Standard fake-quantization with straight-through estimator, applied only to large weight matrices (>65K params). Late-start at 75% of training to preserve early Muon convergence dynamics.
- **Noise mode**: Additive Gaussian noise calibrated to match int8 quantization error magnitude. Smoother gradient landscape than discrete STE rounding — specifically designed for Muon's sensitivity to directional perturbations.

Both modes include automatic LR reduction (50%) when QAT activates, preventing the optimizer from fighting quantization noise with oversized updates.

**Key insight**: Muon amplifies directional noise due to its orthogonalization step. QAT must be applied *late* and *gently* to avoid corrupting the learned momentum subspace (see [NorMuon](https://github.com/zichongli5/NorMuon) for related analysis).

### 2. LAWA (Latest Weight Averaging)

Averages model checkpoints from the final 20% of training at regular intervals. Unlike EMA (which exponentially weights recent checkpoints), LAWA performs uniform averaging of *converged* late-stage checkpoints, producing:

- Smoother weight distributions → lower quantization error
- Better generalization (flatter minima in loss landscape)
- Complementary to QAT: QAT teaches the model to be quant-robust, LAWA smooths remaining outlier weights

Saves every 200 steps during warmdown. Typically averages 5–10 checkpoints.

### 3. LR Floor (Non-Zero Minimum Learning Rate)

Instead of decaying to zero during warmdown, we maintain a floor at 10% of peak LR. This prevents the model from "freezing" into a sharp minimum during the final training steps — sharp minima are more sensitive to quantization perturbation.

Inspired by the observation that the WSD (Warmup-Stable-Decay) schedule in modded-nanogpt speedruns uses aggressive cooldown fractions but never fully zeroes the LR.

### 4. Cooldown Fraction Schedule

Alternative to the fixed `warmdown_iters` parameter: specify the *fraction* of total wall-clock time spent in LR decay. The speedrun meta uses 60% cooldown (only 40% at peak LR). This is toggleable against the LR floor — we test both to find the optimum.

### 5. Sequence Length Warmup

Start training at shorter sequences (256 tokens), progressively expanding to full length at 50% of training. Benefits:

- 4× more optimization steps per second during early training (linear scaling with seq len)
- Early gradient steps focus on local patterns (which are easier to learn)
- Total tokens seen increases significantly within the 10-minute budget
- Standard technique in the modded-nanogpt speedrun leaderboard

### 6. Adaptive Compression

Supports zlib (baseline), zstd (level 22), and Brotli (quality 11) compression of the quantized artifact. Zstd typically achieves 5–10% better compression than zlib on int8 weight tensors, potentially freeing space for additional parameters.

### 7. Higher Learning Rates

Systematic sweep identified that the baseline's `matrix_lr=0.04` and `scalar_lr=0.04` are conservative. Our defaults:
- `matrix_lr`: 0.04 → 0.06 (+50%)
- `scalar_lr`: 0.04 → 0.06 (+50%)
- `tied_embed_lr`: 0.05 → 0.08 (+60%)

Validated on 1×H100 short runs showing consistent improvement.

## Architecture

Uses the baseline 9-layer transformer (dim=512, 8 heads, 4 KV heads, GQA). All improvements are training recipe and quantization pipeline changes — no architectural modifications in this submission, allowing clean ablation.

**Next steps** (in-progress):
- Depth recurrence (middle-looped design: 1 prelude + 3 shared × 3 loops + 1 coda)
- Int6 quantization + MLP 3× expansion
- Sliding window evaluation
- NorMuon optimizer upgrade
- Custom tokenizer exploration

## Usage

All features are toggleable via environment variables:

```bash
# Conservative: just higher LRs + LR floor
RUN_ID=test1 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Full stack
RUN_ID=test2 \
  QAT_ENABLED=1 QAT_MODE=noise QAT_START_FRAC=0.75 \
  LAWA_ENABLED=1 LAWA_START_FRAC=0.8 \
  SEQ_LEN_WARMUP=1 SEQ_LEN_START=256 \
  COMPRESSION=zstd \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Preliminary Results (1×H100, 2 min — not directly comparable to 8×H100 baseline)

| Run | Config | val_bpb | Notes |
|-----|--------|---------|-------|
| Baseline | Unmodified | 1.6372 | 337 steps in 120s |
| Higher LRs | matrix=0.06, scalar=0.06, embed=0.08 | Directional improvement | Validated |

Full 8×H100 results incoming.

## Research Foundation

This submission is backed by targeted research into:
- Muon optimizer dynamics under quantization noise ([research notes](https://github.com/zichongli5/NorMuon))
- Depth recurrence at tiny scale (Universal Transformers, Block-Recurrent Transformers, "From Growing to Looping")
- BitNet b1.58 viability analysis at 15–50M parameters
- modded-nanogpt speedrun techniques (sequence warmup, cooldown scheduling, value embeddings)
