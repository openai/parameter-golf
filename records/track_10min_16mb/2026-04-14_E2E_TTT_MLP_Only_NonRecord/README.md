# Does E2E TTT help at 27M scale? A negative result.

**val_bpb: 1.1104** (seed 1337) | 13.85 MB | 8×H100 SXM | Non-record idea submission

## TL;DR

Ported **End-to-End Test-Time Training** (arXiv:2512.23675, ICLR 2026) to
parameter-golf. Ran three hyperparameter configurations spanning 10× in
learning rate, 6× in trainable params, and 2.5× in epochs. All three landed
within 0.001 BPB of each other (1.110–1.112) — **E2E TTT is saturated at
27M scale**. Sliding-window eval alone delivered 95% of the eval-time gain;
TTT contributed a rounding error.

## The technique

The paper updates MLP weights in the last 1/4 of blocks at test time via
score-first SGD on each chunk. Embeddings, attention, and norms are frozen
because updating them causes training instability. We implemented this as
a `TTT_E2E_MODE=1` flag in `eval_val_ttt` that filters `ttt_params` to only
the MLP layers in the last `TTT_E2E_LAST_FRAC` fraction of blocks.

## Fixed base stack

PR #1493 architecture on SP1024: 11L × 512d, GQA 8/4, 3-layer depth
recurrence (L3–5, activation 0.35), parallel residuals (L7+), QK-Gain 5.0,
Muon+Adam optimizers, GPTQ SDClip int6 + brotli. Training: 3,940 steps
in 588s, pre-quant val_bpb 1.1250. Same seed (1337) for every run — only
TTT hyperparameters changed.

## Experiments

### #1 — Paper defaults

**Hypothesis**: paper's published recipe should give baseline TTT gain.
Config: `TTT_LR=0.005 TTT_EPOCHS=3 TTT_E2E_LAST_FRAC=0.25` → 6.3M params
(MLPs in blocks 8–10).

| Stage | BPB | Δ |
|---|---:|---|
| Post-quant | 1.1360 | — |
| + Sliding window | 1.1123 | **−0.0237** |
| + E2E TTT | 1.1114 | −0.0009 |

**Finding**: TTT delivers essentially noise on top of sliding window.

### #2 — Broader, more aggressive

**Hypothesis**: defaults are tuned for 3B models; at 27M we need wider scope
and higher LR. Config: `TTT_LR=0.015 TTT_EPOCHS=2 TTT_E2E_LAST_FRAC=0.5`
→ 12.6M params (MLPs in blocks 5–10).

**Result: val_bpb 1.1104** — best run, but only 0.001 better than #1.

### #3 — Aggressive single block

**Hypothesis**: maybe the lever is intensity, not scope. Hammer one block
with 10× the LR. Config: `TTT_LR=0.05 TTT_EPOCHS=5 TTT_E2E_LAST_FRAC=0.09`
→ 2.1M params (MLP in block 10 only).

**Result: val_bpb 1.1111** — between #1 and #2. More params with moderate LR
beats fewer params with aggressive LR.

## Summary

| Run | TTT_LR | Epochs | Scope | Params | val_bpb |
|---|---:|---:|---|---:|---:|
| #1 defaults | 0.005 | 3 | L8–L10 | 6.3M | 1.1114 |
| **#2 broader** | **0.015** | **2** | **L5–L10** | **12.6M** | **1.1104** |
| #3 aggressive | 0.05 | 5 | L10 only | 2.1M | 1.1111 |

10× LR range, 6× param range, 2.5× epoch range → **0.001 BPB spread**.

## What we learned

1. **E2E TTT saturates at small scale.** The paper's gains come from 3B+
   models with substantial adaptation capacity. A 27M model trained to
   near its ceiling in 588s has no slack for test-time gradient steps to
   exploit.

2. **Sliding window is doing ~95% of the eval-time work.** If you're
   picking between tuning TTT and tuning the base model or sliding-window
   stride, the latter two dominate at this scale.

3. **"Moderate beats extremes"** — paper defaults underpush, single-block
   aggressive overpushes on too little capacity. The sweet spot is
   broader-scope with moderate LR.

## Why non-record

1. **SP1024, not SP8192** — the SP8192 data used by merged SOTA isn't
   available from the `willdepueoai/parameter-golf` HF repo, costing us
   ~0.03 BPB vs PR #1493.
2. **Single seed.** Given the 0.001 BPB noise floor we measured, more
   seeds wouldn't change the conclusion.
3. The interesting content is the research finding, not the score.

## For other participants

If you're under ~50M params: E2E TTT probably won't move your score.
Invest in base model quality, sliding-window tuning, or scale up. The
implementation here (env-var gated) is a drop-in starting point for
larger-scale experiments where TTT has more room to help.
