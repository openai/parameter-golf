# Hypothesis: QK_Gain_SLOT_Legal
Date: 2026-03-31
Track: neural
Parent: neural/2026-03-31_QK_Gain_SLOT/ (SLOT code, with enable_grad fix)

## What changes (ONE variable)
SLOT eval mode: original → Context-Only (legal variant)

The original SLOT (QK_Gain_SLOT leg) optimized the hidden-state delta using ALL
tokens in the window including the tokens being scored. This is a potential
causality violation — the same tokens are used both for optimization and scoring.
Those PRs (#1172, #1176) were ruled illegal in competition.

This leg implements **Context-Only SLOT**:
- Window 0 (ws==0): base model only — no delta (no prior context to optimize from)
- All other windows: optimize delta for `slot_steps` steps using ONLY positions
  `0..wlen-stride-1` (context tokens already scored in prior windows), then score
  positions `wlen-stride..wlen-1` (new tokens) under the optimized delta

Mathematically guaranteed causal: `hidden[t]` depends only on `tokens[0..t]`
(bigram hash uses t and t-1, attention is causally masked, norms are
position-independent). `hidden[0..wlen-stride-1]` physically cannot contain
information from `tokens[wlen-stride:]`.

## Why
Prior (ambiguous) SLOT showed -0.0085 BPB on sliding_bpb at 1200-step proxy.
The legal version optimizes on strictly past context — the gradient signal is
weaker (fewer target tokens per optimization step) but should still generalize
to new tokens if the delta is learning a useful hidden-space direction.
Signal of -0.003 to -0.005 would be meaningful and potentially submittable.

## Gate target (1-GPU, 1200 steps, SLOT_MAX_WINDOWS=512)
- sliding_bpb delta vs baseline: **< -0.003** (half the ambiguous version's signal)
- No regression on post_ema_bpb (training identical, eval-side only change)
- Clean paired run: baseline and slot_legal in same script, same pod, same seed
