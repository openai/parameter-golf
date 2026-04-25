# Experiment 0021_matrix_lr_06_on_winner

Parent: canonical (env-var sibling of `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_16k`)

## Question
All schedule pushes have effectively raised the avg `lr_mul` (0.083 →
0.318) at fixed `MATRIX_LR=0.04`. Is `MATRIX_LR` itself a separate lever
that pays at batch=16k? Test 1.5× scaling (0.04 → 0.06) with the 0020
schedule. Effective peak LR becomes 0.06 × 1.0 = 0.06; effective avg
becomes 0.06 × 0.318 = 0.019 (vs current 0.0127 at MATRIX_LR=0.04).

## Hypothesis [CONJECTURE]
Δ vs 0020 winner (2.22595) ≈ 0 to +0.020. Ambiguous because:
- Schedule push at fixed LR has been working — suggests headroom for
  more effective LR.
- 0018 batch=32k mode-collapsed at the same MATRIX_LR=0.04 + lr_mul=1.0
  peak — that was a bigger-batch issue, not a per-LR issue, but it shows
  there *is* an instability ceiling.
- LR scaling ≠ schedule scaling: schedule increases avg LR but keeps the
  peak shape; raw LR scaling pushes both peak and avg.

Slight lean to positive Δ because schedule push has paid off; if avg LR
is the lever, raw scaling would too.

## Change
`env.sh`:
- `LR_WARMUP_STEPS=30`
- `WARMDOWN_ITERS=300`
- `MLP_MULT=4`
- `TRAIN_BATCH_TOKENS=16384`
- `MATRIX_LR=0.06`

Note: not scaling TIED_EMBED_LR (stays 0.05) or SCALAR_LR (stays 0.04)
to isolate the matrix-LR effect.

## Disconfirming
- Δ ≤ +0.005 (noise): schedule was the LR knob; raw scaling at the same
  shape doesn't help.
- Δ < 0: more LR hurts at this batch size. Suggests current LR is at
  the optimum, schedule push is finding edge by changing shape not
  magnitude.
- NaN around peak: MATRIX_LR=0.06 × lr_mul=1.0 = 0.06 effective peak; if
  this NaN's, the 0020 schedule was already near the LR ceiling.
- Mode collapse like 0018: very low quant_tax and divergent train/val
  losses. Same gradient-variance dynamic at higher LR.

## Notes from execution
<!-- Filled after the run. -->
