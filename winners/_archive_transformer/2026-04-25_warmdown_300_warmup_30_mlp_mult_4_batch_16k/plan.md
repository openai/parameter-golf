# Experiment 0020_warmdown_300_warmup_30

Parent: canonical (env-var sibling of `winners/2026-04-25_warmdown_400_warmup_20_mlp_mult_4_batch_16k`)

## Question
0005 schedule (warmdown_600+warmup_10) gave Δ=+0.116 vs canonical.
0015 schedule (warmdown_400+warmup_20) gave Δ=+0.055 vs 0013.
Schedule push pattern has been monotonic but with diminishing returns.
Does another step (warmdown_300+warmup_30) keep paying?

Schedule values:
- step 0-29: warmup 0.033 → 1.0
- step 30: warmdown branch fires, lr_mul=(200-30)/300=0.567
- step 90: lr_mul=(200-90)/300=0.367
- step 200: 0.0
- avg lr_mul ≈ 0.318 (1.25× the 0015 schedule's 0.255)

Time at lr_mul ≥ 0.5: warmup steps 15-29 (15 steps) + warmdown step 30-50
(20 steps) = ~35 steps in elevated regime. Larger than 0015's window
(steps 10-19 + step 20 = ~10 steps). Higher NaN risk.

## Hypothesis [LIKELY]
Δ vs 0015 winner (2.25468) ≈ +0.010 to +0.025. Diminishing returns kick
in vs 0015's +0.055. If gain matches the 0008→0015 step (a single
schedule push gave +0.055), we'd see +0.020-0.030. If much smaller
(<+0.005), schedule has plateau'd.

## Change
`env.sh`:
- `LR_WARMUP_STEPS=30`
- `WARMDOWN_ITERS=300`
- `MLP_MULT=4`
- `TRAIN_BATCH_TOKENS=16384`

## Disconfirming
- NaN: 35 steps at lr_mul ≥ 0.5 is too many for batch=16k. Would back
  off to WARMDOWN_350_WARMUP_25 or scale MATRIX_LR down.
- Δ ≤ +0.005 (noise): schedule has plateau'd. Pivot to other axes (init,
  LR, optimizer).
- Δ < 0: more LR hurts at this batch size. Confirms 0018's lesson that
  LR-by-schedule scaling has a ceiling.
- Mode collapse (very low quant_tax + train_loss collapse like 0018):
  combined effect of bigger schedule + bigger batch = same dynamics that
  broke 0018.

## Notes from execution
<!-- Filled after the run. -->
