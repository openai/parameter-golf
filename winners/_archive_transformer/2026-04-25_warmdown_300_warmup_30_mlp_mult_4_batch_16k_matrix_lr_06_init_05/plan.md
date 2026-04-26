# Experiment 0024_tied_embed_init_std_05_on_winner

Parent: canonical (env-var sibling of `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_16k_matrix_lr_06_init_02`)

## Question
0023 showed init scale 0.005 → 0.02 (4×) gives Δ=+0.011. Does pushing
further to 0.05 (10× baseline, 2.5× current) keep paying?

At init=0.05, tok_emb rows have norm sqrt(512) × 0.05 = 1.13 (≈1).
This is approaching trained-embedding magnitudes — larger init means
less relative gradient adjustment per step.

## Hypothesis [LIKELY]
Δ vs 0023 winner (2.19847) ≈ -0.005 to +0.010. Lean to small or zero
gain because 0.02 may already be near the init optimum.

## Change
`env.sh`:
- `LR_WARMUP_STEPS=30`
- `WARMDOWN_ITERS=300`
- `MLP_MULT=4`
- `TRAIN_BATCH_TOKENS=16384`
- `MATRIX_LR=0.06`
- `TIED_EMBED_INIT_STD=0.05`

## Disconfirming
- Δ ≤ +0.005: init has saturated at 0.02; stick there.
- Δ < -0.005: 0.05 overshoots; init has a real optimum near 0.02.
- Δ ≥ +0.010: keep paying; try 0.1 next.

## Notes from execution
<!-- Filled after the run. -->
