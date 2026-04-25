# Experiment 0005_warmdown_600_warmup_10

Parent: canonical

## Question
0004 (WARMDOWN_ITERS=600, no warmup) showed the issue: a single optimizer
step at lr_mul=0.333 from cold-start tok_emb is enough to overshoot
(step 2 train_loss spiked from 6.94 → 8.40, recovered slowly). Adding
`LR_WARMUP_STEPS=10` ramps `lr_mul` linearly 0.1 → 1.0 over the first
10 steps, then the warmdown branch takes over with `lr_mul=(200−step)/600`,
which equals 0.317 at step 10 and decays to 0 at step 200.

Trace of the schedule (lr_mul values):
- step 0:  0.1
- step 5:  0.6
- step 9:  1.0   ← peak (one step at full canonical)
- step 10: 0.317 (warmdown branch kicks in; discontinuous drop)
- step 100: 0.167 (= the *peak* of the canonical baseline schedule)
- step 200: 0.0

Average lr_mul ≈ 0.178 over the 200 steps (vs baseline 0.083) — 2.14× more
total LR delivered to the model. The one-step spike at lr_mul=1.0 is brief;
the journal-recorded NaN happened with lr_mul=1.0 sustained over ~160
steps (WARMDOWN_ITERS=40, no warmup). Single-step exposure should be
recoverable. The "elevated regime" between steps 5 and ~50 is the new
territory the model has never been trained in.

## Hypothesis [LIKELY]
Δ ≈ +0.030 to +0.080. The model is currently severely under-trained per
the 0002 / 0003 results. Doubling effective LR while staying broadly
within the journal-validated bf16 stability envelope should give a real
gain. Predicting big because it's the first time we've actually tested
"more LR" cleanly.

## Change
`env.sh`:
- `export LR_WARMUP_STEPS=10`
- `export WARMDOWN_ITERS=600`

No code edits.

## Disconfirming
- **NaN around steps 5-15**: the brief lr_mul=1.0 spike is enough to
  destabilize tok_emb / skip_weights even with prior warmup. Would
  motivate trying `LR_WARMUP_STEPS=20 + WARMDOWN_ITERS=400` or scaling
  per-optimizer LRs down.
- **Step 2 spike (loss > step 1)**: warmup is too steep; even 0.1 → 0.2
  at step 1 is enough to throw tok_emb. Would motivate scaling
  WARMUP_STEPS up to 20 or 30.
- **Δ ≤ +0.005 (noise)**: more LR doesn't help — surprising, would
  shift focus to non-LR axes (init scale, longer schedule, depth).
- **Δ ≥ +0.080 (very large)**: confirm with SEED=42 re-run before
  promoting (per noise-floor section).

## Notes from execution
<!-- Filled after the run. -->
