# Experiment 0015_warmdown_400_warmup_20

Parent: canonical (env-var sibling of `winners/2026-04-25_warmdown_600_warmup_10_mlp_mult_4_batch_16k`)

## Question
0013 winner uses `WARMDOWN_ITERS=600 + LR_WARMUP_STEPS=10`, giving avg
lr_mul ≈ 0.178. Bigger batches (16k) provide more accurate per-step
gradients, which classically tolerate (and benefit from) higher LR.
Does pushing the schedule further help?

`WARMDOWN_ITERS=400 + LR_WARMUP_STEPS=20` schedule:
- step 0-19: warmup 0.05 → 1.0 (linear over 20 steps)
- step 20: warmdown branch fires, lr_mul=(200-20)/400=0.45 (drops from 1.0)
- step 200: 0.0
- avg lr_mul ≈ 0.255 (1.43× the 0013 schedule)

The peak lr_mul=1.0 at step 19 is the same regime that NaN'd at sustained
160 steps with WARMDOWN_ITERS=40+no warmup; here it's one step. Time
spent at lr_mul ≥ 0.5 is steps 10-19 (10 steps in warmup) + step 20 at
0.45 — about 10 steps in the elevated regime.

Per the 0005 finding, bigger batches should be more LR-tolerant; the
combination should be safe.

## Hypothesis [LIKELY]
Δ vs 0013 winner (2.30956) ≈ +0.020 to +0.050. The bigger batch +
more aggressive schedule combination unlocks more learning per step.
Confidence is `[LIKELY]` because of the brief lr_mul=1.0 region — could
spike-recover similarly to 0005 (which gained +0.116 despite the spike).

## Change
`env.sh`:
- `LR_WARMUP_STEPS=20` (was 10)
- `WARMDOWN_ITERS=400` (was 600)
- `MLP_MULT=4`
- `TRAIN_BATCH_TOKENS=16384`

## Disconfirming
- NaN around steps 10-20: peak + sustained high LR pushes tok_emb past
  bf16 dynamic range. Would back off to WARMDOWN_500 or warmup=30.
- Step 2 spike (loss > step 1) much worse than 0005's 7.06: the smaller
  warmup-step magnitudes amplify too much. Would revisit.
- Δ ≤ +0.005 vs 0013: the 0013 schedule was already near-optimal at this
  batch size. Schedule push won't help further; pivot to other axes.
- Δ ≤ −0.010 (worse): more LR hurts; the 0013 schedule was actually
  *over*-tuned, and we'd want to test smaller schedules.

## Notes from execution
- val_bpb_post=2.25468 vs 0013 winner 2.30956 → **Δ=+0.0549** (above the
  +0.050 "suspicious-large" threshold; SEED=42 confirm required).
- pre-quant Δ vs 0013 = +0.0540 — almost all the gain is from training,
  not quantization. Quant tax 0.0028 (cleaner than 0013's 0.0037).
- artifact 12.919 MB (vs 0013's 11.753 — about +1.2 MB extra; the
  better-trained model has more weight structure that zlib compresses
  less). Still 3+ MB cap headroom.
- step_avg 2469 ms (vs 0013's 2578 ms; basically same).
- **Trajectory is much cleaner than 0013/0008**: at step 9 train_loss=5.85
  (vs 0008's 7.06 spike). The 20-step warmup ramps lr_mul smoothly to 1.0
  at step 19 *without* having visited 1.0 cumulatively — this is a much
  better-conditioned warmup. Then warmdown branch jumps to 0.45 at step 20
  (downward step is safe), and decays linearly. By step 80 train_loss=4.20,
  already below 0008's *final* 4.42. By step 200 the descent has clearly
  surpassed the prior winner.
- **Status: parked**, pending 0016 SEED=42 confirm. Δ=+0.055 is in the
  "suspicious large" zone but the cross-seed variance for these big-batch
  configs has been ~0.0024 (per 0013/0014). Expect 0016 to land in
  [2.252, 2.258] if the gain is real.
