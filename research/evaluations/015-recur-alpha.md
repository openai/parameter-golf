# Evaluation — Spec 015 (Recur-Alpha, α=0 init)

**Run dir:** `runs/015-recur-alpha/seed_42/`
**Commit:** `a9aa141` on `exp/recur-alpha`
**Baseline:** spec 008 `runs/008-1736-reproduction/seed_42/final.json` (val_bpb 1.0697 endpoint, 1.1110 @ step 4000)
**Eval date:** 2026-04-21

## Result

| metric | spec 008 | spec 015 | Δ |
|---|---|---|---|
| endpoint val_bpb (screening mode) | 1.0697 | **1.0696** | −0.0001 |
| matched-step val_bpb @ 4000 | 1.1110 | **1.1078** | **−0.0032** |
| stopping_early step | 4828 | 4761 | −67 (JP hardware variance) |
| training wallclock | 596 s | 596 s | = |

## Noise/signal judgment

**Endpoint Δ = −0.0001: null bucket.** This is dominated by the 67-step deficit from JP pod hardware variance (~1.4% throughput shortfall vs spec 008's pod). At the same wallclock cap, 015 got 67 fewer gradient steps.

**Matched-step Δ @ 4000 = −0.0032: real signal.** This removes the hardware variance by measuring both runs at the same training step. −0.0032 is 2× the single-seed std floor (~0.0015), pointing to a genuine mechanism effect. The training trajectory shifted from near step 1 and maintained the gap to saturation — not a transient fluctuation.

## α trajectory — the rich signal

Layout: `[[pass2_L3, pass2_L4, pass2_L5], [pass3_L3, pass3_L4, pass3_L5]]`

```
step | pass-2 (earlier extra)    | pass-3 (later extra)
-----+---------------------------+------------------------
2000 | 0.00  0.00  0.00          | 0.00  0.00  0.00       pre-activation (loop off)
2142 | layer_loop:enabled (frac 0.350)
2200 | 0.03  0.07  0.14          | 0.16  0.24  0.33       just activated
2500 | 1.00  1.16  1.37          | 0.85  0.76  0.75
3000 | 1.04  1.16  1.38          | 0.98  0.86  0.76
4000 | 1.04  1.16  1.38          | 1.01  0.89  0.77       saturated
4761 | 1.04  1.16  1.38          | 1.01  0.89  0.77       (endpoint)
```

**Three non-trivial observations:**

1. **Pass-2 amplifies**: final values [1.04, 1.16, 1.38] — all ≥1.0, growing with depth. The model is not just gating loop contribution; it's *amplifying* later layers' loop pass-2 output beyond the residual. This is strictly above the Loop45 no-α baseline (implicit α=1.0 everywhere).

2. **Pass-3 damps**: final values [1.01, 0.89, 0.77] — decreasing with depth. Deeper in the loop → partial commitment. Layer 5 contributes only 77% of pass-3's residual update.

3. **Depth gradient asymmetry**: pass-2 α *increases* with layer depth; pass-3 α *decreases*. The learned shape is non-trivial — the model found a non-uniform per-pass-per-layer allocation that differs from both the null (all-zero) and identity (all-one) initializations.

This pattern is **informative for future architecture work**: it suggests the later layers in the loop are better treated differently between early and late re-passes — a lead for fixed-α architectural priors or per-layer loop budget reallocation.

## Known issue: grad_norm logging

`recur_alpha grad_norm` printed as `0.000000` every step due to logging firing after `optimizer.zero_grad()`. This is a cosmetic bug — α values clearly move (up to 1.38), proving the gradient path works. Fixed in commit `4dd2d63` (spec 016 onward).

## Decision — PROMOTE to spec 016, then evaluate stacking

The −0.0032 matched-step improvement is real. Spec 015 was designed as a two-step: first establish α=0 init signal, then test α=1 (spec 016) to rule out init sensitivity. That pipeline ran cleanly.

**Do NOT submit standalone.** 015 does not reach a new SOTA on its own — endpoint is roughly tied with 008, and the matched-step advantage needs to survive GPTQ+TTT to matter for the leaderboard.

**Next steps:**
- Evaluate spec 016 for init sensitivity (done).
- If recur-alpha survives as a stack ingredient, consider 3-seed confirmation (~$36) before submission promotion.
- Investigate whether the asymmetric α pattern could be hardcoded as a fixed architectural prior (no learned params, lower complexity).

## Cost

~$11 (smoke false-halt + rsync pod + screen run).

## Note on throughput deficit — see spec 016b

The 67-step deficit vs spec 008 (stopping_early 4761 vs 4828) was partially attributed to JP hardware variance at eval time. **Spec 016b confirms the deficit is real architectural overhead**, not just node luck. Recur-alpha costs ~1–2% throughput at full model size due to the blend op `x = α * x_new + (1-α) * x_before`. This explains ~30–40 of the 67 lost steps; the remainder is genuine node variance. The matched-step Δ (−0.0032) remains the honest signal. See `research/evaluations/016b-recur-alpha-throughput.md`.

## Cross-references

- Spec: `research/specs/015-recur-alpha.md`
- Execution notes: `runs/015-recur-alpha/seed_42/notes.md`
- Follow-up: `research/evaluations/016-recur-alpha-ones.md`
- Throughput diagnostic: `research/evaluations/016b-recur-alpha-throughput.md`
