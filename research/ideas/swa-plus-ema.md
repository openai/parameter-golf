# SWA + EMA combo

**Status:** candidate
**Expected Δ:** +0.001 to +0.003
**Source:** 2026-03-25 submission used SWA on top of EMA.

## Idea
SOTA code already uses EMA (exponential moving average of weights, decay 0.9965). SWA (stochastic weight averaging) is a separate, complementary technique: sample weights at fixed intervals (e.g., every 50 steps) during the warmdown phase and average them uniformly.

The evaluated model becomes EMA + SWA combined, or a weighted mix of the two.

## Why it might help
- EMA biases toward recent weights (exponentially). SWA is uniform over sampled points.
- During warmdown, the model is in a flat region of the loss landscape; uniform averaging is known to find wider minima than single-point evaluation.
- Almost free: one extra set of weights held in memory, one parameter-wise add per sample.

## Code-change sketch
- Add a second weight tracker alongside the EMA tracker.
- During warmdown (last 72% of training in SOTA), sample every K steps.
- At eval time, compare: pure EMA vs pure SWA vs (EMA + SWA)/2. Pick the best.

## Risks / open questions
- The 2026-03-25 submission was not top-ranked, so this alone isn't a huge win.
- Memory cost: one extra copy of model weights (~20MB) — fine during training.
- Which phase to sample SWA from? Likely warmdown only (flat region).
- Sample rate K — standard literature uses every epoch, but here an "epoch" is fuzzy. Use every ~100 steps as a starting point.

## If this works
Stacks cleanly. Evaluation-time only change once weights are collected; no training compute overhead.
