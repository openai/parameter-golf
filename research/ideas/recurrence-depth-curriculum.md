# Recurrence Depth Curriculum

**Source:** PR #1756 (romeerp, 2026-04-21)
**Author credibility:** romeerp has #1610, #1729 in the record track — bridges SGD-TTT and tokenizer levers; has produced validated results before.

## What it is

Replace the fixed recurrence depth during training with a phased curriculum, then evaluate at the deepest trained depth:

- **Phase 1 (steps 0–1/3):** depth=1 — model trains as single forward pass, no recurrence
- **Phase 2 (steps 1/3–2/3):** depth=3 — matches our current Loop45 config (3 total depth applications)
- **Phase 3 (final 1/3):** depth=4 — one extra loop application
- **Eval:** always at depth=4 (one extra loop vs. our current inference depth)

Controlled by three env vars in the training script:
- `TRAIN_LOOP_PHASE_DEPTHS=1,3,4`
- `TRAIN_LOOP_PREWARM_DEPTHS=3,4`
- `EVAL_LOOP_DEPTH=4`

No new dependencies. The loop block is unchanged; only its repetition count during training and eval changes.

## Why it might work

Teaching the shared loop block progressively: depth-1 trains it as a standalone layer (forces it to be useful without recurrence); ramping to depth-3 recovers the current regime; depth-4 trains an extra refinement pass the current model never sees during training, then harvests that signal at eval. The curriculum prevents the block from over-specializing to recurrent context from step 1.

## Evidence from #1756

- 3-seed mean: **1.06505** vs. our baseline **1.06549** (Δ = −0.00044)
- Seed breakdown: 0 → 1.06417, 42 → 1.06520, 1234 → 1.06578
- Pre-TTT base model also improves (mean 1.07772), confirming the gain isn't TTT-only
- Std = 0.00081 (~4× SOTA std) — elevated; seed 1234 is slightly worse than our baseline

## Estimated Δ-bucket

−0.0004 to −0.001 bpb. Small but genuine if validated. Combinatorial with other levers.

## Risks

1. **Variance:** 1 of 3 seeds is worse than our baseline. Need our own independent run to confirm.
2. **Eval throughput cost:** depth=4 loop at TTT/quant time runs ~33% more compute in the loop block. May affect 10-min wall clock constraint. Check step time in mini run.
3. **Interaction with phased TTT:** our TTT adapts the model at eval — if the TTT schedule was tuned for depth-3 inference, switching to depth-4 may require re-tuning TTT phases.
4. **Compatibility:** our baseline uses `loop_start=4, loop_end=5` (Loop45). The curriculum is expressed in terms of the number of loop applications (1, 3, 4). Map carefully to Loop45 config params before spec.

## Feasibility

High. Env-var controlled, no new code outside the training loop scheduler. Builds directly on #1736. The code is already in romeerp's PR — can extract the curriculum logic cleanly.

## Next step

Spec as a mini run (2×H100, screen-level). Target: independently reproduce the −0.00044 Δ with our own seed. If confirmed, run 3-seed official. Cost: ~$3–4 for screen.
