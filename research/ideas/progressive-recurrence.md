# Progressive Recurrence

**Status:** candidate
**Expected Δ:** +0.001 to +0.002
**Source:** 2026-04-06 submission (1.0835 bpb) used this over hard on/off recurrence activation.

## Idea
Current SOTA code activates depth recurrence with a hard switch at step N (loop layers 3–5 are no-ops before N, full 3-layer recurrence after). The 2026-04-06 submission uses a **fractional activation schedule**: α(step) ramps smoothly from 0 → 1 over a window around N. The output of the recurrent block becomes `α(step) * recur_out + (1 - α(step)) * skip_out`.

Claim in submission notes: reduces the loss spike at the activation transition.

## Why it might help
- The hard switch is a discontinuity in the effective computation graph; gradients seen right after the switch differ sharply from right before.
- Smoothing the transition keeps optimization in-distribution throughout.
- Essentially free: one scalar schedule, no param cost, no FLOP cost (the recurrent pass happens anyway).

## Code-change sketch
- In `train_gpt_sota.py`, find where recurrence activation is gated (likely a step check in the forward pass or a `recur_active` flag).
- Replace boolean with a scalar α(step). Linear ramp over, say, 200 steps centered on the current activation step.
- Apply α as a blend on the recurrent block's output.

## Risks / open questions
- Where exactly is the hard switch implemented? Need to locate the code path.
- Is the recurrence additive (extra layers stacked) or multiplicative (output blended)? The blend interpretation only works if the block's output can be meaningfully interpolated with the skip.
- Ramp window length — too short = same as hard switch, too long = diluted benefit. Start with 200 steps.

## If this works
Combines cleanly with everything else; no interaction concern.
