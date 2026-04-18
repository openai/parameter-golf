# Progressive Recurrence

**Status:** candidate — design fork needs resolving before spec
**Expected Δ:** +0.0005 to +0.0015 bpb if the port works; 0 or negative if the hard-switch spike wasn't actually costing much
**Source:** `records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/README.md`, "Progressive Recurrence" section.

## Idea (near-SOTA version)
That submission had **two separate loops** and activated them at staggered fractions: LOOP_PHASE1_AT=0.50 and LOOP_PHASE2_AT=0.65. The claim: enabling both loops at once causes a sharper loss spike; staggering gives the optimizer time to adapt to each added pass before the next.

## Port problem
Current SOTA has a **single activation** at `enable_looping_at` (≈0.35) that flips `looping_active=True` in one shot. See `train_gpt_sota.py:517-525` (loop index construction) and `1348-1353` (activation). Before the switch, layers 3-5 each run once. After, they each run 3 times (via repeated indices in `encoder_indices`). There's no natural "phase 1 / phase 2" structure to stagger.

## Design fork
Three candidate reinterpretations of "progressive":

### Option A — ramp `num_loops` from 0 to 2 over a window
Fire three activations at e.g. frac ∈ {0.30, 0.375, 0.45}. Each activation adds one more pass through layers 3-5, rebuilding `encoder_indices`/`decoder_indices`. Each rebuild triggers a compile of the new forward graph (baseline already pays this cost once).

**Pros:** most faithful to the near-SOTA's intuition — smaller optimizer shocks spread over time.
**Cons:** 3 recompiles instead of 1. Need to verify compile cost doesn't eat into the 10-minute training budget.

### Option B — output-blend α(step) between looped and non-looped forward paths
During the ramp window, run both the looped and non-looped forward passes and linearly blend outputs. α(step) goes 0 → 1 over the window.

**Pros:** smooth; no recompiles.
**Cons:** **doubles compute during the ramp window**, which directly costs training-budget seconds. And the two paths produce outputs from different layer-iteration orders — blending them may not be semantically meaningful.

### Option C — just activate recurrence earlier
Move `enable_looping_at` from 0.35 to 0.25. More training time with recurrence active, less time with it off. Not really "progressive," but it's the trivial control experiment for the whole hypothesis.

**Pros:** trivial code change. Fair baseline for the other two options.
**Cons:** if this alone helps, then "progressive" was a red herring.

## Hotstart screening plan
All three options are training-dynamics changes. Each needs real GPU tail compute, but hotstart saves ~25-30% of training.

- **Hotstart from:** earliest pre-activation checkpoint. For Option A/B (ramp starts at frac ≈ 0.30), use `ckpt_event_step1137` (25%) — the closest `CKPT_STEPS` milestone before the ramp starts. For Option C, use `ckpt_pre_recurrence` (frac ≈ 0.35 in baseline, but baseline recurrence is what we're *replacing*, so using the pre-recurrence checkpoint is fine).
- **Tail:** from 25% (step 1137) to end (step 4550) ≈ 3400 steps ≈ 7-8 min on 8×H100, or ~30 min on 2×H100.
- **Control:** same hotstart, baseline single-activation recurrence in the tail. Run this 2-3 times to establish the hotstart-tail noise floor.
- **Variant:** same hotstart, progressive variant (pick one option per screen).
- **Wall per run:** ~7 min (8×H100) or ~30 min (2×H100).
- **Cost per run:** ~$2.50 (8×H100) or ~$2 (2×H100).
- **Budget for a screen:** 2-3 control seeds + 2-3 variant seeds per option ≈ $10-15 per option.
- **Promotion threshold:** variant mean minus control mean ≥ 2× control std, AND |Δ| ≥ 0.0005. Tighter than Hessian-SDClip because screen noise is nonzero.

**Order of screens:** run Option C first (trivial, cheapest to implement). If C already wins, the "progressive" framing is probably unnecessary and we save weeks. If C loses or ties, run Option A.

## Code-change sketch

### Option A
- Parameterize `num_loop_activations` (default 1 = baseline) and `progressive_activation_fracs` (default = `[h.enable_looping_at]`).
- In the training loop, track which activation(s) have fired. On firing each one, append another copy of `range(loop_start, loop_end+1)` to `encoder_indices` / `decoder_indices` and force a compile.
- Total final num_loops = len(progressive_activation_fracs) (replaces `h.num_loops` as the authoritative count).

### Option C
- Env var to set `enable_looping_at`. That's it. Probably already parameterizable.

## Risks / open questions
- **Is the hard-switch spike real?** Check baseline training logs for a visible loss bump around frac=0.35. If it's not there, the whole premise evaporates and the fork resolves to "none of the above."
- **Compile cost for Option A.** Each `encoder_indices` rebuild re-compiles. Three recompiles might cost several seconds each, eating training budget. Verify empirically or instrument.
- **Seed variance in the tail.** Training from step 1137 to 4550 accumulates seed variance. Need proper multi-seed screen to see past noise, which inflates the screen cost.
- **Interaction with warmdown.** If the last progressive activation fires after warmdown starts (frac ≈ ?), the LR is already decaying and the "adaptation window" is shorter than intended. May need to gate activations to before warmdown.

## If this works
- Orthogonal to Candidates 1 and 3 (those are quant-time; this is training-time). Stacks cleanly in a final record attempt.
- Winning variant becomes a training-code change (on an `exp/progressive-recurrence` branch) rather than a hyperparam-only spec.
