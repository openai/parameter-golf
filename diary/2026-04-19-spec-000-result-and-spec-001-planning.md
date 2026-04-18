# 2026-04-19 — Spec 000 result + into research phase + spec 001 planning

## Spec 000 ran — and landed outside the accept window

The first real run on 8×H100 NA-1 completed cleanly, no NaN, no divergence. Final post-TTT val_bpb: **1.08622**. Target was 1.0810 ± 0.002 (leaderboard SOTA). We missed by +0.0032, outside the window.

Good news first: the code is faithful. The Δ vs SOTA is **~+0.005 bpb consistently across every eval stage** — training-end, post-EMA, quantized, sliding, TTT. And the within-run contributions (quant penalty, sliding gain, TTT gain) match SOTA almost exactly. If the code were broken, we'd expect the miss to appear at one stage and not others. It doesn't.

## The diagnosis: throughput, not code

Our pod ran **3849 steps** in 588 seconds (~6.5 steps/s). SOTA's pod ran **4550 steps** in 588 seconds (~7.74 steps/s). Same nominal hardware (8×H100 SXM), same code, same env — different pod. We got ~85% of SOTA's step rate, which cost us ~700 steps and ~0.005 bpb across the board.

Hypothesis: Runpod's NA-1 8×H100 pool is heterogeneous. Nominal spec is the same, but NVLink topology, host memory bandwidth, or driver/BIOS stack can differ. We pulled a slow one.

This is frustrating because there's nothing to *fix*. The throughput deficit tracks cleanly through every downstream number. We're under-trained by a fixed amount, and it shows exactly where you'd expect.

Cost: **$13.10** total — $9.50 on the actual run, $3.60 on provisioning churn (an SSH heredoc issue abandoned a pod). So not just slow, also some waste. For a $3.50 estimate. Not great. Lessons for execution: avoid long heredocs, use setsid/tmux from the start.

## Decision: adopt 1.08622 as our baseline, don't re-roll

Tempting option: re-provision and hope for a faster pod. Bad option, I think:
- Another ~$10 and still a random hardware draw.
- Our real goal isn't 1.0810-replication; it's beating 1.0810. We've already validated that the pipeline works end-to-end.
- All our downstream screens either don't care about pod speed (quant-time hotstart screens) or control for it via paired control+variant on the same pod.

So 1.08622 is our operating baseline. Any Δ we measure from here compares to it. If a stack of improvements lands at ≤1.0810 on our slow hardware, that's enough evidence to do the final record-attempt run on a pod-shopped-for-speed — with a tok/s preflight ($1, 2min) to filter the slow ones.

## Now we're in the research phase

This is the part of the project where it stops being pure operations and becomes actual research. We have:

- A faithful baseline (spec 000's 1.08622).
- 9 phase-boundary checkpoints on NA-1 volume (2.7 GB total), usable as hotstart seeds:
  - `ckpt_final_pre_ema_step3849` — for quant-time experiments (no training needed)
  - `ckpt_pre_recurrence_step1378` — for recurrence-schedule experiments
  - `ckpt_warmdown_start_step1048` — for warmdown/TTT experiments
  - Plus interior steps at 455, 1137, 1500, 2275, 3412
- A four-candidate idea queue (after surveying the near-SOTA records and cross-referencing with our current code):
  - **Hessian-SDClip** — quant-time, cheapest screen, ~60-70% confidence of some Δ
  - **BigramHash** — already in our code but disabled; needs full retrain; possibly biggest single Δ (+0.003-0.005)
  - **Progressive recurrence** — training-dynamics, design fork we haven't settled
  - **Per-group bit allocation** — quant-time, speculative, 16MB budget is the gating question

Workflow in research phase is: survey → rated proposal → user picks → long discussion → hotstart screen → ship to execution. We explicitly agreed to keep step 3 (the discussion) — no shortcuts from "here are three ideas" straight to "spec written."

## How we picked spec 001

Several convergent reasons to start with **Hessian-SDClip**:

1. **Cheapest screen by far.** It's post-training only — load the checkpoint, re-run quantization with a different clip formula, read the bpb. No training, no DDP, no distributed anything. Runs on 1×H100 at ~$0.45 for an initial 3-λ probe.

2. **Unaffected by the throughput story.** The screen runs GPTQ on fixed weights. Whether our 8×H100 pod was slow or fast doesn't matter — we're using the exact same checkpoint file either way. This isolates the Hessian-SDClip signal from the hardware noise we can't control.

3. **Cleanest A/B.** Control (λ=0) must reproduce our baseline's 1.10430 quantized bpb to within 0.0001. Deterministic given fixed calibration data. Noise floor is essentially zero, so even +0.0003 bpb shows up clearly.

4. **Informs the next move.** If Hessian-SDClip shows signal, we know quant-pipeline tweaks transfer to our architecture and per-group bit allocation becomes the natural follow-up. If it shows nothing, we save budget and pivot.

## Why 1×H100 and not more

I had 2×H100 in the spec initially (matching CLAUDE.md's "mini rung" convention), but we walked it back to 1×H100. Reason: GPTQ's per-matrix work is sequential — 70+ matrices each doing Cholesky + column-by-column quantize — and doesn't shard across GPUs. Eval passes and Hessian collection do parallelize, but those are the small parts. Net: more GPUs = same wall time, higher cost. The "mini rung" convention exists to catch DDP bugs before scaling up; this spec has no scale-up path, so the convention doesn't apply.

## Why only 3 λ values to start

Originally I proposed 8 (a proper grid around the near-SOTA-reported sweet spot of 0.175). User cut it to 3: {0.00, 0.05, 0.10}. The philosophy shift is sharp:

- "Does it change anything at all?" is a different question from "where's the optimum?"
- Start at the conservative low end. If even λ=0.05 moves the number, we know the effect is real and can drive higher or lower live.
- If 0.05 and 0.10 both do nothing, the larger question is "does this technique transfer to SOTA-stack at all?" and we might be better off killing rather than exhaustively searching.

Importantly: the executioner keeps the pod alive after the initial 3. Hessian reuse means each additional λ value costs pennies (setup already paid). So the live-exploration mode isn't lossy — it's strictly better than pre-committing to a grid.

## Spec 001 status

Frozen at `research/specs/001-hessian-sdclip.md`. Code change on `exp/hessian-sdclip` @ `74c8385` — 12 lines added to `train_gpt_sota.py`. λ=0 is a no-op (reverts to baseline SDClip), so the control is a strict validity check: if it doesn't reproduce 1.10430 exactly, we have a bug and stop.

Handing to an execution session next. Budget: $0.45 base, ~$1 cap if follow-ups expand.

## Running tally

- $13 of $200 budget spent.
- 11 days to deadline.
- 1 baseline run complete, 1 spec ready to ship, 3 more candidates in the queue.
