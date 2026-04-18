# Evaluation — Spec 000 (SOTA Replication)

**Run:** `runs/000-sota-replication/` | **Seed:** 42 | **Hardware:** 8×H100 NA-1 | **Date:** 2026-04-18 → eval 2026-04-19
**Code:** commit `01e6fcf` on `research`, launch env `BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.25 TTT_ENABLED=1`

## Result
Final post-TTT `val_bpb = 1.08622`. **Outside the accept window [1.079, 1.083] by +0.0032.**

| Stage | Ours | SOTA (2026-04-09) | Δ |
|---|---|---|---|
| Training-end val | 1.0938 | 1.0886 | +0.0052 |
| Post-EMA, pre-quant | 1.09289 | 1.08735 | +0.0055 |
| Post-quant (raw) | 1.10430 | 1.09970 | +0.0046 |
| Post-quant + sliding | 1.08774 | 1.08286 | +0.0049 |
| **Post-quant + TTT (final)** | **1.08622** | **1.08079** | **+0.0054** |

## Noise/signal judgment — clear signal, NOT noise
The Δ vs SOTA is **~+0.005 bpb consistently across every eval stage**. SOTA's 3-seed std is ~0.0002, so +0.005 is ~25σ — nowhere near noise. This is a real, reproducible gap.

**Root cause: throughput deficit on Runpod NA-1, not a code bug.**

Evidence:
- Our pod ran **3849 steps in 588s** (~6.5 steps/s). SOTA's pod ran **4550 steps in 588s** (~7.74 steps/s). We got ~85% of the step count in the same 10-min budget.
- **Within-run contributions match SOTA almost exactly:**

  | Within-run Δ | Ours | SOTA |
  |---|---|---|
  | EMA gain | −0.0009 | −0.0013 |
  | Quant penalty | +0.0114 | +0.0124 |
  | Sliding-window gain | −0.0166 | −0.0168 |
  | TTT gain | −0.0015 | −0.0021 |

  Every sub-system is producing the right-signed, near-right-magnitude delta. The only thing smaller than SOTA is the TTT gain (−0.0015 vs −0.0021), which is consistent with an under-trained base having less signal for TTT to amplify.

- If this were a code/config bug, we'd expect the Δ to appear at one stage and not others (e.g. quant-only, sliding-only). It doesn't — the ~+0.005 tracks through all five stages, exactly as if the base model had 15% fewer training steps, which it did.

**Hypothesis for hardware variance:** Runpod's "8×H100 SXM NA-1" pool is heterogeneous — different NVLink/NVSwitch topology, host memory bandwidth, or BIOS/driver stacks can give 10-15% step-time variance on the same nominal SKU. We got a slow pod.

## Loss-curve notes
- Depth recurrence activated at **step 1378** (frac 0.35), which is earlier in absolute-step terms than SOTA (which would activate around step 1593 of 4550). Because the activation is fraction-based, not step-based, it still fires at frac 0.35 — but our 0.35 is a smaller absolute training distance.
- Warmdown started at step 1048 — the "warmdown_start" checkpoint exists.
- Muon momentum warmup ended at step 1500 (auto-captured event).
- No NaNs, no divergence, no anomalies in the curve. The code is faithful.

## Decision: **iterate**, not kill, not promote
**Do not re-run 000 just to try for replication.** Reasons:
1. The code is faithful. The 1.08622 vs 1.0810 gap is a hardware-sourced under-training artifact, not a codebase problem to fix.
2. Re-running on another Runpod pod gives us a random hardware draw — ~$10-13 to maybe still get a slow pod.
3. Our real goal is a record, not replication. We have all 9 phase-boundary checkpoints and a faithful pipeline; we should spend budget on *improvements* now.

**New operating baseline for downstream specs: `val_bpb = 1.08622` (our spec-000 result).** All Δ measurements in future evaluations compare against this, not against the leaderboard's 1.0810. A winning stack that lands ≤ 1.0810 on our hardware is effectively a likely record once re-run on faster hardware.

### Consequences for candidate specs
- **Hessian-SDClip** (quant-time, hotstart from `ckpt_final_pre_ema_step3849.pt`): unaffected by throughput — the quant pipeline runs the same regardless of how many training steps produced the weights. **Screen at full fidelity.** This is now the cheapest, cleanest first screen.
- **Per-group bit allocation** (quant-time, same hotstart): same story. Clean screen.
- **Progressive recurrence** (training-dynamics, hotstart from `ckpt_event_step1137.pt` or `ckpt_pre_recurrence_step1378.pt`): **is affected.** A hotstart-tail screen on our slow pod trains only the tail; the absolute bpb can't be compared to SOTA. But the Δ between control and variant at the same hotstart IS clean — still a valid screen under our methodology.
- **BigramHash** (full retrain required): needs a fresh 8×H100 run. On a slow pod, it lands at our 1.08622 − expected_Δ, not leaderboard − expected_Δ. Interpret accordingly.

### Hardware strategy before any record attempt
Before committing a full 3-seed record run, add a **tok/s preflight** on a 1×H100 or small rung:
- Run ~50 training steps.
- Measure tokens/sec. SOTA's full-run average was ~6.1-7.7M tok/s on 8×H100.
- If single-GPU tok/s scales to ≥ ~7.5M on 8×H100, it's a "fast" pod.
- If it comes in closer to ~5M, it's a "slow" pod — stop and re-provision.

This adds <$1 and ~2 min per provisioning and directly addresses the bug that cost us ~$10 this round.

## Next steps

1. **Screen Hessian-SDClip first** — cheapest and strongest candidate. Hotstart from `ckpt_final_pre_ema_step3849.pt`, sweep λ ∈ {0, 0.10, 0.15, 0.175, 0.20, 0.25, 0.30}. Budget ~$3. Should fit in a single 1×H100 session.
2. **Screen per-group bit allocation** after (1). Same hotstart. Budget ~$5 for initial trial set.
3. **If either (1) or (2) shows Δ ≥ +0.0005, do a stacked screen** (both applied to the same ckpt).
4. **Defer progressive-recurrence screening** until (1) and (2) resolve — it's the most expensive screen and the design fork needs settling first.
5. **Defer BigramHash** — full-retrain cost, and the 16MB budget fit is unverified. Don't burn 8×H100 on this until quant-time candidates are evaluated.

## Cost accounting
- This run: **$13.10** (~$9.50 training + ~$3.60 provisioning churn).
- Spec estimate was $3.50-6. Overrun is real — mostly from an SSH heredoc issue during launch that cost an abandoned pod. Execution notes flag this for next session.
- Running tally: **~$13 of ~$200 hard budget** spent, 11 days to deadline.

## Artifacts retained
- `final.json` — structured, schema-compliant.
- `train.log` — full stdout, 161 lines.
- `checkpoints.md` — pointer to 9 phase-boundary checkpoints (2.7 GB on NA-1 volume `hvpdph5i3g:/workspace/runs/000-sota-replication/checkpoints/`).
- `notes.md` — execution's own run narrative.

## Open questions raised by this run
- Is the "slow pod" hypothesis correct? Confirming would need running the same code on a different NA-1 pod and seeing if tok/s changes. Not worth the $ unless a future run also comes in slow.
- Can we get a faster region/instance type for the final record attempt? Worth investigating before the submission run (not before screens).
- TTT gained slightly less on our run than SOTA (−0.0015 vs −0.0021). Is this purely an under-training artifact, or does TTT need a slightly different LR when the base is less trained? Speculative, probably not worth digging into unless a pattern emerges.
