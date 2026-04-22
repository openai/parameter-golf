# Spec 014 execution — shelving BPB-weighted loss

**Date:** 2026-04-21 (UTC) / 2026-04-20 evening (US Central)
**Mode:** execution session
**Spec:** `014-bpb-weighted-loss` (port #1519: byte-weight per-token CE so training objective matches eval BPB metric)
**Result:** **SHELVED** — clear regression. Endpoint val_bpb +0.0619 vs spec 008 baseline.

## TL;DR

- Smoke (2×H100, 500 steps): no NaN, no destabilization, monotonic decrease. Train_loss number was higher than spec 008's at step 500, but that's the metric-mismatch artefact (014 logs byte-weighted CE; 008 logs uniform), not a destabilization signal. Smoke PASSED on every literal criterion.
- 8H full screening run: completed cleanly to step 4797 / 596s wallclock. No NaN.
- **Apples-to-apples val_bpb at step 4000: +0.0633 worse than spec 008.**
- **Endpoint val_bpb: 1.1316 vs spec 008's 1.0697 → +0.0619.**
- 62× the spec's regression threshold (+0.001). Verdict per accept criteria: shelve.
- Cost: ~$7.50 (smoke + parallelized 8H idle + full run + endpoint kill).

## What we learned

The misalignment between training CE and eval BPB is real and well-motivated — multi-byte tokens contribute 4× to eval but 1× to gradient under standard CE. Reweighting to align should help in principle.

In practice on this stack: doesn't help, hurts measurably. Two plausible reasons (research's call to investigate or just shelve):

1. **Effective-LR shift.** Muon's LR schedule is tuned for uniform CE. With weighted gradients concentrated on multi-byte tokens, the effective per-step update magnitude on those parameters is ~5× larger. Net result: the model is implicitly running at a different LR profile than tuned for, and underperforms.
2. **CaseOps approximation.** Spec used `base_bytes_lut` (surface-piece bytes) not the context-aware `val_bytes` sidecar. Case-flag tokens get up-weighted by their surface bytes but contribute zero canonical bytes in eval — small over-weighting of a population the eval doesn't reward. This effect should be small (case-flags are rare), but compounds with (1).

Either fix needs more work than a single env-var flip, and the prior on the lever transferring just dropped a lot.

## The metric-mismatch interpretation hazard

Spec 014's train_loss at step 500 was 3.27, vs spec 008's 2.58 — Δ +0.69 nats. For ~30 seconds at smoke completion, this looked like destabilization (which the spec explicitly warned about as a documented #1519 risk on large vocabs).

**But the loss numbers are on different metric definitions:**
- Spec 008: `loss = mean(ce_per_token)` (uniform).
- Spec 014: `loss = sum(byte_weight × ce) / sum(byte_weight)` (weighted average).

When the model is trained partway, multi-byte tokens are still relatively harder than single-byte (model learns common short tokens first). Byte-weighting then gives the still-hard tokens 5–15× more weight in the reported number. So weighted-loss can be 0.5+ nats higher than uniform-loss without anything being wrong with training.

The apples-to-apples comparison is val_bpb (computed identically for both runs by the eval harness). Once that landed at step 4000 (1.1743 vs 1.1110, Δ +0.0633), the verdict was unambiguous.

**Lesson for future specs that change loss definition:** the spec should state up front "ignore train_loss comparison; trust val_bpb only." Otherwise execution wastes 30 seconds re-deriving it (and almost makes the wrong destabilization-kill call).

## Cost accounting today

| spec | cost | outcome |
|---|---|---|
| 011 (training-bundle) | ~$9.50 | training healthy, taper-never-fired bug, +0.0009 endpoint Δ |
| 013 (bigram-hash) | ~$5.00 | training healthy, +0.0025 endpoint Δ (likely null) |
| 014 (BPB-weighted) | ~$7.50 | training healthy, **+0.0619 endpoint Δ** (clear regression, shelve) |
| **Day total** | **~$22** | within $20/day soft budget bumped slightly |

## Three results, one pattern

Spec 014's pre-registered "honest self-check" called this exactly:

> Three null/regression results in a row (011 GradPower, 013 BigramHash confounded, hypothetical 014 null) would suggest a meta-pattern: **#1736 is already well-tuned, incremental ports from other stacks don't stack on top.** If 014 is also null, the better question isn't "what lever next" but:
> - (a) Retune something #1736 already does (Muon schedule, LR, etc.) for our specific data
> - (b) Commit to a 3-seed official on plain spec 008 and squeeze std advantage
> - (c) Fundamentally different quant approach (AWQ instead of GPTQ)

We now have 014 hitting hard regression (worse than null). The meta-pattern is real. Research's call on the pivot.

## PR #1519 formula sanity-check

User flagged a sharp question: are we computing val_bpb correctly? Verified end-to-end:
- Spec 014's training-loss formula matches PR #1519 exactly: `(ce * byte_weight).sum() / byte_weight.sum()`.
- Eval is uniform CE in both codebases (PR gates on `self.training`; spec 014's `eval_val()` bypasses `model.forward()` and runs its own uniform `F.cross_entropy(reduction="none")`).
- Same val_bpb formula `loss/log(2) * tokens/bytes` for both runs. Apples-to-apples.

The +0.0619 regression is consistent with PR author's own pre-registered failure mode: lever works on SP1024 (max byte weight ~8), explicitly warned-against on GPT-2 50K. SP8192 (max 15) sits in the gray zone, plus #1736 baseline is 8x stronger than the PR's. Both factors compound against transfer; both materialized.

## Handback

Spec 014: **SHELVED**. Notes at `runs/014-bpb-weighted-loss/seed_42/notes.md`. Pod `9l34ork0rt5vc0` stopped (will terminate end-of-day).

Recommended pivot (per spec author's pre-registered self-check): option (b) — 3-seed official on plain spec 008 to lock in the std advantage. Plus possibly (a) — retune Muon LR for our specific data instead of porting external levers. Option (c) AWQ-vs-GPTQ is a bigger commit; only worth if (b) doesn't widen the leaderboard gap.
