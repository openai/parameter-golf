# Evaluation — Spec 014 (BPB-weighted CE loss)

**Run dir:** `runs/014-bpb-weighted-loss/seed_42/`
**Commit:** `ab6a131` on `exp/bpb-weighted`
**Baseline:** spec 008 `runs/008-1736-reproduction/seed_42/final.json` (val_bpb 1.0697)
**Eval date:** 2026-04-21

## Result — clear regression

| metric | spec 008 | spec 014 | Δ |
|---|---|---|---|
| endpoint val_bpb (bare, screening mode) | 1.0697 | 1.1316 | **+0.0619** |
| mid val_bpb @ step 4000 | 1.1110 | 1.1743 | +0.0633 |
| stopping_early step | 4828 | 4797 | −31 |

**62× past the shelve threshold.** Not a null — an actual regression.

## Noise/signal judgment

This is well outside any noise interpretation. The +0.0633 at step 4000 is consistent with the +0.0619 endpoint, meaning the trajectory shifted from step 1 and never recovered. Not destabilization (no NaN), but a persistent ~0.06 bpb penalty across the whole run.

## What went wrong

No destabilization (no NaN, smooth curve). Subtler failure:

- Byte-weighted gradient has a **different directional signature** than uniform CE.
- Muon's LR schedule, momentum warmup, and weight decay were all tuned on uniform CE.
- With byte-weighted gradients, the effective step-size profile is mismatched — the optimizer is pushing in off-axis directions relative to its tuning.

Execution flagged: "byte-weighted gradient direction is structurally different from uniform CE, and Muon's LR schedule is tuned on uniform CE. With weighted gradients concentrated on multi-byte tokens, the effective LR / step-size profile is mismatched, and the model under-trains at endpoint."

This is exactly the "#1519 destabilization on large vocabs" failure mode, just manifesting as trajectory offset rather than NaN. The author's warning held.

## Decision — SHELVED (permanently for this push)

**Do not retune with co-swept MUON_LR.** Three reasons:

1. **Magnitude of regression.** +0.0619 is not "retune-recoverable" in any realistic scenario. Even if a lucky LR combo clawed back most of the gap, we'd be left with uncertainty about whether the clawback was real signal or coincidence.

2. **Meta-pattern from this push.** 011 null, 013 null, 014 regression — three incremental ports from different-stack authors have failed to transfer to #1736. The evidence is now strong that this class of experiment has low EV on our stack. See `project_frontier_advancement_pattern.md` memory.

3. **Budget and focus.** We have ~$126 remaining, 10 days, and an active recurrence research thread (Recur-Alpha, spec 015). Spending $20-40 on a 014 retune pulls budget and attention from higher-EV work.

## Methodological lesson captured

If a future push wants objective-alignment levers:
- **Don't port-and-evaluate.** Co-tune the objective with optimizer schedule from the start.
- **Use the context-aware byte sidecar**, not the surface-piece `base_bytes_lut`. The approximation error may have been a contributing factor.
- **Train with the lever ON from step 0.** Bolting it on at eval doesn't match how it was tuned by the author.

## Cost

~$7.50 (2H smoke + 8H screening).

## Revisit criteria (post-deadline only)

- If Recur-Alpha (spec 015) lands AND remaining budget allows AND the result hints at optimizer-miscalibration issues, a retuned 014 variant could be a confirmation experiment. But that's contingent.
- If this push ends without 014 being revisited, file the lever as "interesting in principle, didn't transfer on #1736" and move on.

## Cross-references

- Spec: `research/specs/014-bpb-weighted-loss.md`
- Source PR: #1519 (elliottdehn)
- Author's own pre-registered failure mode: large-vocab destabilization
- Execution notes: `runs/014-bpb-weighted-loss/seed_42/notes.md`
