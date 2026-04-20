# Evaluation — Spec 013 (BigramHash auxiliary embedding)

**Run dir:** `runs/013-bigram-hash/seed_42/`
**Commit:** `66e57bf` on `exp/bigram-hash`
**Baseline:** spec 008 `runs/008-1736-reproduction/seed_42/final.json`
**Eval date:** 2026-04-20

## Result

| metric | spec 008 | spec 013 | Δ |
|---|---|---|---|
| endpoint val_bpb (bare, screening mode) | 1.0697 | 1.0722 | **+0.0025** |
| step-4000 mid-train val_bpb | 1.1110 | 1.1144 | +0.0034 |
| stopping_early step | 4828 | 4833 | +5 (no step-time regression) |

Single seed, screening mode (no TTT/GPTQ/sliding measured).

## Noise/signal judgment

**Probably real, small regression.** #1736-class per-seed std on val_bpb is ~0.0005–0.0007. The 95% CI on a single-seed point estimate is ~±0.0015. Observed +0.0025 is outside that on the unfavorable side. Not a clean null, not a catastrophic regression.

## Matched-step train_loss trajectory

| step | spec 008 | spec 013 | Δ |
|---|---|---|---|
| 1 | 9.0180 | 9.0118 | −0.0062 |
| 500 | 2.5807 | 2.6130 | **+0.0323** |
| 1500 | 2.6434 | 2.6610 | +0.0176 |
| 2500 | 2.5580 | 2.5692 | +0.0112 |
| 3500 | 2.5716 | 2.5737 | **+0.0021** |
| 4500 | 2.2803 | 2.2872 | +0.0069 |

Early divergence of +0.0323 at step 500 is **~50× larger than the pre-registered expectation** (spec said `<0.001` at step 500 because of zero-init projection). The gap narrows to +0.0021 by step 3500, then re-widens slightly.

## Interpretation

Two readings of the curve:

1. **"Bigram learns useful signal."** Gap closes ~0.030 nats between steps 500 and 3500 as the embedding table specializes. Bigram then stalls in late training.
2. **"Two different init paths converge to similar solutions."** The early +0.0323 is dominated by RNG-stream drift from adding a new module to the init tree. The gap closure is regression-toward-mean, not bigram-learning.

**A single seed can't distinguish these.** The honest read is: *even if* reading (1) is correct, the composite net effect at endpoint is +0.0025 — the lever, as implemented, does not help on this seed at this hardware/time budget.

## Why not RNG-control retry

Tempting to re-run with `torch.get_rng_state()` / `torch.set_rng_state()` around the `BigramHashEmbedding` construction so spec 008's RNG path is preserved. That would isolate the lever's contribution from the drift artifact. Two reasons not to:

1. **Doesn't reflect shipping reality.** The drift cost is a real cost of the module-addition lever; authors don't RNG-control their claimed deltas either. An RNG-controlled +Δ would be a scientifically cleaner number but not the number to ship on.
2. **Doesn't answer the shipping question.** That needs seed-averaged Δ, which costs ~$60 for 3-seed × 2-spec. That's 40% of remaining budget for a lever whose single-seed point is already on the wrong side.

## Decision

**Kill for this push.** Single-seed +0.0025 doesn't warrant 3-seed confirmation when we have untested candidate levers (spec 014 BPB-weighted CE, others) with comparable expected Δ and no such drift artifact.

Shelve for potential post-deadline research writeup — the "bigram is learning something but doesn't net positive" observation is interesting on its own.

## Next steps

- **Spec 014 (BPB-weighted CE, port #1519)** becomes the next candidate. Move to front of queue.
- **Spec 012 (softer QK_GAIN)** remains shelved until 014 lands.
- **Spec 013 RNG-controlled retry** is NOT scheduled — see reasoning above.

## Cost

~$5 (skip-smoke gamble paid off — 110 LOC of new code ran first try at 8×H100).

Running total after 011 + 013: **~$133 remaining** of the $200 push budget.

## Memory-worthy

Zero-init projections don't prevent RNG-drift from perturbing sibling module init. For future "safe by zero-init" patches to actually be baseline-preserving at init, wrap the new-module construction in `torch.get_rng_state()` / `set_rng_state()` so no downstream RNG-consuming init is perturbed. (Didn't do this for 013 because spec's open question 1 was artifact size, not RNG control; the +0.0323 step-500 gap makes the cost visible in retrospect.)
