# Next Build Plan — 2026-03-20

This note translates the current PR audit into a concrete next model direction.

## Decision

Do not keep iterating on the old `10L` leader-core validity lane as the main branch.

Start a new primary candidate around the current durable frontier:

- `11L`
- `512d`
- `GQA 8/4`
- `MLP 3x`
- `TRAIN_BATCH_TOKENS=524288`
- `MUON_WD=0.04`
- `ADAM_WD=0.04`
- `MATRIX_LR ~= 0.025`
- `SCALAR_LR ~= 0.025`
- `TIED_EMBED_LR ~= 0.03-0.035`
- late SWA
- careful post-quant artifact measurement

## First implementation target

Base the first new candidate on the portable intersection of:

- [PR #236](https://github.com/openai/parameter-golf/pull/236): smaller batch for more useful steps
- [PR #198](https://github.com/openai/parameter-golf/pull/198): strongest overall `11L + WD + SWA` stack
- [PR #179](https://github.com/openai/parameter-golf/pull/179): best older example of a portable non-sliding `11L` line

## What to include in the first pass

1. `11L` depth
2. `WD ~= 0.04`
3. `524k` train batch
4. late SWA
5. export discipline with strict roundtrip validation

## What to defer from the first pass

1. QAT
   Reason: not enough clear portable benefit for the complexity cost.

2. Mixed `int5/int6`
   Reason: useful only if bytes become the dominant bottleneck again.

3. Sliding-window-dependent claims
   Reason: useful for scoring context, but not the main target when deciding whether the model family is actually stronger.

4. Paid-prefix / answer-storage paths
   Reason: this is a separate rules-sensitive strategy, not the current model-improvement path.

## SmearGate and BigramHash

Keep them as optional, but do not let them dominate the branch design.

Current read:
- likely helpful
- widely adopted
- not obviously the primary source of the latest leap

So the right framing is:
- include them if the branch can support them cleanly
- do not mistake them for the main reason the frontier moved

## Success criteria for the next run

The next branch should be judged first by:

1. step count reached in `600s`
2. non-sliding post-quant roundtrip quality
3. artifact size margin
4. only then sliding-window score

## Working hypothesis

The main thing we were missing was not another export tweak or another small schedule ablation on the old family. It was that the competition’s center of gravity moved to:

- deeper models
- stronger WD
- more updates in fixed time

That is the branch to build next.
