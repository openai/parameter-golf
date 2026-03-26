# Winning PR #639 Breakdown

This note assumes the target PR is [#639](https://github.com/openai/parameter-golf/pull/639), the compliant `~1.1175` submission discussed in the upstream thread.

The most important point is:

- `#639` is not one magic trick
- it is a staged stack on top of an already-strong base from [#589](https://github.com/openai/parameter-golf/pull/589)
- the final score comes from reducing the deployed loss, not just improving raw training loss

## What #639 Actually Did

From the PR body and diff, the stack is:

1. Strong inherited base from `#589`
- `11L` model
- `VE128`
- late QAT machinery already present in the base
- strong no-TTT model before the new changes

2. Full GPTQ int6 export
- Hessian collection on calibration batches
- act-order style column handling
- Cholesky-based error compensation
- blockwise GPTQ with `block_size=128`
- final `lzma` compression
- pruning pass to hit the artifact target

3. XSA everywhere, not just a few late layers
- code path makes XSA configurable by `XSA_LAST_N`
- PR writeup says the actual submission used XSA on all `11` layers

4. SWA/EMA smoothing before export
- EMA state with decay `0.997`
- SWA snapshots every `50` steps late in training
- final exported weights are a `50/50` blend of SWA and EMA

5. Score-first legal TTT
- score each chunk first
- then adapt on that chunk
- SGD-based by default, with optional AdamW and EB-TTT controls in code
- PR writeup says the actual submission used a conservative score-first TTT recipe

6. Longer late phase
- the diff moves warmdown from `1200` to `3500`
- the body describes a long late-training phase before export

## What #639 Changed Versus #589

This is the key distinction.

`#589` was already very strong:
- headline: `1.1178` mean with late soft-round QAT + score-first TTT
- base model: already beyond the simple stage2_1 stack

So the `#639` score is not mostly explained by the new TTT or a tiny training tweak.
The meaningful deltas versus `#589` are:

- swap the export path from late-QAT-centered submission logic toward full GPTQ deployment logic
- push XSA more aggressively
- smooth the late trajectory with SWA+EMA specifically for export
- tighten the legal TTT protocol around that deployed model

## What I Think Each Change Contributed

### 1. Full GPTQ

This is the biggest new contribution.

The PR claims:
- naive int6 quantization gap: about `+0.0083 BPB`
- full GPTQ gap: about `+0.0039 BPB`

Why that matters:
- the benchmark is scored on the deployed compressed artifact
- once the base model is already good, cutting deployment damage by a few thousandths is first-order
- this is exactly the sort of effect that a normal training-side screen will miss

My read:
- full GPTQ is the biggest reason the final score gets down into the `1.11x` regime while staying compliant
- it is not making the model fundamentally smarter
- it is preserving more of the trained model through compression

### 2. XSA-All

This is probably the biggest modeling-side delta.

Why it helps:
- it gives more layers access to cross-sequence context
- that means more of the network can use extra context at eval time
- it is not just a late-head refinement

My read:
- XSA-all likely gives a real pre-quant gain
- but its real value is only realized because the rest of the stack preserves that gain through export

### 3. SWA/EMA Blend

This is not exciting by itself, but it is exactly the right kind of helper for this stack.

Why it helps:
- GPTQ and other export steps like smoother weights
- EMA alone picks a stable point
- SWA late in warmdown averages a small local region
- the blend reduces sharp local curvature before quantization

My read:
- this is a deployment helper, not a headline modeling innovation
- it probably matters because the stack is export-sensitive

### 4. Score-First TTT

This is the smallest piece of the story.

The PR body says:
- no-TTT: `1.1182`
- score-first TTT: `1.1175`

So the TTT delta is only about `-0.0007`.

My read:
- TTT is not why this stack is in the `1.11x` band
- TTT is the final polish after the stronger base + export path are already in place
- the main lesson is not "TTT won"
- the main lesson is "the no-TTT deployed model is already very strong"

## Why The Stack Gets `val_bpb` That Low

From first principles, the stack wins because it attacks the full deployed objective in the correct order:

1. Build a stronger model family than the simple control
- `11L`
- VE
- broader context use

2. Shape the late trajectory for deployment
- long warmdown
- EMA
- SWA

3. Preserve the trained model through compression
- full GPTQ
- artifact-budget-aware pruning/compression

4. Only then add tiny eval-time lift
- score-first legal TTT

That is why the score is so low.

The stack is not "one good trick."
It is:

- stronger representation
- smoother late weights
- much lower quantization damage
- tiny final eval lift

## Why This Matters For Us

The difference between this PR and our failed `stage2_1` wave is not just "better ideas."
It is that `#639` is aligned to the real bottleneck:

- it stages the process
- it treats deployment loss as first-order
- it uses the artifact budget aggressively
- it does not confuse throughput wins with score wins

Our `stage2_1` slate mostly tried:
- local training helpers
- local geometry helpers
- throughput helpers

`#639` instead treats the game as:
- first get a stronger model family
- then smooth the late trajectory
- then protect that model through deployment

That is the main lesson to keep.

## Bottom Line

If you compress the story to one sentence:

- `#639` gets to `~1.1175` because it combines an already-strong `#589` base with a much better deployment path, broader context usage, and only a very small amount of legal TTT polish.

If you compress it to one mechanism:

- the single most important extra mechanism in `#639` is full GPTQ on top of a base that is already good enough for that export path to matter.
