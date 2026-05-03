# Hypothesis Generation Failures

This note is not about one bad run. It is about the generator that produced the run.

The core problem is:

- we are not sampling a wide enough set of mechanisms
- the mechanisms we do sample are usually too local
- so even when the search is disciplined, it mostly searches inside a weak neighborhood

## Core Diagnosis

Our hypothesis generation is biased toward:

- helper patches
- local refinements
- schedule nudges
- mild architecture tweaks
- mechanisms that help one metric lane in isolation

It is weak at generating:

- process splits
- objective splits
- budget reallocations
- state-selection mechanisms
- hypotheses that change the dominant failure mode

That is why we keep finding ideas that are:

- plausible
- measurable
- occasionally positive in short runs

but still too small to beat the current SOTA-aligned default at the `600s` deployed horizon.

## What The Winning PRs Imply

The strongest upstream stacks do not win because they found one more helper.

They usually do some combination of:

- start from a stronger architecture family
- change the late training regime
- reduce deployment damage aggressively
- use the artifact budget more intelligently
- separate raw training quality from deployed score quality

So the right lesson is:

- the frontier is not mostly operating at the "small local tweak" scale
- our generator mostly is

## What We Over-Generate

These idea classes are overrepresented in our search:

- `tighten`
  - more decay
  - more clipping
  - slightly different momentum or warmdown
- `specialize`
  - small per-layer or per-family variants that keep the same causal story
- `borrow`
  - ideas copied from a PR without checking whether they are first-order or just stack-specific helpers

These are not useless. They are just too small to be the main engine of discovery against a strong default.

## What We Under-Generate

These idea classes are underrepresented and should be forced into every next-wave slate:

- `stage`
  - different rules in warmup, bulk training, and late phase
- `reallocate`
  - move time, bytes, steps, or capacity from one subsystem to another
- `externalize`
  - shift work into export, eval, or checkpoint selection
- `internalize`
  - pull deployment loss into training
- `factorize`
  - separate tensor families, checkpoints, contexts, or objectives
- `invert`
  - challenge the current dominant story instead of polishing it

If we do not force these classes, the generator collapses into local mode.

## The False Invariants We Keep Accepting

These are the assumptions our generator keeps leaving intact:

- one training regime for the entire run
- one data order for the entire run
- one objective: fit first, deploy later
- one checkpoint: use the last one
- one optimizer law for all parameter families
- one context budget for all phases
- one export target chosen implicitly rather than explicitly

The next real wins are likely to come from breaking one or more of these.

## The Real Failure Mode

We keep asking:

- what patch could help this stack?

We should be asking:

- what assumption about this process is false?

That difference matters.

A local patch question produces:

- better helper ideas
- cleaner ablations
- narrower search

A false-invariant question produces:

- bigger mechanisms
- more consequential mutations
- hypotheses that can survive a strong default

## New Acceptance Rule

A hypothesis should not enter the next stage unless it satisfies all of:

- distinct causal story from the current default
- plausible path to lower deployed compressed score
- meaningful effect size if true
- observable at the intended horizon
- not reducible to a coefficient tweak of an already-tested idea

If it does not clear that bar, it is not a next-stage hypothesis. It is a local variant.

## Replacement Generator Rubric

For every future stage, force at least one candidate from each bucket:

1. Phase-split mechanism
- Different rule for warmup, bulk, and late phase.

2. Deploy-alignment mechanism
- Explicitly reduce train-to-export or raw-to-deployed mismatch.

3. Budget-reallocation mechanism
- Move bytes, time, context, or capacity from one subsystem to another.

4. Checkpoint/export-selection mechanism
- Change which weight state is exported or how it is chosen.

5. Parameter-family split mechanism
- Different late behavior for embeddings, trunk matrices, controls, or value paths.

6. Data-order split mechanism
- Different data regime or ordering by phase.

7. Wildcard anti-dominant mechanism
- A hypothesis that directly challenges the current leading story.

Only after generating one candidate from each bucket should we prune redundancy.

## What This Would Have Prevented

If we had applied this rubric earlier, `stage2_1` would not have been dominated by:

- small training-side helpers
- throughput interpreted as a score mechanism
- static curriculum variants
- local geometry patches without a larger process change

It would have forced us to test:

- late deploy alignment
- checkpoint selection
- two-stage curriculum
- phase-specific optimizer law
- parameter-family late freeze

Those are much closer to the scale of the winning upstream ideas.

## Practical Rule For Evolutionary Search

The search should stop mutating only:

- model details
- scalar hyperparameters
- local helper patches

It should start mutating:

- the training process
- the timing of objectives
- the object that gets exported
- the allocation of budget across phases and tensor families

That is the level where an evolutionary winner is still plausible.
