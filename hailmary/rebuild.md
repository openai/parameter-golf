# Hailmary Rebuild

This document states plainly where `hailmary` does and does not satisfy the new initial-idea bar.

## What Still Holds

`hailmary` is still stronger than `stage2_1` on:

- first-principles framing
- code-mandatory hypotheses
- coverage of export and eval lanes
- willingness to include genuinely high-upside mechanisms like `Full GPTQ`

## What Does Not Hold Yet

The current runnable slate is still biased toward ideas that were easy to patch.

That means it overrepresents:

- export refinements
- eval-policy changes
- local geometry effects
- throughput probes

and underrepresents:

- process splits
- objective splits
- checkpoint-selection mechanisms
- parameter-family late rules
- data or context staging by phase

So the docs were ahead of the runnable slate.

## The Rebuild Target

The correct `hailmary` target is a moonshot stage whose lead packs are built around broken invariants, not just patch convenience.

The most important false invariants are:

- one training regime for the whole run
- one objective for the whole run
- one checkpoint is always the right export target
- one data order is used for the whole run
- one context budget is used for the whole run
- all tensor families should keep adapting the same way late in training

## The Right Lead Families

The lead `hailmary` families should now be:

1. late deploy alignment
2. deployed checkpoint selection
3. two-stage curriculum
4. parameter-family late freeze
5. two-stage context budget
6. alternating objective microcycles

The older `geometry` and `throughput` packs are still useful, but they are now support lanes, not lead lanes.

## Implementation Policy

Do not pretend these rebuilt families are runnable if they are not.

For `hailmary`, the right pattern is:

- keep current runnable packs available
- add at least one runnable lead rebuild pack
- mark missing mechanisms `needs_patch`
- only promote them into the active tournament once the patches exist

This keeps the folder honest while still moving the search toward the right mechanism scale.

## Current Code State

That policy is now reflected in code:

- `phase_split` is the new default pack and is fully runnable
- `checkpoint_selection` and `staged_curriculum` are now also runnable lead packs
- `alternating_objective`, `moonshot_core`, `moonshot_second_wave`, `moonshot_geometry`, and `moonshot_throughput` remain runnable support packs
- `parameter_family_split` and `context_stage` remain explicit rebuild lanes in [`run_configs.json`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/run_configs.json)
- the tournament runner now reads lead/support/deferred pack roles from config instead of hardcoding the older moonshot pack list

## Practical Rule

If a proposed `hailmary` idea is only:

- a local helper
- a throughput multiplier
- a geometry tweak
- or a stack-specific polish

then it is not a lead moonshot anymore.

Lead moonshots must change:

- the process
- the objective timing
- the exported object
- or the budget allocation across phases and tensor families
