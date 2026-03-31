# Stage 3.2 Evolution Strategy

Date: 2026-03-25

This file is intentionally separate from the stage design docs.
It is about how to optimize the `stage3_2` controller space once the DSL exists.

## Objective

Search over bounded state-conditioned controller policies that can beat the strong static default on deployed `val_bpb`.

This is not a hyperparameter sweep.
It is an evolutionary search over:

- controller structure
- controller wiring
- controller phaseing
- controller magnitudes

## Why This Is A Good Evolutionary Space

`stage3_2` has the right properties for mutation search:

- small bounded policy objects
- mixed discrete + continuous decisions
- non-linear interactions
- strong phase sensitivity
- real child composition opportunities

The likely wins are not single coefficients.
They are policy shapes:

- when late pressure turns on
- which families it touches
- how snapshots are captured and chosen
- when expensive actions should back off

## Search Unit

One candidate is one controller spec.

That spec is the genome.

## Genome Schema

The initial genome should include only fields that already clear the bar.

### 1. Phase

- `phase_boundaries`
  - two bounded values: `early -> mid`, `mid -> late`

### 2. Defaults

- `phase_defaults.early`
- `phase_defaults.mid`
- `phase_defaults.late`

Allowed actions in phase defaults:

- `ema_decay`
- `qat_alpha`
- `export_surrogate_weight`
- `checkpoint_capture_rate`
- `checkpoint_selection_mode`
- `token_lr_mult`
- `matrix_lr_mult`
- `scalar_lr_mult`
- `head_lr_mult`
- `freeze_token`
- `freeze_head`

### 3. Gates

Each gate is:

- `feature`
- `op`
- `threshold`
- `action`
- `value`

Allowed features in the first search:

- `progress`
- `step_avg_ms`
- `train_loss_slope`
- `warmdown_frac`

Allowed ops:

- `>`
- `<`

### 4. Snapshot Policy

- `every`
- `start_frac`
- `last_k`
- `score`
- `mode`

Allowed `score`:

- `deployed`
- `raw`

Allowed `mode`:

- `ema`
- `last`
- `best_deployed_last_k`
- `best_raw_last_k`

### 5. Pulse Policy

- `every`
- `late_start`
- `mode`
- `weight`

Allowed pulse `mode` in the first search:

- `export_surrogate`
- `late_qat`

## Search Space Boundaries

The controller must stay small.

Initial hard limits:

- max `2` phase boundaries
- max `3` actions per phase default
- max `2` gates
- max `1` pulse block
- max `1` snapshot policy

This is important.
If the controller becomes too expressive too early, the search becomes mostly noise.

## Seeding Strategy

Do not start from random JSON.

Use structured seeds:

1. `R0A`
- static control

2. `H201`
- late deploy gate

3. `H202`
- best deployed-state controller

4. `H202B`
- best raw-state controller

5. `H204`
- family-split warmdown

6. `H205`
- alternating objective controller

7. `H206`
- systems-aware support controller

8. `R0B`
- repeat control

That gives:

- 2 controls
- 4 lead causal families
- 2 support/diagnostic variants

## Mutation Operators

The search should not mutate everything uniformly.
Use operator classes with different frequencies.

### A. Numeric Mutations

Frequent.

Mutate:

- phase boundaries by small bounded shifts
- gate thresholds
- action magnitudes
- `ema_decay`
- `qat_alpha`
- `export_surrogate_weight`
- `checkpoint_capture_rate`
- pulse cadence
- pulse weight

Examples:

- `late_start: 0.72 -> 0.77`
- `qat_alpha: 0.35 -> 0.22`
- `matrix_lr_mult: 1.10 -> 1.18`

### B. Wiring Mutations

Medium frequency.

Mutate:

- gate feature
- gate action
- gate operator
- whether an action lives in a phase default or a gate

Examples:

- gate `step_avg_ms -> export_surrogate_weight`
- gate `progress -> checkpoint_capture_rate`

### C. Structural Mutations

Rare.

Mutate:

- add/remove one gate
- add/remove pulse block
- switch checkpoint mode
- enable/disable one action in one phase

Examples:

- add late pulse block
- swap `ema -> best_deployed_last_k`
- remove `freeze_head`

## Mutation Schedule

Use an annealed mutation mix:

### Early generations

- `60%` numeric
- `30%` wiring
- `10%` structural

### Mid generations

- `50%` numeric
- `30%` wiring
- `20%` structural

### Composite generations

- `40%` numeric
- `20%` wiring
- `40%` structural

Reason:

- early search should stabilize each causal family
- later search should explore controller compositions

## Evaluation Protocol

Use staged evaluation, not one-shot ranking.

### Wave 1: Primitive Screen

`8 x 1xH100`

- 2 controls
- 6 seeded controller families

Phases:

- `sanity` `90s`
- `screen` `180s`

Primary questions:

- do actions trigger
- does deployed score move
- is throughput still acceptable

### Wave 2: In-Family Evolution

One family at a time.

For each surviving family:

- 1 control repeat
- 1 parent
- 6 mutated children

Goal:

- optimize the family before mixing families

### Wave 3: Cross-Family Compositions

Only proven winners.

Priority compositions:

1. `H201 + H202`
2. `H202 + H204`
3. `H205 + H202`
4. `H201 + H206`

Goal:

- see whether late deploy timing, state selection, and family splitting stack

### Wave 4: Decision

`600s`, `1 GPU` each

This is where:

- snapshot selection becomes real
- family-split warmdown becomes more legible
- alternating objective should separate from noise

### Wave 5: Champion

`600s`, `8xH100`

Only one winner gets this.

## Bar By Generation

The idea bar must stay active after seeding.
It just changes form by stage.

### Generation 0: Seed Admission

Every seed must clear the full idea bar.

Required:

- broken invariant
- concrete mechanism
- causal why
- believable deployed-score path
- expected lift range
- expected horizon
- failure mode
- kill rule

If a seed does not clear that bar, it should not enter the first wave.

### Generation 1-2: Child Admission

Children do not need a brand-new broken invariant.
They are allowed to optimize a parent that already cleared the bar.

But they still need to clear a child bar.

Required:

- explicit parent reference
- statement of what is being optimized
- reason the mutation could change the outcome materially
- expected direction of effect
- expected horizon
- kill rule

Examples that pass:

- later deploy gate because the parent seems early-harmful
- denser checkpoint capture because snapshot choice appears real
- weaker pulse because the parent is throughput-limited

Examples that fail:

- random coefficient drift with no stated reason
- cosmetic mutation of a non-binding threshold
- child that changes several unrelated things at once

### Composite Stage Admission

A composite is not admissible just because two parents both won.

A composite must state:

- which proven parents it combines
- what interaction it is testing
- why the combination could exceed the better parent
- what failure mode is expected if they interfere

Examples that pass:

- `late deploy gate + best deployed-state selection`
  - tests whether late trajectory shaping and late state choice reinforce each other
- `family-split warmdown + best deployed-state selection`
  - tests whether family-specific late motion creates better deployable snapshots

Examples that fail:

- bundle of three winners with no interaction story
- “try all good things together”
- mixing one winner with one unvalidated support idea

### Late-Stage Admission

Later generations are allowed to be narrower.
But they still must be consequential.

Each late-stage candidate must answer one of:

- is this materially optimizing a proven mechanism?
- is this testing a meaningful interaction between proven mechanisms?

If the answer is neither, it should not run.

## Later-Stage Consequence Rule

To prevent the search from collapsing into tiny local moves, require one of:

### A. Mechanism optimization

The mutation changes a decision likely to matter:

- phase boundary
- action enable/disable
- gate wiring
- checkpoint mode
- pulse cadence or weight
- family-specific late multipliers

### B. Interaction test

The mutation tests whether two proven controller stories compose or interfere.

### C. Constraint recovery

The mutation specifically addresses a known failure mode:

- throughput collapse
- late trigger never activating
- snapshot policy degenerating to EMA/final
- excessive controller oscillation

If a candidate is not doing one of these, it is probably below the bar.

## Scoring

Use different ranking logic by horizon.

### Sanity

Kill-only.

Score for survival:

- no crash
- actions trigger at least once if expected
- no catastrophic step-time blowup

Kill if:

- no controller actions ever activate
- step time regresses badly without any score promise
- logs show degenerate snapshot spam

### Screen

Primary metric:

- `post_quant_bpb`

Secondary metrics:

- `delta_quant_gap`
- `step_avg_ms`
- `steps`

Interpretation:

- faster is not enough
- raw `val_bpb` is not enough
- controller survives only if the wallclock budget turns into deployed gain

### Decision

Primary metric:

- `post_quant_bpb`

Secondary metrics:

- chosen snapshot differs from raw final / EMA in meaningful ways
- deploy-vs-raw checkpoint criteria separate
- step cost remains acceptable

### Champion

Primary metric:

- full final deployed score

That is the only metric that matters at the end.

## Selection Rules

Use three simultaneous criteria:

### 1. Fitness

Keep the best deployed performers.

### 2. Novelty

Always preserve one structurally distinct child if it is not clearly dead.

Examples of structural novelty:

- first candidate using `best_deployed_last_k`
- first candidate with a pulse block
- first candidate with family splitting and no deploy gate

### 3. Stability

Prefer candidates whose trigger behavior is interpretable and repeatable.

Do not promote policies that win only through one noisy short-run fluke.

## Archive Rules

Canonicalize every controller before archiving.

Canonicalization should:

- sort gate lists deterministically
- strip no-op actions
- collapse equivalent defaults
- normalize small floats by bounded rounding

Do not keep near-duplicate controllers as separate archive members.

## What To Hold Fixed

In the first real evolutionary pass, do not mutate:

- model architecture
- data path
- tokenizer
- full quantizer implementation
- unrelated static stack helpers

`stage3_2` should answer:

- can dynamic control beat static control

not:

- can any possible code mutation beat the base

## What To Add Later

Only after the first controller wave proves itself:

- `quant_gap_proxy` as a live feature
- curriculum switching
- context switching
- deeper family splitting
- composite late-policy controllers

These are later because they add space faster than they add certainty.

## First Three Generations

### Generation 0

Use the seed pack exactly.

Goal:

- identify real controller families

### Generation 1

For the top `2-3` families:

- mutate boundaries
- mutate magnitudes
- mutate snapshot cadence

No large structural jumps yet.

### Generation 2

Introduce:

- one gate mutation
- one checkpoint mode mutation
- one pulse mutation

Goal:

- test whether controller logic, not just late coefficients, is the source of lift

## Expected Win Pattern

The likeliest real winner from this search should look like:

- normal early training
- mild mid tracking
- strong but bounded late deploy alignment
- nontrivial late snapshot choice
- some family-specific late damping

If the winner instead looks like:

- always-on pressure from step 0
- many gates constantly flipping
- large throughput regression

then the search is probably overfitting noise.

## Short Version

The right strategy is:

1. evolve controller families, not raw knobs
2. stabilize families before composing them
3. rank by deployed score first
4. preserve structural novelty
5. keep the controller DSL small until a real win appears
