# Stage 3.2 Controller DSL

## Target

Optimize final deployed `val_bpb` under:

- `600s` wallclock
- fixed artifact cap
- real training + export pipeline
- limited expensive evaluations

The controller must stay:

- bounded
- interpretable
- cheap to evaluate
- evolvable without exploding the search space

## Core Abstraction

A candidate is a bounded policy:

- state features
- gates over those features
- a small set of actions
- optional phase transitions

The controller does not directly set weights.
It only modulates training/export behavior.

## Design Rules

1. No free-form neural controller in v1.
2. No per-layer unconstrained policies in v1.
3. No more than 5 state features active per candidate.
4. No more than 4 actions active per candidate.
5. No action can change every step without smoothing or hysteresis.

## Observable State Features

Only cheap features are allowed in v1.

### Phase Features

- `progress`
  - `step / iterations`
- `warmdown_frac`
  - derived from LR multiplier / remaining wallclock
- `late_phase`
  - binary gate from progress or warmdown

### Optimization Features

- `train_loss_ema`
- `train_loss_slope`
  - short EMA slope or delta over window
- `grad_norm_ema`
- `update_norm_ema`

### Deploy Features

- `quant_gap_proxy`
  - cheap surrogate for post-quant damage
- `ema_raw_gap`
  - difference between EMA and raw short eval
- `artifact_size_proxy`

### Systems Features

- `step_avg_ms`
- `effective_steps_left`

### Family Features

- `embed_drift`
- `matrix_drift`
- `control_tensor_drift`

## Allowed Actions

Actions must hit first-order levers.

### Deploy Alignment

- `late_qat_alpha`
- `export_surrogate_weight`
- `checkpoint_capture_rate`
- `checkpoint_selection_mode`

### State Selection

- `ema_decay`
- `ema_enable`
- `swa_enable`

### Geometry / Stability

- `weight_perturb_scale`
- `grad_centralization_enable`
- `grad_clip_mult`

### Data / Context

- `curriculum_mode`
- `context_mode`
- `eval_stride_mode`

### Family-Specific Controls

- `matrix_lr_mult`
- `embed_lr_mult`
- `scalar_lr_mult`
- `matrix_decay_mult`
- `embed_freeze`
- `head_freeze`

## Policy Form

Version 1 uses a bounded hybrid:

### 1. Global Phase Skeleton

At most 3 phases:

- `early`
- `mid`
- `late`

Each phase has:

- default action values
- optional gate overrides

### 2. Gate Rules

Each gate is:

- one feature
- one threshold
- one direction (`<`, `>`)
- one bounded override

Examples:

- if `quant_gap_proxy > 0.007`, raise `export_surrogate_weight`
- if `step_avg_ms > threshold`, disable heavy deploy pulses
- if `progress > 0.75`, turn on checkpoint capture

### 3. Hysteresis

Each action can have:

- `hold_steps`
- `max_delta_per_update`

This prevents thrashing.

## Candidate Schema

A candidate should contain:

- `phase_boundaries`
- `active_features`
- `active_actions`
- `phase_defaults`
- `gates`
- `hysteresis`

## Bounded Ranges

Initial suggested bounds:

- `late_qat_alpha`: `0.0 - 1.0`
- `export_surrogate_weight`: `0.0 - 0.5`
- `ema_decay`: `0.990 - 0.9999`
- `weight_perturb_scale`: `0.0 - 0.03`
- `matrix_lr_mult`: `0.5 - 1.5`
- `embed_lr_mult`: `0.0 - 1.5`
- `grad_clip_mult`: `0.5 - 2.0`

Discrete action sets:

- `curriculum_mode`: `sorted | reverse | size_desc | shuffle | staged`
- `context_mode`: `base | xsa4 | xsa_all | staged`
- `checkpoint_selection_mode`: `last | ema | best_raw_last_k | best_deployed_last_k`

## Mutation Operators

Evolution should mutate policy structure, not just magnitudes.

### Structure Mutations

- add/remove a gate
- swap the feature used by a gate
- swap the action controlled by a gate
- move a phase boundary
- toggle an action on/off

### Parameter Mutations

- threshold perturbation
- action magnitude perturbation
- hysteresis width perturbation

### Constraint Mutations

- promote a support action to a lead action
- demote a heavy action if systems pressure is too high

## Negative Knowledge

Do not allow:

- full per-layer continuous control in v1
- controller access to hidden activations
- unbounded action values
- dozens of gates
- actions with no direct link to deployed score

## What Counts As A Good Controller

A good candidate:

- changes a first-order causal story
- is cheap enough to survive the wallclock game
- improves deployed score, not just raw loss
- remains interpretable after mutation
