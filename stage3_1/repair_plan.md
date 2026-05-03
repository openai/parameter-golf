# Stage 3.1 Repair Plan

Date: 2026-03-25

## Goal

Bring `stage3_1` up to the current bar:

- lane claims must match implementation
- one-command execution must match the documented stage flow
- early packs must preserve observability
- the lead hypotheses must be wide enough to matter against the current SOTA-aligned base

## Priority 0: Make The Stage Honest

### P0.1 Implement a real Lane B export-only mode

Problem:

- `lane_b_bakeoff` is described as checkpoint-only export/eval
- the current code still runs the full training script

Files:

- [orchestrate_stage3_1.py]( nanoevolve/pgolf/parameter-golf/stage3_1/orchestrate_stage3_1.py)
- [base_train_gpt.py]( nanoevolve/pgolf/parameter-golf/stage3_1/base_train_gpt.py)
- [run_configs.json]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json)

Change:

- add an explicit export-only entry mode
- require a checkpoint path for Lane B
- skip the training loop entirely
- run only:
  - load checkpoint
  - apply export patch
  - quantize/export
  - roundtrip eval

Acceptance:

- `lane_b_bakeoff` logs no training steps
- wallclock is materially below a 600s train run
- Lane B comparisons are now same-checkpoint comparisons

### P0.2 Make `--phase all` actually execute the documented stage

Problem:

- config declares `lane_b_bakeoff`, `composite`, and `decision`
- `--phase all` skips them

Change:

- define the actual stage order explicitly
- one-command path should run:
  1. `sanity`
  2. `screen`
  3. `lane_b_bakeoff`
  4. `composite`
  5. `decision`
  6. `champion_8x`

Acceptance:

- `--phase all --dry-run` shows every declared stage in order
- `all` and the docs describe the same tournament

## Priority 1: Fix Observability

### P1.1 Stop mixing Lane A and Lane B in the same short ranking pack

Problem:

- export-only ideas and training ideas are screened together in `sanity` and `screen`
- their signals are not comparable at the same horizon

Change:

- split screens into:
  - `screen_train`
  - `screen_export`
- or keep one shared sanity crash pass but use separate score summaries

Acceptance:

- Lane B is ranked against matched Lane B controls
- Lane A is ranked against matched Lane A controls

### P1.2 Add explicit matched controls per lane

Change:

- Lane A:
  - 2 controls
  - 6 train candidates
- Lane B:
  - 2 controls on the same checkpoint
  - 6 export candidates

Acceptance:

- no candidate is compared against the wrong control family

## Priority 2: Raise Hypothesis Consequence

### P2.1 Promote only the strongest process-level ideas as leads

Keep as lead ideas:

- `quant_anneal`
- `staged_objective`
- `fisher_bit_allocation`
- `companding`

Demote to support ideas:

- `byte_weighted_loss`
- `sparsify_5pct`
- `companding_plus_sparsify`

Reason:

- the first group changes a first-order story
- the second group is more incremental or conditional

### P2.2 Add at least one broken-invariant family that is missing today

Best candidates:

- checkpoint selection
- staged curriculum
- parameter-family specialization

Minimum standard:

- at least one new family must change process control, not just loss/export math

Acceptance:

- the stage contains at least one hypothesis from:
  - export function/policy
  - training process split
  - objective schedule
  - state/checkpoint selection

## Priority 3: Redesign The Tournament

### P3.1 Recommended stage structure

Use this flow:

1. `sanity_train`
2. `sanity_export`
3. `screen_train`
4. `lane_b_bakeoff`
5. `composite`
6. `decision`
7. `champion_8x`

### P3.2 Finalist construction

Finalists should be:

- 2 controls
- best train-process winner
- best export winner
- best objective/process child
- best export child
- 2 composites

Not:

- just the top `k` from one mixed short screen

## Priority 4: Add Explicit Pass Criteria

`stage3_1` should only be considered fixed if all are true:

- Lane B is truly export-only
- `--phase all` runs every declared stage
- train and export lanes are screened separately
- one-command tournament still fills the 8 GPUs at each stage
- at least 4 finalist slots are consequence-level mechanisms, not local tweaks

## Suggested Implementation Order

1. implement real Lane B export-only mode
2. fix `--phase all`
3. split train/export screens
4. rebuild finalist logic around lane winners
5. prune weak lead hypotheses
6. add one missing broken-invariant family

## Short Version

`stage3_1` does not need a cosmetic cleanup.

It needs:

- honest lane execution
- honest one-command staging
- stronger observability separation
- a smaller number of more consequential lead ideas
