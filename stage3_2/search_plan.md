# Stage 3.2 Search Plan

## Objective

Search over bounded state-conditioned controllers that can beat strong static defaults on final deployed `val_bpb`.

## Why Evolution Here

This is a good evolutionary target because:

- the controller has bounded structure
- many decisions are discrete
- interactions are non-linear
- manual tuning is likely to miss good phase/gate combinations

## Candidate Evaluation Lanes

### Train-Screen

Use short runs to measure:

- learning speed
- step cost
- quant-gap proxy
- whether gates trigger sensibly

### Late-State Screen

Use medium runs to measure:

- checkpoint/state behavior
- late deploy alignment
- family split effects

### Final Aligned Run

Use full `600s` runs for:

- deployed score
- artifact size
- state-selection payoff

## Tournament Structure

### Wave 1: Primitive Controllers

8 x `1xH100`

- 2 controls
- 6 single-family controllers

Goal:

- identify which controller families deserve deeper search

### Wave 2: Policy Children

8 x `1xH100`

- mutate the best primitive families
- vary thresholds, actions, phase boundaries

Goal:

- learn whether dynamic control is truly better than static policies

### Wave 3: Composites

8 x `1xH100`

- combine only proven winners
- no speculative compounds

Goal:

- find a controller that beats the best primitive family

### Wave 4: Final Single-GPU

Long `600s` runs on the finalists.

### Wave 5: Champion 8xH100

Best controller gets the full-box run.

## Mutation Priorities

Mutate in this order:

1. phase boundaries
2. gate thresholds
3. action magnitudes
4. action enable/disable
5. feature/action wiring

Avoid large structural jumps too early.

## Kill Rules

Kill immediately if:

- gate logic never activates
- `step_avg_ms` regresses badly without score compensation
- controller oscillates actions too often
- deployed score is worse than control noise floor

Do not carry dead controller families into composite waves.

## Pass Criteria

`stage3_2` is successful only if it finds at least one controller that:

- beats the strong static control at `600s`
- remains interpretable
- wins on deployed score, not just raw loss
- survives rebasing onto a frontier-aligned stack

## Short Version

`stage3_2` should search small dynamic controllers, not giant policy spaces.

The point is to evolve:

- when to do something
- for which families
- under what state conditions

not to evolve an opaque black-box scheduler.
