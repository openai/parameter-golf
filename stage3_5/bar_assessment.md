# Stage 3.5 Bar Assessment

Date: 2026-03-25

## Verdict

`stage3_5` clears the idea bar.

It is larger than `stage3_4` in the same branching realm because it changes three things at once:

- branch timing becomes state-conditioned
- late finishers become a small DSL rather than one fixed set
- export-state choice becomes part of the branch tournament

The core broken invariants are:

- branch timing should be fixed in advance
- one export-state style should be chosen in advance
- late branching should only compare one hand-written ending per run

## Why This Clears The Bar

It clears because it:

- changes the unit of optimization from a single late path to an event-triggered branch tournament
- attacks training-state uncertainty and export-state uncertainty together
- has a believable route to lower deployed `val_bpb`
- is large enough to matter against a strong default

## Expected Lift

### H501

- `0.005 - 0.015 BPB`
- Why:
  - combines adaptive branch timing with within-branch export selection

### H502

- `0.004 - 0.013 BPB`
- Why:
  - tests whether branch depth is the real missing variable

### H503

- `0.005 - 0.017 BPB`
- Why:
  - uses branching to make aggressive late finishers safe enough to try

### H504

- `0.003 - 0.009 BPB`
- Why:
  - directly tests whether export-state style dominates late performance

### H505

- `0.003 - 0.010 BPB`
- Why:
  - removes the trivial fallback and forces real nontrivial late-mechanism competition

### H506

- `0.004 - 0.012 BPB`
- Why:
  - tests whether hybrid adaptive-plus-failsafe triggering is more robust than one trigger style alone

## What Would Falsify The Stage

The stage would look weak if:

- branch timing always collapses to the same effective point
- export-mode selection is almost always irrelevant
- the winning branch is almost always the conservative EMA branch
- the extra branching logic consistently underpays the trunk
