# Stage 3.4 Bar Assessment

Date: 2026-03-25

## Verdict

`stage3_4` clears the idea bar as a separate stage.

It is materially different from `stage3_2` because it does not optimize one trajectory.
It changes the process by allocating most of the budget to a shared trunk and the rest to multiple late finishers.

The core broken invariant is:

- one late trajectory should be chosen in advance

That is a real false invariant and a first-order process story.

## Why This Clears The Bar

It clears because it:

- changes the unit of optimization from one run to a shared-trunk late-branch program
- attacks a full-pipeline failure mode
- has a believable path to lower deployed `val_bpb`
- is large enough to matter against a strong static base

## Expected Lift

### H401

- `0.004 - 0.012 BPB`
- Why:
  - if late policy uncertainty is real, picking the best of three plausible finishers should beat committing to one

### H402

- `0.003 - 0.010 BPB`
- Why:
  - if H401 is starved, earlier branching gives each finisher enough time to matter

### H403

- `0.003 - 0.011 BPB`
- Why:
  - fewer finishers means deeper finishers

### H404

- `0.002 - 0.008 BPB`
- Why:
  - tests whether the real value is in mechanism-heavy late finishers

### H405

- `0.002 - 0.007 BPB`
- Why:
  - isolates whether export-state style is itself a branch dimension

### H406

- `0.003 - 0.013 BPB`
- Why:
  - branching may allow harder late swings than a single trajectory would tolerate

## What Would Falsify The Stage

The stage would look weak if:

- the selected branch is almost always the trivial EMA branch
- branching breadth always loses to a single strong late path
- earlier branching consistently weakens the shared trunk too much
- branch winners do not separate at long horizon
