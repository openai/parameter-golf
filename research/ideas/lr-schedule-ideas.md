# LR Schedule Ideas

This note collects small LR-schedule levers that are interesting enough to keep
on the frontier, but do not all need immediate specs.

## 1. `MIN_LR` floor

Keep late-training LR above zero instead of annealing all the way down.

Why it is interesting:

- very cheap to test
- already supported by the current code
- likely to matter in short wallclock-capped runs
- easy to layer onto existing frozen-carry lines like `034`

Current natural first test:

- `034 + MIN_LR=0.10`

Useful ladder:

- `0.05`
- `0.10`
- `0.15`

## 2. Plateau around loop onset

Instead of making the whole tail hotter, hold LR flat for a short window around
the point where looping turns on.

Why it is interesting:

- loop onset is a regime change
- the model may need a brief adaptation window right when recurrence appears
- this is more targeted than a global `MIN_LR` floor

Conceptual schedule shape:

- normal decay before loop onset
- flat LR window spanning the loop-activation zone
- resume decay afterward

Important implementation note:

- in this codebase, loop onset is wallclock-fraction based
- so any plateau should be defined against wallclock fraction, not raw step

Open design choices:

- plateau width
- plateau level
- whether to center exactly on `ENABLE_LOOPING_AT` or start slightly before it

## 3. Plateau around later recurrence/depth transition

This is the same idea as above, but for a later curriculum transition rather
than the first loop onset.

Why it is more speculative:

- more relevant to `NUM_LOOPS=3` / recurrence-curriculum lines like `032`
- harder to transfer cleanly
- less urgent while the frontier is staying on `NUM_LOOPS=2`

## Current priority

If we are choosing only one LR idea first:

1. `MIN_LR` floor
2. loop-onset plateau
3. later-transition plateau

## Notes

- These ideas are schedule levers, not architectural levers.
- They are especially attractive when the structural line already looks decent
  but slightly underconverged.
- They should usually be tested on a stable frozen-carry branch rather than on
  a moving calibration line.
