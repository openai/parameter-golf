# Idea 034d — LR plateau around loop onset on `034`

## Thesis

The possible problem is not just that the tail gets too cold.

It may be that the model hits a regime change exactly when LR is already
dropping:

- looping turns on at `ENABLE_LOOPING_AT`
- the optimizer is already partway into warmdown

So instead of globally raising the whole tail with `MIN_LR`, we can target the
transition itself:

- hold LR flat for a short window around loop onset
- then resume the usual decay

## Why this is interesting

- more targeted than `MIN_LR`
- fits the intuition that loop onset is the hard transition
- could preserve normal early training and normal late cooling outside the
  transition window

## Important codebase constraint

In this stack, loop onset is wallclock-fraction based, not raw-step based.

So any plateau should also be defined in wallclock-fraction terms.

For the `034` live-like regime:

- `MAX_WALLCLOCK_SECONDS=1200`
- `ENABLE_LOOPING_AT=0.35`

That means the natural plateau should be centered around wallclock fraction
`0.35`, not around an assumed step number.

## First simple shape

The simplest useful first version:

- normal schedule before `plateau_start`
- constant multiplier during `[plateau_start, plateau_end]`
- resume the usual post-warmdown decay afterward using paused-time semantics

Example first guess:

- `plateau_start = 0.35`
- `plateau_end = 0.45`
- `plateau_value = LR multiplier at onset`

This is intentionally simple:

- narrow window
- no extra ramp segments
- easy to reason about

Why this first window:

- it starts exactly when looping turns on
- it gives a clean post-onset adaptation interval
- it makes comparison against the no-plateau `034` baseline easier to interpret

Important subtlety:

- the sensible implementation is the paused-time version
- that avoids a sharp LR cliff after `plateau_end`
- but it also means the final tail stays hotter than plain `034`, because the
  schedule is effectively shifted later by the plateau width

## Why not start here before `MIN_LR`

`MIN_LR` is still the cheaper first test because:

- already implemented
- one env knob
- no schedule patch

So this plateau idea should come after the first `MIN_LR` rung unless the
schedule evidence strongly points to the transition being the real issue.

## Main question

Is the gain, if any, coming from:

- a globally hotter tail

or from:

- extra learning capacity exactly at the recurrence transition?

This idea is the clean way to test the second hypothesis.
