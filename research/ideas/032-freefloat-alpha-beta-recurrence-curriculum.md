# Idea 032 — Free-floating alpha/beta under `NUM_LOOPS=3` + recurrence curriculum

## Thesis

`029` likely regressed because it reused frozen carry constants that were calibrated in the older 2-loop regime. The next clean test is a **free-floating** carry calibration run in the real target regime:

- `NUM_LOOPS=3`
- recurrence curriculum active
- detached carry
- learnable `alpha` and `beta`
- no TTT

The point of this run is not submission quality by itself. The point is to measure what carry structure the 3-loop regime actually wants, then freeze those learned values into a follow-up run.

## Why separate from `031`

`031` is the clean 2-loop direct-carry probe. It isolates carry behavior in the simpler regime and stays close to the successful `025b/025c` lineage.

This idea is different:

- it re-enters the 3-loop regime
- it includes recurrence curriculum again
- it is specifically about re-learning `alpha/beta` under that harder regime

So it should be a separate spec, not a mutation of `031`.

## Proposed parameterization

Start from the `024c` style family rather than the `031` direct-carry family:

- per-pass learnable `alpha`
- per-pass learnable `beta`
- detached carry
- neutral init:
  - `alpha = 0`
  - `beta = 1`

Reason:

- one run answers both questions at once:
  - what values should be frozen for the 3-loop regime?
  - do passes actually want different carry patterns?

If the learned pass-specific patterns collapse toward the same structure, the frozen follow-up can simplify back toward shared indexing. If they diverge, the frozen follow-up should stay per-pass.

## Runtime shape

Use a 4×H100 calibration run with about 20% extra wallclock relative to the normal screen, but preserve recurrence onset at the same absolute step:

- target loop start remains around step `~2100`
- after extension, `ENABLE_LOOPING_AT` must be recomputed from:

```text
new_loop_start_ratio = 2100 / new_total_steps
```

If the run is expected to reach about `6000` total steps, this implies:

```text
ENABLE_LOOPING_AT = 0.35
```

The purpose of the extension is to create more **post-onset learning time** for `alpha/beta`, not to delay onset.

## What to inspect after the run

- final `val_bpb`
- `alpha` and `beta` trajectories over time
- whether the third pass learns something meaningfully different
- whether the late-stage drift is low enough to justify freezing
- whether the depth-upgraded portion of training changes the learned pattern materially

## Decision rule after calibration

- if pass-specific rows/matrices are clearly different: freeze a per-pass version
- if they are near-identical: freeze a shared version
- if parameters are still drifting hard at the end: rerun longer before freezing
