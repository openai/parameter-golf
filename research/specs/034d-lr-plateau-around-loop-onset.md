# Spec 034d — LR plateau around loop onset on `034`

**Slug:** `lr-plateau-around-loop-onset`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/034d-lr-plateau-around-loop-onset`
**Commit:** `d856957`
**Links to:** `research/ideas/034d-lr-plateau-around-loop-onset.md`, `research/ideas/lr-schedule-ideas.md`, `research/specs/034-frozen-direct-carry-from-031a.md`, `research/specs/034c-min-lr-on-034.md`

## Hypothesis

The key schedule problem on `034` may be the loop-onset transition itself, not
the whole tail. A short LR plateau around `ENABLE_LOOPING_AT` may outperform a
global `MIN_LR` floor by concentrating learning capacity exactly where the
regime changes.

## Baseline

Direct baseline:

- corrected `034`

Pinned lineage:

- branch `exp/034d-lr-plateau-around-loop-onset`
- runnable code commit `d856957`
- inherited `034` stack from `exp/034-frozen-direct-carry-from-031a`

Schedule comparison baseline:

- `034cB` with `MIN_LR=0.10`

This spec is best interpreted only after at least the center `MIN_LR` rung is
known.

## Config diff

Requires a small schedule patch on top of the existing `034` code path.

Proposed envs:

- `LR_PLATEAU_ENABLED`
- `LR_PLATEAU_START`
- `LR_PLATEAU_END`

Only intended diffs from corrected `034` are:

- plateau code present via the pinned `034d` branch/commit
- `LR_PLATEAU_ENABLED=1`
- `LR_PLATEAU_START=0.35`
- `LR_PLATEAU_END=0.45`

Everything else must remain identical to the inherited `034` stack:

- dataset/tokenizer paths
- CaseOps / gated-attn / quant-gate settings
- model width/depth/head counts
- quantization bits and clip sigmas
- TTT settings and phase count
- shard selection and validation token count
- any other env not explicitly changed in this spec

Pinned semantics:

- before `LR_PLATEAU_START`: use the normal schedule
- between `LR_PLATEAU_START` and `LR_PLATEAU_END`: hold LR constant at the
  schedule value reached at `LR_PLATEAU_START`
- after `LR_PLATEAU_END`: resume the normal schedule with **paused-time**
  semantics

Paused-time semantics means:

- let `plateau_width = LR_PLATEAU_END - LR_PLATEAU_START`
- after the plateau, evaluate the baseline schedule at:
  - `effective_frac = frac - plateau_width`

This is the intended behavior because it avoids a sharp LR cliff immediately
after the plateau.

## First probe

Use a single simple first rung:

- `LR_PLATEAU_ENABLED=1`
- `LR_PLATEAU_START=0.35`
- `LR_PLATEAU_END=0.45`

Plateau value:

- freeze at the schedule multiplier reached at `LR_PLATEAU_START`

That keeps the patch simple and avoids adding another amplitude hyperparameter
in the first test.

Why this first window:

- it starts exactly at the loop-onset event
- it gives a direct post-kick adaptation interval
- it makes comparison against the plain `034` baseline easier to interpret

Important consequence:

- with paused-time semantics, the end LR is **not** the baseline end LR
- at wallclock end, the schedule behaves like the baseline evaluated at
  `1.0 - plateau_width`
- for a `0.35 -> 0.45` plateau, that means the end LR matches the baseline at
  effective fraction `0.90`

So this is not purely local; it also keeps the very end of training hotter than
plain `034`.

## Regime

Keep the `034` stack fixed:

- `DIRECT_CARRY_MODE=frozen_edge_self`
- `NUM_LOOPS=2`
- `MAX_WALLCLOCK_SECONDS=1200`
- `ENABLE_LOOPING_AT=0.35`
- `TTT_ENABLED=1`
- `PHASED_TTT_PREFIX_DOCS=2000`
- `PHASED_TTT_NUM_PHASES=3`
- `DATA_DIR=/workspace/parameter-golf/data`

## Why wallclock fractions matter

In this codepath, loop onset is compared against wallclock fraction.

So the plateau must be specified in wallclock-fraction terms too.

Do **not** define this plateau by assumed training steps.

## Hardware ladder

1. `4×H100`, `1200s`, full pipeline

This should be a single-rung test first. If it is promising, later variants can
change plateau width.

## Run protocol

Single first rung:

- `034dA`

Execution rule:

- launch from `exp/034d-lr-plateau-around-loop-onset`
- use runnable code commit `d856957`
- only schedule diffs allowed are the three plateau envs above
- if the produced `config.json` differs from inherited `034` on anything else,
  the rung is invalid and must be aborted/relaunched

Pinned command shape after patch lands:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_034d_a \
DATA_DIR=/workspace/parameter-golf/data \
ARTIFACT_DIR=/workspace/runs/034d-lr-plateau-around-loop-onset/run_a/seed_314 \
DIRECT_CARRY_MODE=frozen_edge_self \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
LR_PLATEAU_ENABLED=1 LR_PLATEAU_START=0.35 LR_PLATEAU_END=0.45 \
MAX_WALLCLOCK_SECONDS=1200 SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## What to watch

- pre-EMA endpoint loss
- post-EMA pre-quant diagnostic
- quantized diagnostic
- post-TTT `val_bpb`
- whether the plateau helps more cleanly than `MIN_LR`

## Required artifacts

- `final_model.pt`
- `final_model.int6.ptz`
- training log
- final metrics JSON
- `config.json`

## Sanity gate before accepting the rung

Before comparing to `034` or `034c`, execution must verify from `config.json`
that the only intentional diffs are:

- the plateau-support code lineage
- `LR_PLATEAU_ENABLED`
- `LR_PLATEAU_START`
- `LR_PLATEAU_END`

## Accept criteria

Strong success:

- beats corrected `034`
- and is competitive with or better than the best `034c` rung

Weak success:

- directionally positive, worth width tuning

Failure:

- flat or worse than both corrected `034` and `034cB`

## Notes

- This is intentionally a second schedule idea, not the first one.
- Run `034cB` first unless there is a strong reason to believe the transition
  itself, not the tail, is the dominant issue.
