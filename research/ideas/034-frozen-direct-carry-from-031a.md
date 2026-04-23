# Idea 034 — Freeze the learned `031A` direct-carry pattern

## Thesis

`031A` was the cleanest carry calibration line:

- `NUM_LOOPS=2`
- detached carry
- no recurrence curriculum confound
- healthy `ratio0272` rerun with clear learned structure

The next question is not whether the richer direct-carry probe can learn. It did.

The next question is:

- if we freeze the learned `031A` carry pattern into a zero-overhead deployment
  object, does it hold up in a live-like 4×H proxy run?

## Why this line before freezing `032`

`032` produced stable late `alpha/beta`, but it learned across a phase change:

- looping onset
- later depth upgrade

That makes transfer trickier.

`031A` is simpler:

- cleaner regime
- cleaner interpretation
- no curriculum-phase mismatch

So it is the better first frozen follow-up.

## What to freeze

Use the healthy `031A-ratio0272` learned snapshot from `val_step_4000` as the
first freeze source.

Frozen values:

```text
self =
[[0.92578125, 1.5390625, 1.921875],
 [2.0,       2.0,       1.4296875]]

edges_pass1 =
[[ 0.349609375,   0.06005859375,  0.0615234375 ],
 [ 0.1337890625, -0.369140625,   -0.04150390625],
 [-0.0247802734375, 0.3828125,   -0.353515625  ]]

edges_pass2 =
[[ 0.55859375,   -0.03515625,    0.3046875,     -0.59765625,   -0.027099609375,  0.0162353515625 ],
 [ 0.036376953125,-0.28125,     -0.1123046875,   0.30078125,   -0.251953125,     0.0164794921875 ],
 [ 0.052001953125, 0.1103515625,-0.01336669921875,0.0576171875, 0.2353515625,    -0.07275390625   ]]
```

## Storage plan

Do not store these as trainable parameters.

Store them as frozen buffers in the model:

- `direct_carry_self_frozen` shape `[2, 3]`
- `direct_carry_edges_frozen_pass1` shape `[3, 3]`
- `direct_carry_edges_frozen_pass2` shape `[3, 6]`

This gives:

- zero optimizer overhead
- state_dict persistence
- clean logging/verification

## Regime

This is no longer a calibration run.

It should be a live-like proxy:

- `4×H100`
- `MAX_WALLCLOCK_SECONDS=1200`
- same overall pacing as the intended `8×H100 10min` line
- save the checkpoint
- run the whole pipeline, including quantized/TTT evaluation

## Timing

Use the original live-like onset timing:

- `ENABLE_LOOPING_AT=0.35`

Reason:

- this run is meant to proxy the live-style regime, not the longer calibration
  regime that required the `0.272` correction

## Core question

Can the richer `031A` learned structure survive freezing and still produce a
competitive end-to-end result?

If yes, this becomes a strong candidate frozen carry line.

If no, then either:

- the richer probe does not compress well into a frozen object
- or direct-carry needs to remain a diagnostic instrument rather than a
  deployment mechanism
