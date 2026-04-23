# Evaluation — `031` / `032` / `033` / `033b` carry and TTT follow-ups

## Scope

This note summarizes the recent carry-calibration and TTT alpha/beta adaptation
experiments:

- `031A` direct-carry free-float (`NUM_LOOPS=2`)
- `032` free-floating `alpha/beta` under `NUM_LOOPS=3` + recurrence curriculum
- `033` TTT adaptation of frozen alpha/beta
- `033b` same as `033`, but with aggressive TTT alpha/beta LR

## Main findings

### 1. `031A` learned a real direct-carry structure in the clean 2-loop regime

`031A` moved decisively off neutral init after looping activated. By the late
tail, the parameters were still moving a bit, but the learned structure was
already clear.

Useful late snapshot from `val_step_4000`:

- `self`:

```text
[[0.92578125, 1.5390625, 1.921875],
 [2.0, 2.0, 1.4296875]]
```

- `edges`:

```text
[[[0.349609375, 0.06005859375, 0.0615234375],
  [0.1337890625, -0.369140625, -0.04150390625],
  [-0.0247802734375, 0.3828125, -0.353515625]],

 [[0.55859375, -0.03515625, 0.3046875, -0.59765625, -0.027099609375, 0.0162353515625],
  [0.036376953125, -0.28125, -0.1123046875, 0.30078125, -0.251953125, 0.0164794921875],
  [0.052001953125, 0.1103515625, -0.01336669921875, 0.0576171875, 0.2353515625, -0.07275390625]]]
```

Interpretation:

- the cleaner 2-loop direct-carry line is alive
- there is meaningful structure to freeze from
- this is the easier line to reason about than the 3-loop curriculum run

`031B` does not currently have a completed artifact in the local repo.

### 2. `032` did converge late, but the calibration object is intrinsically harder

`032` produced clear late-stage stable `alpha/beta` snapshots under the 3-loop
curriculum regime.

Final stable snapshot (`val_step_5454`):

- `beta`:

```text
[[1.0, 1.2109375, 1.265625],
 [1.796875, 2.0, 1.5703125],
 [2.0, 1.96875, 1.296875]]
```

- `alpha`:

```text
[[[0.220703125, -0.011962890625, 0.1357421875],
  [0.09814453125, -0.216796875, -0.06396484375],
  [0.091796875, 0.1875, -0.28515625]],

 [[-0.11962890625, -0.04345703125, 0.1748046875],
  [0.251953125, -0.65625, -0.1484375],
  [0.05712890625, 0.20703125, -0.150390625]],

 [[-0.0517578125, -0.05859375, 0.328125],
  [0.1513671875, -0.189453125, -0.10595703125],
  [0.041748046875, 0.11865234375, -0.01287841796875]]]
```

Late drift was very small:

- `alpha_max_drift = 0.001220703125`
- `beta_max_drift = 0.0`

So the run did reach a stable late regime.

But the caution is structural:

- `032` is fitting parameters across a phase transition
- first looping turns on
- later depth upgrades

That means the final stable values are probably best interpreted as
**late-phase/final-regime values**, not as one universal tensor that should be
expected to transfer cleanly across the whole curriculum path.

Interpretation:

- `032` succeeded as a measurement run
- `032` is difficult as a freezing target if one tensor must serve both pre-upgrade
  and post-upgrade phases

### 3. `033` showed that TTT alpha/beta adaptation is possible, but low impact

`033` let TTT update frozen `alpha/beta` on top of the same `026 seed_42`
checkpoint used by `028B`.

Result:

- `028B`: `1.0664948109`
- `033`: `1.0664878103`

Delta:

- about `-7e-06` bpb

That is directionally better, but effectively negligible.

Observed behavior:

- `recur_alpha` moved a little
- `recur_beta` did not move materially

Interpretation:

- TTT alpha/beta adaptation is not obviously harmful at very low LR
- but the effect size is too small to justify promotion

### 4. `033b` answered the “was LR just too low?” question cleanly

`033b` reran the same line with aggressive TTT alpha/beta LR.

Observed drift:

- `recur_alpha_max_drift = 0.240723`
- `recur_beta_max_drift = 0.062500`

So unlike `033`, this time both parameter sets moved materially.

But the final result got worse:

- `033`: `1.0664878103`
- `033b`: `1.06666734`

Interpretation:

- the flat `033` result was not because alpha/beta were impossible to move
- they are movable
- moving them harder hurts

This gives a coherent story:

- tiny TTT alpha/beta adaptation is negligible
- aggressive TTT alpha/beta adaptation is harmful

## Decision

### Promote

- `031A` as the cleaner carry-freeze candidate line

### Keep as a measurement result

- `032` as evidence of what the late 3-loop curriculum regime wants

### Deprioritize

- `033` / `033b` as a mainline TTT lever

The remaining plausible TTT-side follow-ups would be narrower:

- `alpha`-only adaptation with `beta` frozen
- or one medium-LR interpolation between `033` and `033b`

But this line should be treated as low priority now.

## Recommended next framing

- use `031A` for the cleaner freeze path
- treat `032` late values as a final-regime calibration artifact, not a universal
  across-curriculum answer
- do not assume “more movement” in TTT carry parameters is beneficial
