# Spec 032 — Free-floating alpha/beta under `NUM_LOOPS=3` + recurrence curriculum

**Slug:** `freefloat-alpha-beta-recurrence-curriculum`
**Created:** 2026-04-23
**Status:** DRAFT
**Branch:** `exp/032-freefloat-alpha-beta`
**Commit:** `TBD`
**Links to:** `research/ideas/032-freefloat-alpha-beta-recurrence-curriculum.md`, `research/specs/024b-cross-layer-carry-blend.md`, `research/specs/024c-cross-layer-carry-per-pass.md`, `research/specs/029-full-stack-025b.md`, `research/specs/031-direct-carry-freefloat-neutral.md`

## Hypothesis

`029` likely regressed because it reused frozen `025b`-family carry constants that were calibrated in the older 2-loop regime. Once we moved to:

- `NUM_LOOPS=3`
- recurrence curriculum
- later depth upgrade behavior

the optimal carry coefficients likely changed.

So this spec proposes a **free-floating calibration run** with learnable `alpha` and `beta` under the real 3-loop regime. The goal is to read off the learned values, decide whether pass-specific structure matters, and freeze the result into a later spec.

This is a calibration spec, not a submission spec.

## Why this is separate from `031`

`031` deliberately isolates carry behavior in the cleaner `NUM_LOOPS=2` setting with the newer direct-carry combiner.

`032` asks a different question:

- under the full 3-loop recurrence-curriculum regime, what do the older `alpha/beta` style carry parameters want to become?

So `032` is the right separate line:

- `031` = cleaner 2-loop carry probe
- `032` = targeted re-calibration of `024b/024c` style alpha/beta in the actual 3-loop regime

## Baselines

| run | regime | pre-quant EMA | post-TTT | note |
|---|---|---|---|---|
| 024b | learnable shared `alpha/beta`, 2-loop family | historical | not relevant directly | source of the frozen 025b idea |
| 024c | learnable per-pass `alpha/beta`, 2-loop family | historical | not relevant directly | source of the per-pass 025c idea |
| 029 | frozen `025b` carry under 3-loop curriculum | ~1.07007 | bad screen result | likely mis-calibrated carry |
| 030 seed 314 | healthy 2-loop frozen reference | 1.06821629 | 1.06471941 | strong 2-loop comparison point |

Primary question:

- does learned `alpha/beta` under the true 3-loop regime produce a healthier trajectory than the frozen carry inherited by `029`?

## Parameterization

Take inspiration from the `024c` family, but implement this as a **new minimal modern carry block** in the current codepath rather than reviving old code directly.

Pinned design intent:

- per-pass learnable `beta`
- per-pass learnable `alpha`
- detached carry
- neutral init
- modern env mode: `DIRECT_CARRY_MODE=alpha_beta`

Conceptual form:

```text
x = beta[pass_off, i] * x_new + sum_j(alpha[pass_off, i, j] * carry[j])
```

with:

- `carry[j]` taken from detached prior looped activations
- for each destination pass, read from the **immediately previous pass only**
- pass-specific indexing retained so this run can tell us whether different passes want different carry structure
- implementation attached to the current carry-combine insertion point used by the modern codebase

For the current loop segment:

- `LOOP_START=3`
- `LOOP_END=5`
- `num_looped = 3`

and with:

- `NUM_LOOPS=3`

the modern `alpha_beta` mode uses:

- `alpha.shape = [3, 3, 3]`
- `beta.shape = [3, 3]`

Total: **36 learnable scalars**

### Initialization

- `alpha = 0`
- `beta = 1`

This preserves identity-like behavior at init.

## Regime

Pinned intent:

- `NUM_LOOPS=3`
- recurrence curriculum active
- no delayed recurrence-kick changes in this spec
- detached carry
- no TTT for the calibration run

This should stay as close as possible to the intended 3-loop training regime while changing only the carry coefficients from frozen to learnable.

## Implementation direction

Do **not** restore historical `024c` code verbatim.

Instead:

- use the current training code as the base
- keep the same modern carry insertion point used by the `031` direct-carry work
- add a new learnable `alpha/beta` mode beside the current direct-carry modes
- keep the patch as small and isolated as possible

Why:

- less code archaeology
- lower risk of reviving stale assumptions from the old stack
- easier to compare against the current 3-loop codepath
- easier to freeze into a follow-up spec afterward

## Runtime / schedule

Use a 4×H100 calibration run.

Extend the usual screen by about **20%** so the learned `alpha/beta` values have more post-onset time to settle.

### Critical timing rule

If runtime is extended, the onset ratio must be corrected so recurrence still activates at about the same absolute step.

Target:

- loop start around **step 2100**

Rule:

```text
new_loop_start_ratio = 2100 / new_total_steps
```

If expected total steps are about `6000`, then:

```text
ENABLE_LOOPING_AT = 0.35
```

Do **not** leave the old ratio unchanged.

The purpose of the extension is to increase the **post-loop learning tail**, not to delay loop activation.

### Depth-upgrade timing

This spec also keeps recurrence curriculum active, so there is a **second phase boundary** to pin:

- `LOOP_DEPTH_UPGRADE_AT`

For this calibration run, we should **not** keep the old `029` value of `0.67`.

Reason:

- in `029`, depth=4 came in too late
- most of its training happened before the final depth regime
- that is bad for a stabilization-oriented calibration run whose whole purpose is to observe learned `alpha/beta` under the final regime

Pinned draft choice for `032`:

- `ENABLE_LOOPING_AT = 0.35`
- `LOOP_DEPTH_UPGRADE_AT = 0.50`

Interpretation:

- preserve loop onset at about the usual absolute step
- move the depth-4 phase earlier than `029`
- give the final 3-loop depth-upgraded regime materially more training time

If later code-level validation shows a different expected total-step count than ~6000, the loop-on ratio should still be recomputed from absolute onset, but the current intended phase structure remains:

- loop onset near step `~2100`
- depth upgrade materially earlier than the old `0.67`

## Optimizer treatment

Use a separate scalar param group for `alpha/beta`.

Pinned default:

- `alpha/beta` LR = **1.5×** the normal scalar LR

Reason:

- these are tiny calibration parameters
- they start from neutral identity-like init
- we want them to move meaningfully within one extended run
- but we do not want an aggressive LR that makes the learned values noisy

## Logging requirements

In addition to normal metrics, emit:

- full `beta` tensors at each val/log interval
- full `alpha` tensors at each val/log interval
- drift vs previous snapshot
- simple summaries:
  - max abs alpha
  - row norms
  - per-pass norms

This run is only useful if we can inspect parameter trajectories, not just the final scalar loss.

## Accept criteria

This is a calibration spec, so acceptance is about usefulness of the learned carry values.

### Primary accept criteria

- run completes cleanly with no NaN / shape bug
- learned `alpha/beta` move away from init in structured ways
- trajectory is competitive enough that carry is clearly not broken
- late-stage drift is low enough that freezing looks plausible

### Strong success

- healthy float trajectory relative to the frozen 3-loop baseline
- pass-specific `alpha/beta` patterns are interpretable
- late-stage drift is small enough to freeze directly into a follow-up spec

### Weak / inconclusive

- run is healthy, but parameters are still moving materially near the end
- action: extend runtime before freezing

### Failure

- run regresses badly or destabilizes
- coefficients stay near init or become noisy without clear structure
- carry implementation is wrong for the 3-loop regime

## Decision after the run

- if passes converge to clearly different patterns: freeze a per-pass follow-up
- if passes are close: freeze a shared follow-up
- if drift remains high: rerun longer before freezing

## Hardware ladder

1. **4×H100 calibration run**
2. **Frozen follow-up spec** after inspecting learned values

Because this is a targeted carry-logic change in the real 3-loop regime, do not skip a smoke rung if the implementation diff becomes materially larger than expected.

## Open implementation questions

These need to be pinned before the spec becomes `READY`:

1. exact code branch / commit that adds the new modern `alpha/beta` mode under the current codebase
2. exact current recurrence-curriculum env var set to match the intended `029`-family regime
3. exact expected total-step count after the 20% wallclock extension
4. exact launch command under the current `/workspace/...` path convention
5. whether the modern implementation should expose one dedicated env mode for this run or a small family of `alpha/beta` modes

Current draft schedule intent once code exists:

- `NUM_LOOPS=3`
- `DIRECT_CARRY_MODE=alpha_beta`
- `DIRECT_CARRY_LR_SCALE=1.5`
- `ENABLE_LOOPING_AT=0.35`
- `LOOP_DEPTH_UPGRADE_AT=0.50`
- extended 4×H wallclock by about 20%

## Execution note

When this spec becomes ready, execution commands must use:

- `/workspace/...` paths, not `/runpod/...`
- `/tmp` for `TORCHINDUCTOR_CACHE_DIR`

## Proposed next step

Implement a small modern `alpha/beta` carry mode in the current codepath, then freeze the exact branch, commit, and launch command here.
