# Evaluation 031 — Direct-carry free-float calibration (`NUM_LOOPS=2`, neutral init)

**Spec:** `research/specs/031-direct-carry-freefloat-neutral.md`  
**Primary rerun artifact:** `runs/031-direct-carry-freefloat-neutral/031B-ratio0272-freshpod-rerun1/`  
**Date:** 2026-04-23  
**Code line:** pinned spec commit `1cac69b`, plus a local rerun-only fix to synchronize wallclock-based loop activation across ranks before entering the looped graph  
**Status:** partial evaluation; `031B` rerun completed through the carry-calibration phase and was manually stopped early at `train_step 5400/20000`

---

## Result summary

This note records the usable result from the repaired `031B` rerun.

The first `031B-ratio0272-freshpod` attempt was invalidated by a distributed crash immediately after loop activation:

- rank `0` entered `_ALLGATHER_BASE`
- other ranks were still in a 39-element `ALLREDUCE`
- 39 exactly matches the `031B` carry parameter block

The rerun patched that failure mode by synchronizing the wallclock-based loop toggle across ranks before `looping_active` changed. With that fix in place, the rerun:

- passed the old crash window cleanly
- activated looping at `step 2144`, `frac 0.272`
- logged full `edges`, `self`, and `carry_gate` snapshots throughout
- entered a low-drift regime by the mid-5000s
- was manually stopped at `step 5400` to save compute once the carry tensors were operationally converged

Because the run was stopped before EMA/quantized/TTT endpoints, this evaluation is about **carry calibration quality and stability**, not final val_bpb.

---

## Evidence

### Loop-onset transition was real and healthy

The rerun crossed the exact region that had crashed before:

- `layer_loop:enabled step:2144 frac:0.272`

Immediately after onset, the carry block moved hard off init:

- `train_step_2200`
  - `self_max_drift=0.351562`
  - `gate_max_drift=0.572266`
  - `self=[[0.73828125, 1.0, 1.3515625], [0.91796875, 0.953125, 0.98828125]]`
  - `carry_gate=[[0.427734375, 0.4296875, 0.828125], [0.62109375, 0.59765625, 0.71484375]]`

That is strong evidence that the repaired `031B` path is live and that the gate tensor is not a dead parameter.

### Late carry state entered a stable regime

By the time the run reached `5000+` steps, the carry parameters were still changing, but only by tiny increments:

| step | edge_max_drift | self_max_drift | gate_max_drift |
|---|---:|---:|---:|
| 5000 | `[0.0078125, 0.013671875]` | `0.000000` | `0.003906` |
| 5100 | `[0.009765625, 0.01171875]` | `0.000000` | `0.007812` |
| 5200 | `[0.005859375, 0.009765625]` | `0.000000` | `0.007812` |
| 5300 | `[0.01250457763671875, 0.01171875]` | `0.000000` | `0.001953` |
| 5400 | `[0.01030731201171875, 0.01171875]` | `0.000000` | `0.003906` |

The notable part is `self_max_drift=0.0` for the entire last stretch. `carry_gate` also flattened to the `~0.002–0.008` range.

### Final logged carry tensors before manual stop

At `train_step_5400`:

- `self`

```text
[[1.0234375, 1.4921875, 2.0],
 [2.0, 2.0, 1.3125]]
```

- `carry_gate`

```text
[[0.44921875, 0.53515625, 0.6953125],
 [1.046875, 0.6484375, 0.3984375]]
```

- `edges`

```text
[[[0.416015625, -0.04931640625, 0.21875],
  [0.36328125, -0.27734375, -0.0093994140625],
  [-0.1025390625, 0.345703125, -0.4296875]],

 [[0.28125, -0.09130859375, 0.24609375, -0.45703125, -0.06689453125, 0.03271484375],
  [0.10205078125, -0.1337890625, -0.07080078125, 0.42578125, -0.333984375, 0.01348876953125],
  [-0.06494140625, 0.1904296875, -0.060791015625, 0.06396484375, 0.478515625, -0.123046875]]]
```

These are interpretable, nontrivial carry values, and they are much cleaner to freeze than the raw post-onset spike.

### Training loss did not show a compelling continued trend

Recent train losses:

- `5000`: `2.7457`
- `5100`: `2.8394`
- `5200`: `2.8464`
- `5300`: `2.8137`
- `5400`: `2.8163`

That sequence is noisy rather than obviously still improving. Combined with the low-drift carry state, this made manual early stop reasonable.

---

## Noise / signal judgment

**Signal is real for the mechanism, incomplete for endpoint quality.**

What is decisively established:

- `031B`'s `carry_gate` path is now live
- the old crash was a distributed loop-toggle bug, not evidence against the mechanism
- the learned gate values move far off the neutral init and then settle

What is still not established:

- whether `031B` beats `031A` on final pre-quant EMA
- whether `031B` beats the old frozen 025b-style carry on final quality
- whether the stabilized `031B` values are better freeze targets than `031A`

So this is a **successful calibration measurement**, but not yet a completed model-quality comparison.

---

## Decision

**ITERATE / PROMOTE AS A FREEZE CANDIDATE, not as a final quality result.**

Recommended interpretation:

- keep the rerun fix; it looks like the correct repair for the `031B` crash
- treat the `train_step_5000–5400` carry tensors as the first usable `031B` freeze candidate
- do not claim a `031A vs 031B` winner on val_bpb from this run, because the run was intentionally stopped before endpoint evaluation

---

## Next steps

1. Freeze a follow-up spec from the stabilized `031B` tensors rather than from the unstable post-onset region.
2. If needed, also record the corresponding `031A` late snapshot as the clean no-gate comparison target.
3. If a final quality comparison is still wanted, rerun a shorter endpoint-complete version now that the distributed crash is fixed.

The cheapest next discriminating experiment is likely:

- freeze the stabilized `031B` carry tensors
- run a screen-mode endpoint comparison against `025b` / `031A`
- only pay for full EMA/quantized/TTT if the frozen `031B` screen is competitive

---

## Artifacts

- Partial-run note: `runs/031-direct-carry-freefloat-neutral/031B-ratio0272-freshpod-rerun1/notes.md`
- Local summary: `runs/031-direct-carry-freefloat-neutral/031B-ratio0272-freshpod-rerun1/partial.json`
- Full log: `runs/031-direct-carry-freefloat-neutral/031B-ratio0272-freshpod-rerun1/train.log`
- Snapshot stream: `runs/031-direct-carry-freefloat-neutral/031B-ratio0272-freshpod-rerun1/carry_snapshots.jsonl`
