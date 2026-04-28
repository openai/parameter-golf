---
name: launch-and-await
description: Invoke when launching an experiment — the standard pattern for kicking off `run_experiment.sh` in the background, gating on the first 10 training steps, and waiting on completion without busy-polling. Each MPS smoke is ~5 minutes; the launch/await rhythm matters because it lets you do other work while runs progress and catches failure early.
---

# Launch and Await

Each experiment is ~5 minutes on MPS. The pattern: launch in background, gate on the first 10 steps to catch early failure, then wait for completion notification. Don't sit and watch.

## 1. Launch the run

```python
RUN = Bash(
  run_in_background=True,
  timeout=900000,                  # ≥ MAX_WALLCLOCK_SECONDS + margin
  command="cd experiments/NNNN_<slug> && ../../run_experiment.sh",
)
```

`run_experiment.sh` will refuse to launch if `plan.md` has unfilled `<!-- ... -->` template sections — fill them first if it errors.

## 2. Gate on the first 10 steps (~15 s)

```python
Bash(
  run_in_background=True,
  timeout=120000,
  command="./await_steps.sh experiments/NNNN_<slug>",
)
```

`await_steps.sh <exp_dir> [N=10]` blocks until N step lines exist in `run.log`, then prints them. Exits early on crash, log-mtime stall, or hard timeout. The captured stdout *is* the trajectory.

**Trajectory sanity checks** (review the printed steps):
- Step 1 ≈ ln(vocab) ≈ 6.93. Anything wildly different → flag.
- Monotonic descent from step 2 onward.
- Step 2 within ~2× of step 1. Spikes mean LR overshoot from cold-start init.
- Watch for any NaN/Inf.

If trajectory looks bad: `TaskStop(RUN)`, fix the config, relaunch. Don't let a doomed run burn the full 4 minutes.

## 3. Do other work while it runs

Once the first-10 gate passes, the run is healthy. You may switch to other work — read journal entries, fill the next experiment's plan.md, sketch in scratch/, query past work via `search_journal`. The launch task notifies on completion. You may also just wake for the Monitor event if this run would be a blocker for other work and you have to wait for it to finish.

## 4. Mid-run check-ins (optional)

Same script, larger N:

```python
Bash(run_in_background=True, timeout=300000,
     command="./await_steps.sh experiments/NNNN_<slug> 100")
```

Returns when 100 step lines exist (or earlier on crash/stall). Stack as many as you like — the run keeps going underneath.

### Trajectory-gate-then-completion-wait (recommended for unfamiliar architectures)

For runs where a new code path could fail in unanticipated ways (architectural change, new SSM family, custom kernel), split the wait:

```python
# 1. Short trajectory gate — confirm the run is healthy.
Bash(run_in_background=True, timeout=300000,
     command="MAX_WAIT_SECONDS=300 ./await_steps.sh experiments/NNNN_<slug> 100")
# review the printed steps; if NaN/stall/wrong-shape, TaskStop(RUN) and fix.

# 2. Long completion wait on a known-good run.
Bash(run_in_background=True, timeout=2000000,
     command="./await_steps.sh experiments/NNNN_<slug> 99999")
```

Why split: `await_steps.sh`'s default ceiling is 1800s (30 min) — long enough for a full triple-parallel run, but the python-gone and log-stale checks fire well before that on real failures. The split pattern is for the residual "running but in a bad state" failure mode that pure-stall checks can't catch (e.g. NaN that gets clamped, memory leak that doesn't crash, wrong-shape forward path that produces uninformative loss). The trajectory-gate is your eyes on that case.

## 5. Streaming for late-NaN watching (optional)

Only when watching the back half of a long run for late instability:

```python
MON = Monitor(
  description="exp NNNN progress: train/val/errors",
  timeout_ms=900000,                              # ≥ RUN timeout
  persistent=False,
  command=(
    "tail -f experiments/NNNN_<slug>/run.log | grep -E --line-buffered "
    "'^step:[0-9]+/[0-9]+ train_loss|^step:[0-9]+/[0-9]+ val_loss|"
    "final_int8_zlib_roundtrip|Total submission size int8|"
    "Traceback|Error|FAILED|Killed|OOM|assert|[Nn]a[Nn]|[Ii]nf'"
  ),
)
```

When `RUN` notifies completion: **immediately `TaskStop(MON)`**. `tail -f` doesn't self-terminate; without the explicit stop the monitor burns until `timeout_ms`.

## 6. After completion

`run_experiment.sh` writes `result.json`, appends to `results.tsv` with `TODO` placeholders in `status` and `description`. Review the printed summary (val_bpb_post_quant, quant_tax, artifact_mb, the first 10 step lines), then:
- Fill `status` (`keep`/`discard`/`parked`/`crash`/`sentinel`) and `description` (6–10 words + transfer tag for keeps).
- If it beat the current best → invoke `promote`.
- If it crashed → diagnose, fix, retry once. Don't retry the same broken config more than twice.
- Journal selectively — surprising results, novel hypotheses, or lessons future sessions need. Routine sweeps don't earn entries.

## Crash handling

If `val_bpb` empty, `crashed=true`, traceback in `run.log`:
1. `tail -n 50 experiments/NNNN_<slug>/run.log`.
2. Typo / missing import / shape mismatch from your edit → fix and retry. One experiment.
3. Idea fundamentally broken (architectural change you don't fully understand) → set `status=crash`, journal a one-line entry, move on.
4. Don't retry the same broken config more than twice.
