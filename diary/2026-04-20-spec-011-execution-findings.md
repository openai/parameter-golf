# Spec 011 execution — findings

**Date:** 2026-04-20
**Mode:** execution session
**Spec:** `011-training-bundle` (tapered Muon WD + GradPower 0.9; per-layer QK deferred to spec 012)
**Result:** training healthy, **bundle effectively unmeasured** because the taper never fired.

## TL;DR

- Training ran cleanly to wallclock cap at step **4844 / 596s** on commit `893cefd`.
- Endpoint val_bpb (bare): **1.0706** vs spec 008's **1.0697** → **Δ +0.0009** (within ±0.001–0.002 noise band).
- **WD taper never engaged** (start step computed as 14000, never reached). What we actually measured is GradPower-0.9 alone vs spec 008's GradPower-1.0.
- Cost: ~$9.50 (more than ideal, broken down below).
- Pod stopped, container preserved.

## Result table

| metric | spec 008 (#1736 repro) | spec 011 (intended bundle) | Δ |
|---|---|---|---|
| stopping_early step | 4828 | 4844 | +16 |
| stopping_early wallclock (ms) | 596180 | 596160 | ≈ equal |
| step-endpoint val_bpb (bare) | **1.0697** | **1.0706** | **+0.0009** |
| diagnostic post-EMA val_bpb | 1.06922 | 1.06994 | +0.00072 |
| matched train_loss curve | — | within ±0.006 throughout | no regime change |

Train_loss curves overlap to within ±0.006 across the run; no divergence anywhere.

## The taper-never-fired bug

Spec 011's patch computes the WD taper start as:
```python
start_step = int(h.wd_taper_start_frac * total_steps)
# 0.70 * 20000 = 14000
```
where `total_steps = h.iterations` (env var `ITERATIONS=20000`).

But our runs cap on wallclock (`max_wallclock_seconds = 600`), not on iterations. Spec 008 stopped at **4828**; we stopped at **4844**. We never get within 9× of step 14000, so the taper window `[14000 → 20000]` is unreachable in any 600s-cap run with these defaults.

**Implication for evaluation:** the +0.0009 endpoint Δ is a measurement of GradPower-0.9 alone, not of the intended bundle. The "tapered WD" lever is untested by this run.

**Possible fixes (research's call):**
- (A) Anchor taper-start on wallclock — `0.70 × 600s = 420s` of training_time.
- (B) Anchor on estimated stopping_early step — use spec 008's 4828 as a reference; taper start at `~3380`.
- (C) Reuse `warmdown_frac` schedule — its tail already aligns with end-of-training.

Option C feels cleanest because `warmdown_frac` already encodes "the training schedule's late phase" — but that may couple two distinct ideas. Option A is the smallest change.

## Things that went wrong (chronological, with cost)

| # | issue | type | cost | resolution |
|---|---|---|---|---|
| 1 | Created NA pod by reflex (memory said NA-1, but JP volume actually has the data) | env | ~$0.10 | Deleted, re-created in JP |
| 2 | JP secure 2×H100 + 1×H100 + community 2×H100 + 1×H100 all out of capacity for smoke | infra | $0 (poll loop) | Polled every 60s; 5th attempt landed |
| 3 | Spec's launch block had `DATA_DIR=./data`; spec 008's working config used `/workspace/data` | spec/env | ~$1.20 | Fixed in launcher; saved to memory `project_caseops_data_path.md` |
| 4 | Commit `8d54854` had a shape bug — Edit insertion orphaned `block.attn.rotary = Rotary(...)` outside the rope for-loop, causing `RuntimeError: broadcast [1, 98304, 1, 32] vs [1, 98304, 8, 8]` at warmup | logic | ~$1.50 | Halted per execution rules, handed back to research; fix landed at `893cefd` |
| 5 | `pyminify` CLI not installed on pod template; `serialize()` calls `subprocess.run(['pyminify', ...])` and crashes after training | env | ~$1.20 (smoke crash) | Installed via `pip install python-minifier`; already documented in `feedback_preflight_deps_and_gpu_clean.md` (I missed it) |
| 6 | My `stopping_early` watcher used pattern `"stopping_early at step"` but the real log line is `"stopping_early: wallclock_cap"` — watcher never fired | execution sloppiness | ~$0.30 (one extra GPTQ stage before manual kill) | Memory `feedback_screen_via_training_endpoint_val.md` already has the correct pattern; my error was not following it |

Total: ~$9.50. Of that, ~$3 was avoidable if I'd consulted memory more carefully (items 5 + 6) and ~$1.50 was a logic bug not on me.

## What worked

- 2×H100 smoke at `893cefd` ran cleanly through 500 steps then crashed only on `pyminify` (post-training) — confirmed the rotary-orphan fix holds at small batch.
- 8×H100 launch at `893cefd` produced the `training_bundle:` config line at boot with all knobs printed correctly.
- Patch is byte-compatible with spec 008 when its env vars are unset (verified by 008's reproduction working off the same commit if WD/GP defaults were applied — though we didn't actually re-run 008 here).
- tok/s 8.1M → 7.6M throughout training, well above the 6.5M floor — no slow-hardware concern.

## What memory captured / didn't update

**Saved this session:**
- `project_caseops_data_path.md` — the `DATA_DIR=/workspace/data` correction.

**Already in memory and merely re-confirmed:**
- `feedback_preflight_deps_and_gpu_clean.md` — `python-minifier` is in the recommended install set; I missed it in preflight.
- `feedback_screen_via_training_endpoint_val.md` — correct watcher pattern is `"stopping_early: wallclock_cap"`; I used the wrong one.

**Not saved:** debugging the rotary bug or `pyminify` CLI is in the codebase / fixable; nothing memory-worthy beyond what's already there.

## Handback to research

Open questions for research:
1. Is +0.0009 endpoint val_bpb (GradPower-only run) signal or noise? Spec 008 baseline std unknown at single seed, but #1736 cross-pod noise is ~±0.001.
2. Re-spec the bundle so the taper actually fires — pick from anchor options A/B/C above. Or merge the taper logic into `warmdown_frac` directly.
3. Does GradPower-alone warrant standalone screening (no taper, just `MUON_GRAD_POWER=0.9`)? This run effectively gave us that data point already.
4. Spec 012 (per-layer QK) is queued; it sits on this same patch family. Should it wait until taper-fix lands?

## Cost summary, session-level

- **Spec 011 spend:** ~$9.50 (one full training run + multiple debug iterations + smoke).
- **Spec 008 spend:** ~$14 (per its notes; multiple post-training pipeline crashes).
- **Spec 009 spend:** ~$14.70 (per its summary).
- Running total for the post-#1736 baseline migration era: **~$38**, well within the ~$200 hard budget but not yet showing a net win on the leaderboard.
