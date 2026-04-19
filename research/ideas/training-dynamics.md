# Training dynamics — what we know, what we're asking

**Status:** active, 2026-04-20. Consolidates scattered observations from spec 000 / 004b / 004c / 005 into a single reference. Updated after spec 006 results land.

## What we've observed

### Curve consistency (strong finding)
Across 3 independent 8×H100 full-length runs (spec 000, 004b QK=6.0, 004c QK=6.0 verify), train_loss at every 500-step milestone agrees within ~0.01 bpb, despite:
- Different `QK_GAIN_INIT` (5.25 vs 6.0)
- Different pods (bf16 cross-pod drift should accumulate)
- Identical seed (42)

**Implication:** the loss curve is highly reproducible across seed-matched runs, and small architectural knobs (QK gain range explored) do not substantially change gross curve shape at 500-step resolution. bf16 drift is real but apparently bounded.

### The flat zone (moderately strong finding, original motivator)
Reproducible plateau at step ~2000-2500 (possibly extending to 3000):
- Spec 000: `2.9431 → 2.9438` over step 2000→2500 (Δ +0.0007)
- Features in this window show consistently across all 3 full runs at 500-step resolution
- Nothing discrete scheduled here: warmdown fires at ~1048, recurrence at ~1378, Muon ramp ends at 1500. After step 1500, only linear LR decay is active.

We don't know at 100-step resolution whether the plateau is smooth or noisy-flat. Spec 006 answers this.

### Task A weight-delta analysis (partial finding, see runs/005-weight-delta/)
Three checkpoints (steps 1500, 2275, 3412) → two 775/1137-step windows.
- Per-step weight movement was **~2.5× higher** in window 1→2 vs 2→3 (global mean)
- Loop layers (3,4,5) moved only ~8% more per step than non-loop layers in BOTH windows — no strong window-localized loop differential
- Window 2→3 (post-flat, 2275→3412) was the descent phase; window 1→2 (1500→2275) straddled the flat zone's front half

**Interpretation:**
- Cause A (post-recurrence loop-layer adaptation): weak support. If loop-adaptation were driving the flat zone, loop layers should have moved disproportionately *during* the flat zone. They moved slightly more, but nothing dramatic.
- Cause B (LR-schedule artifact): partial support. LR ratio between windows is ~1.94×, weight-movement ratio ~2.5× — reasonably close, the extra factor ~1.3× plausibly from Muon momentum still ramping in window 1.
- Cause C (data-order): not tested; would require a different seed.

Neither window isolates the flat zone cleanly. Spec 006's 44 consecutive 100-step windows fix this.

## Open questions spec 006 will let us attack

### About the flat zone
1. **Flat-zone shape at 100-step resolution**: smooth plateau, noisy oscillation, or stair-step?
2. **Does val_loss flatten too?** If not → flat zone is a train-set-specific artifact, not a representational bottleneck.
3. **Does weight movement per step collapse in the flat zone?** If yes → optimizer is stuck. If no → weights are moving but not helpfully (moving tangentially to the loss gradient).
4. **LR-normalized movement across the flat zone**: is `||ΔW|| / LR` constant or dipping?

### About training dynamics more broadly
5. **Post-recurrence transient**: how does grad norm, per-layer movement, and loss respond in the ~200 steps after recurrence activates (step ~1593 in spec 006)?
6. **Loop-layer differential over time**: do loop layers (3,4,5) decouple from non-loop at any point? When?
7. **Train-val gap evolution**: at what step does val start trailing train? Does the gap open linearly or have structure?
8. **Grad-norm decay**: is it smooth or does it have regime changes at schedule events?
9. **Muon-ramp influence**: does weight-movement-per-step show a clear response to the Muon momentum ramp ending at step 1500?

### Meta
10. **LR-schedule sanity**: is the linear-warmdown curve well-matched to where the model wants to learn? If grad-norm × LR is visibly suboptimal anywhere (too small when model has signal, too large when it doesn't), there's a schedule intervention to consider.

## Interventions this data could motivate (NOT for today)

Explicitly parked — today is understand-first. Future specs might include:
- Modified warmdown curve (cosine, two-phase linear, step-flat-step)
- Delayed warmdown start
- Higher peak LR with faster decay
- Modified recurrence activation step (earlier/later)
- LR schedule specifically designed to hit higher effective movement during flat-zone-equivalent window
- Muon momentum ramp adjustments

None of these should be specced before seeing spec 006's results.

## Analyses to run when spec 006 artifacts land

See accompanying scripts in `research/scripts/` (TBD). Core set:

1. **Per-100-step weight-delta**, all 44 windows, per-layer. Output: `delta_matrix.csv` (rows=steps, cols=layers, values=per-step rel-movement).
2. **LR-normalized movement**: divide each window's per-step rel-movement by the LR-schedule at mid-window. Output: `lr_normalized.csv`.
3. **Loss curves at 5-step resolution** (from TRAIN_LOG_EVERY=5). Output: high-res loss plot + derived rolling-mean and first-difference curves.
4. **Val curve at 100-step resolution** overlaid on train. Output: train-val gap plot.
5. **Grad-norm per-layer per-5-steps** (from the grad-norm-logging diff). Output: grad-norm heatmap (time × layer).
6. **Loop-vs-non-loop differential over time**: time-series of (loop mean movement) / (non-loop mean movement). Is it flat ~1.08, or does it have structure?
7. **Cross-reference the flat zone location**: spec 006 has step-based schedule; flat zone is expected to shift from step 2000-2500 (spec 000 wallclock-based) to some step-scaled equivalent. Verify by looking for zero-progress regions in the finer loss curve.

## Post-spec-006 findings (2026-04-20, added after artifacts landed)

### Flat zone: closed case
Val_bpb at 100-step resolution throughout spec 006 descends smoothly from step 100 (1.725) to 4550 (1.088). **No plateau, no flat bed, nowhere.** The spec 000 "flat zone" at step 2000-2500 was aliasing between single-batch train_loss variance (±0.05-0.10 noise floor) and a true descent of only ~0.005 bpb over those 500 steps.

Explicitly checked the "delayed flat zone" hypothesis: if it were warmdown-phase-linked, spec 006's later warmdown_start (1275 vs 1048) should have shifted any such feature to ~step 2389-2978. Val gains per 100 steps in that window are −0.003 to −0.005 — no localized slowdown beyond the normal mid-training gentle deceleration.

**Takeaway:** the three-cause hypothesis (A post-recurrence adaptation, B LR-schedule artifact, C data-order) is obsolete — there was no phenomenon to explain.

### The real feature: recurrence activation transient
Step 1593 (spec 006) fires recurrence → step 1600 train_loss jumps +0.156, val_bpb +0.040. Full recovery by step 1700 (val ahead of pre-bump trajectory by −0.013). Visible in both train and val → real adaptation event, not noise.

**~100 steps of "recovery cost" to switch on recurrence.** If a softer activation (progressive recurrence, loop-layer warmup) could eliminate this, potential gain ~0.003-0.005 bpb. Worth spec'ing.

### Step-based vs wallclock-based schedule: matched-step comparison
At every matched step, spec 006 is **worse** than spec 000:
| step | spec 006 val_bpb | spec 000 / 004b ref | Δ (006 behind) |
|---|---|---|---|
| 2000 | 1.1816 | 1.1747 (004b) | +0.007 |
| 3000 | 1.1456 | ~1.132 (004b) | +0.014 |
| 3800 | 1.1174 | 1.0948 (004b) | +0.023 |
| 3849 | ~1.115 | 1.0929 (spec 000) | +0.022 |

Mechanism: step-based schedule shifts warmdown_start to step 1275 (vs wallclock-based 1048) → at matched steps, spec 006 has less warmdown progress → higher effective LR → less "polish." Spec 006's final 1.08639 wins only because it gets 701 more warmdown-descent steps.

**Intervention lead:** earlier warmdown (larger warmdown_frac, e.g., 0.80 or 0.85 instead of 0.72) likely yields better per-step val_bpb descent and, at fixed step count, better final bpb. Worth testing in a dedicated spec.

### Curve consistency, yet again
Pre-recurrence (steps 100-1500), spec 006 (QK=5.25) tracks 004b (QK=6.0) within ±0.012 train_loss. **QK_GAIN_INIT in the 5.25-6.0 range is irrelevant at this resolution** — confirmed.

## Links
- Plan: `~/.claude/plans/okay-yeah-so-i-ancient-porcupine.md`
- Spec: `research/specs/006-dense-ckpts.md`
- 5-run curve analysis: (Discord, 2026-04-19)
- Task A artifacts: `runs/005-weight-delta/`
- Prior full-length runs: `runs/000-sota-replication/`, `runs/004b-qk6-full/`, `runs/004c-qk6-verify/`
