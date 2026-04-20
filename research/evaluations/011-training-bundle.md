# Evaluation — Spec 011 (training-bundle: tapered WD + GradPower)

**Run dir:** `runs/011-training-bundle/seed_42/`
**Commit:** `893cefd` on `exp/training-bundle`
**Baseline:** spec 008 `runs/008-1736-reproduction/seed_42/final.json`
**Eval date:** 2026-04-20

## Result

| metric | spec 008 | spec 011 | Δ |
|---|---|---|---|
| endpoint val_bpb (bare, screening mode) | 1.0697 | 1.0706 | **+0.0009** |
| diagnostic post-EMA val_bpb | 1.06922 | 1.06994 | +0.00072 |

Single seed. No TTT/GPTQ/sliding measured (screening mode per `feedback_screen_via_training_endpoint_val.md`).

## Noise/signal judgment

**Within noise.** #1736-family per-seed std is ~0.0002–0.0007 bpb (from #1736's own 3-seed report). Our single-seed Δ of +0.0009 is roughly 1.3–4.5× std. Directionally unfavorable but not decisively worse.

Matched-step train_loss curve tracks spec 008 within ±0.006 throughout, no regime change.

## The attribution bug

`WD_TAPER_START_FRAC=0.70` was defined as a fraction of `iterations` env var (=20000), not of actual stopping_early step (~4844 under the wallclock cap). Taper was scheduled for step 14000 → never reached. **This run effectively measures GradPower p=0.9 alone, not the bundle.**

Tapered WD remains untested on our stack.

## What we actually learned

1. **GradPower p=0.9 is null on #1736's 8×H100/600s regime.** #1682's 1×H100/1200s −0.00353 did not transfer. Either (a) the larger effective batch cleans up the gradient noise that p<1 was correcting for, or (b) #1736's Muon momentum=0.95 + momentum warmup is already doing the equivalent smoothing.
2. **Tapered WD is still an open question.** Our data says nothing about it — the lever never engaged.
3. **Edit-splits-a-block failure mode.** Commit 8d54854's rotary-orphan bug (fixed in 893cefd) was caused by `Edit(old_string=...)` capturing only part of a for-loop body. Future patches: use pure-prepend/append style OR verify old_string ends at a block boundary. Already applied this lesson on the BigramHash patch (spec 013, commit 66e57bf — 0 existing lines altered).

## Decision

**Kill GradPower**; **shelve tapered WD for this push.**

- GradPower: 1×H100 → 8×H100 transfer didn't hold as author warned it might. Not worth the retest cost on a different config.
- Tapered WD: expected Δ was already thin (−0.0005 to −0.002). Fixing the taper-schedule bug + running isolation costs ~$5–8 for a lever whose upside doesn't meaningfully change the trajectory. Come back to it only if specs 013/014 land and we're scraping the last 0.0005.

## Next steps

- Spec 013 (BigramHash) — currently pinned at `66e57bf`, ready to run. Moved to front of queue.
- Spec 014 (BPB-weighted CE) — idea file written; defer until 013 lands.
- QK_GAIN=2.5 (was spec 012 in earlier planning) — idea file written; defer until 013/014 lands.

## Cost

~$9.50 actual (includes a few debug cycles: rotary bug attempt, DATA_DIR path mistake, pyminify install, final clean run). Tracked against the $148 total remaining post-SpinQuant. Running total: **~$138 remaining before 013**.

## Memory-worthy followups from the run

Execution session captured:
- `project_caseops_data_path.md` — JP pod data path is `/workspace/data`, not `./data`.
- Existing `feedback_preflight_deps_and_gpu_clean.md` already covers `pyminify` preinstall.
- The `stopping_early: wallclock_cap` log line format (already in memory but easy to mis-pattern-match).
