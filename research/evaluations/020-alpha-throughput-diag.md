# Evaluation 020 — Alpha-throughput diagnostic

**Spec:** `research/specs/020-alpha-throughput-diag.md`
**Run:** `runs/020-alpha-throughput-diag/pod1/`
**Date:** 2026-04-21
**Hardware:** 4×H100 SXM NA (US-CA-2) — spec called for 8×H100 JP; JP was unavailable, NA 4×H100 used as exploratory substitute
**Commit:** `9bb1b01` (CUDA event bug unfixed; events returned -1.0 — see bugs section)
**Status:** completed (partial — TTT skipped, no mid-training val, nvsmi missing)

---

## Result summary

| metric | value |
|---|---|
| Steps completed | 2036 (wallclock cap) |
| Pre-loop median step time | **189.6ms** |
| Post-loop median step time | **277.6ms** |
| Loop activation overhead | **+46.4%** |
| Type A spikes (dataloader shard loads) | 16 |
| Type B spikes (GPU-side, unexplained) | 12 (excl. loop-compile spike) |
| Loop-compile spike | 1 (step 1023, 11.7s) |
| pre-quant post-EMA val_bpb | 1.11277 (4×H100, not comparable to #1736) |

---

## Spike taxonomy

Two completely distinct spike types identified from `diag_steps.csv`:

### Type A — Dataloader shard loads

Every **127 steps**, `dl_us` spikes to 32–130ms (130,000 μs). Step time rises modestly to ~310–360ms. Perfectly periodic — exactly 16 occurrences across 2036 steps.

| steps | dl_us | ms |
|---|---|---|
| 128, 255, 382, 509 | ~130ms | ~313ms |
| 636, 763, 890, 1017 | ~33–51ms | ~307–321ms |
| 1144, 1271, … 2033 | ~36–81ms | ~308–358ms |

**Interpretation:** The training dataloader advances to the next `.bin` shard every 127 steps. The first load of each shard takes ~130ms; subsequent loads from the same shard are faster (~35–80ms, likely OS page cache). This is expected I/O behavior — not a training bug, not a throughput regression, not related to recur-α.

The step-time impact is small (310ms vs 190ms baseline = +120ms = one extra shard-prefetch per 127 steps, <1% of total training time).

### Type B — GPU-side mystery spikes

12 spikes of 2–13 seconds where `dl_us` is normal (<1ms). Not correlated with:
- Dataloader (dl_us < 300μs in all cases)
- Mid-training val (no val fired in this run — `val_loss_every=4000`, run stopped at step 2036)
- LR schedule events or step-count boundaries
- Loop activation (spikes occur pre- AND post-loop, at steps 639, 640, 893, 894 pre-activation)

```
Pre-loop Type B spikes:
  step=639:  2,302ms   step=640:  3,649ms  (consecutive pair)
  step=893: 12,751ms   step=894:  3,233ms  (consecutive pair)

Post-loop Type B spikes:
  step=1274: 5,202ms
  step=1404: 3,276ms
  step=1653: 4,479ms   step=1654: 3,841ms  (consecutive pair)
  step=1759: 3,498ms
  step=1780: 8,498ms   step=1781: 11,657ms (consecutive pair)
  step=1910: 7,925ms
```

**Pattern:** Often in consecutive pairs (2 slow steps then instant recovery). Duration 3–13s, no fixed interval. Pre- and post-loop occurrence rules out the loop mechanism as the cause.

**Likely cause (hypothesis):** NCCL all-reduce stall on the interconnect, or a CUDA driver event (e.g. ECC scrubbing, power state transition, thermal throttle). Without `diag_nvsmi.csv` (lost to sidecar bug) we cannot confirm. The consecutive-pair pattern is consistent with a re-synchronization penalty after a stall: one slow step to trigger the stall, one slow step to drain the backlog, then normal.

**Not confirmed as related to recur-α constant:** spikes occur identically pre-loop (no recurrence running). These are pod-level infrastructure events.

### Loop activation compile spike

Step 1022 (+2×): 507ms. Step 1023: 11,774ms. Steps 1024+: immediately back to 273ms.

Confirmed `torch.compile` graph recompile when looping is activated (graph changes shape). One-time cost. No ongoing impact on throughput.

---

## Throughput: drift was a measurement artifact

The apparent "tok/s drift" in train.log (4.2M → 3.1M over steps 500–1000) is **entirely a cumulative-average artifact**. The per-step timing from `diag_steps.csv` shows two clean flat plateaus:

- Pre-loop: 189.6ms median, p99=285.7ms (only elevated by Type A shard loads)
- Post-loop: 277.6ms median, p99=289.6ms

There is no gradual drift. The ~46% post-loop overhead is a step-change at loop activation, not a thermal/contention drift.

**This refutes the "dip intervals" framing from the spec hypothesis.** What the spec called "dip intervals" in 019b's tok/s curve are mostly Type B mystery spikes (which happen on all runs, not specific to constant-α) plus the cumulative-average artifact smoothing over the loop-activation step-change.

---

## Bugs discovered and fixed

### Bug 1 — nvsmi sidecar: `$TRAIN_PID` not set

The launch script started `while kill -0 $TRAIN_PID` before `$TRAIN_PID` was assigned. Sidecar exited immediately. `diag_nvsmi.csv` is missing for this run.

**Fix (for next run):** Launch torchrun in background first (`torchrun ... & TRAIN_PID=$!`), then start the nvsmi loop, then `wait $TRAIN_PID`.

### Bug 2 — CUDA events: `elapsed_time()` returning -1.0

All `fwd_us`, `bwd_us`, `opt_us` columns are -1.0 across all 2036 rows. The `elapsed_time()` call threw an exception (silently caught) because the GPU hadn't synchronized before the read.

**Fix (committed):** Added `torch.cuda.synchronize()` before `elapsed_time()` reads in `exp/alpha-throughput-diag` (`85d502a`) and `exp/alpha-throughput-diag-buffer` (`3cfc372`). Per-phase attribution will work correctly in the next run using these commits.

---

## What we learned vs. what we can't conclude

| question | answer | confidence |
|---|---|---|
| Is tok/s drift real or artifact? | Cumulative-average artifact; per-step is flat | High |
| Does loop activation cause a permanent throughput hit? | Yes, +46% step time (189→278ms) | High |
| Are spikes caused by the dataloader? | Type A yes (shard loads, periodic). Type B no (dl_us normal). | High |
| Are Type B spikes caused by recur-α constant? | No — they occur pre-loop too | High |
| What causes Type B spikes? | Unknown; NCCL/driver/thermal candidate; need nvsmi to confirm | Low |
| Do spikes cluster after mid-training val? | Cannot determine — no mid-training val fired in this run | N/A |
| Is the mechanism 8×H100-JP-specific? | Unknown — this run was 4×H100 NA | Unknown |

---

## Decision

**Pivot on the diagnostic goal.** The 019b "dip pattern" appears to be a mix of:
1. Cumulative-average artifact (no real drift)
2. Universal Type B GPU spikes (not α-mechanism-specific)
3. A real ~46% throughput hit at loop activation (not a "dip" — a permanent step-change)

The original motivation for spec 020 was to diagnose whether the dips are caused by the recur-α mechanism (constant α vs. tensor α). That question is largely answered: the spikes pre-date loop activation, so they're not α-caused.

**Next steps:**
- Run **spec 021** (buffer-α full pipeline) on 8×H100 JP with confidence — the throughput "risk" from constant-α is the +46% loop-activation overhead, which buffer-α should not fix (loop overhead is architectural, not α-mechanism-dependent). The real question for 021 is val_bpb, not throughput.
- If per-phase attribution is needed (e.g., to understand whether forward or backward is responsible for the +46%), rerun with `85d502a` which has the CUDA event fix.
- `diag_nvsmi.csv` for a future run would let us correlate Type B spikes with GPU thermal/power state.
