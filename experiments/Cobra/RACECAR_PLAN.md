# COBRA Racecar Plan

## Objective
Find the best **base-only** 10-minute config with minimal wasted runs.

## Metric Contract
1. Rank by `final_int6_sliding_window_exact val_bpb` (lower is better).
2. Tie-breaker #1: `DIAGNOSTIC post_ema val_bpb`.
3. Tie-breaker #2: steps reached by 600s.
4. Hard fail: missing final base metric line.

## Run Policy
1. Use `MAX_WALLCLOCK_SECONDS=600` for full runs.
2. Disable n-gram eval for Cobra profiling (`NGRAM_EVAL_ORDER=0`) to cut turnaround and isolate base quality.
3. Keep architecture fixed (11L/512d, GQA 8/4, RoPE 24, XSA last 4).

## Laps

### Lap 0: Sanity (single seed, 120s)
- Purpose: reject unstable configs fast.
- Env override: `MAX_WALLCLOCK_SECONDS=120`.
- Pass if:
  - no runtime errors,
  - no NaN loss,
  - step time within +3% of reference.

### Lap 1: Full run (seed 1337, 600s)
- Run all surviving candidates once.
- Keep top 3 by base BPB.

### Lap 2: Stability check (seeds 42, 2025)
- Run top 3 only.
- Choose winner by mean base BPB and low variance.

## Selection Rule
Choose the config with the best mean base BPB across seeds while preserving throughput and no instability signs.

## Notes for the later compression stage
- Cobra intentionally defers compression tuning.
- Once the winning base config is chosen, run compression/artifact tuning as a separate pass.
