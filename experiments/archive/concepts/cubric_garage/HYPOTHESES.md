# Cubric Garage — Test Hypotheses

All tests use copies of the SOTA. The original is NEVER modified.

## Test A: Baseline (no cubric)
- **File:** train_gpt_baseline.py (unmodified SOTA copy)
- **Script:** run_baseline.sh
- **Hypothesis:** Establishes the control number. Should reproduce 0.9625 BPB.
- **Expected:** 0.9625 (seed 1337)

## Test B: Cubric Cadence 4 (aggressive)
- **File:** train_gpt_cadence4.py (SOTA + cubric C-step)
- **Script:** run_cadence4.sh
- **Env:** CUBRIC_CADENCE=4
- **Hypothesis:** Frequent C-steps (every 4 eval batches) catch fast-changing patterns in the n-gram tables. Decay stale counts, boost confirmed patterns, prune hash collisions, reweight orders by accuracy. The hash tables become adaptive rather than static.
- **Expected:** +0.003-0.010 over baseline
- **Risk:** Aggressive optimization may corrupt good counts. 4 batches may not be enough signal per C-step.

## Test C: Cubric Cadence 10 (balanced)
- **File:** train_gpt_cadence10.py (SOTA + cubric C-step)
- **Script:** run_cadence10.sh
- **Env:** CUBRIC_CADENCE=10
- **Hypothesis:** More data per C-step = better decisions. Less disruption to tables. Sweet spot between adaptation speed and stability.
- **Expected:** +0.002-0.008 over baseline
- **Risk:** Slower adaptation may miss short patterns.

## Rules
1. NEVER modify the original SOTA file
2. Each test is a separate copy with its own run script
3. One variable per test (CUBRIC_CADENCE)
4. All training is identical — cubric only affects n-gram eval
5. Compare final_int6_sliding_window_ngram BPB across all three
