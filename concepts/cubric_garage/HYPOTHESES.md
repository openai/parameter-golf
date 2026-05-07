# Cubric Garage — Test Hypotheses

All tests use copies of the SOTA. The original is NEVER modified.

## Test A: Baseline (no cubric)
- **File:** train_gpt_baseline.py (unmodified SOTA copy)
- **Script:** run_baseline.sh
- **Hypothesis:** Establishes the control number. Should reproduce 0.9625 BPB.

## Test B: Cubric Cadence 4 (aggressive)
- **File:** train_gpt_cadence4.py (SOTA + cubric C-step)
- **Script:** run_cadence4.sh
- **Env:** CUBRIC_CADENCE=4
- **Hypothesis:** Frequent C-steps catch fast-changing n-gram patterns. Decay stale counts, boost confirmed, prune collisions, reweight orders.
- **Expected:** +0.003-0.010 over baseline
- **Risk:** Too aggressive, may corrupt good counts.

## Test C: Cubric Cadence 10 (balanced)
- **File:** train_gpt_cadence10.py (SOTA + cubric C-step)
- **Script:** run_cadence10.sh
- **Env:** CUBRIC_CADENCE=10
- **Hypothesis:** More data per C-step = better decisions, less disruption.
- **Expected:** +0.002-0.008 over baseline
- **Risk:** Slower adaptation.

## Rules
1. NEVER modify the original SOTA file
2. Each test is a separate copy with its own run script
3. One variable per test
