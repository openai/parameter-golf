# COBRA Hypothesis (Base-Only)

## Core hypothesis
For this stack, we gain more from **stable, full-600s training throughput + low-noise optimizer tuning** than from adding eval-time n-gram complexity.

## What Cobra optimizes
1. Base quality at timer end (`final_int6_sliding_window_exact`), not n-gram score.
2. Step throughput consistency (`step_avg`, steps reached by 600s).
3. Low-variance knobs with prior evidence in this repo.

## Candidate classes
1. Complementary training strength (`COMPLEMENT_ALPHA`: 0.0 / 0.25 / 0.5)
2. SWA cadence (`SWA_EVERY`: 80 / 100 / 120)
3. Weight decay pair (`MUON_WD`, `ADAM_WD`: 0.035 / 0.040 / 0.045)
4. Late-QAT threshold (`LATE_QAT_THRESHOLD`: 0.45 / 0.50 / 0.55)

## Explicit non-goals for Cobra
1. No architecture jumps (depth/width/head geometry unchanged)
2. No prime/odd dimension exploration in the core model
3. No varlen-attention behavior experiments
4. No TTT, no post-hoc oracle mixer logic

## Success criteria
- Reproduce <= `1.1195` consistently on seed 1337 with Cobra harness
- Beat <= `1.1190` on at least one seed without regressing runtime stability
- Preserve artifact budget margin for later compression pass
