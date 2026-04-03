# BW15_AllCrawler_2k — Hypothesis

Primary objective: maximize crawler interaction gains by running high-amplitude architecture shifts first, then sweep smaller interaction/quant variants in one uninterrupted sequence.

## Hypotheses

- `H1`: The largest remaining gain is in architecture phase-shift behavior, not quant micro-adjustment.
- `H2`: `NUM_FLAT_LAYERS=6` is currently the strongest big-swing lever and should be revalidated first in every consolidated pass.
- `H3`: Tap-off quant policies remain sequential/post-window candidates and should be tested after one tap-off control window.
- `H4`: Tap-shared and tap-off families should keep separate control baselines to avoid cross-family delta confusion.

## Execution Ordering Rule

1. Run `BIG_SWING` arms first.
2. Run `SMALL` interaction and quant arms after big swings.
3. Deduplicate repeated controls when architecture is identical (BW13 tap-off control reuses BW14 tap-off control).

## Promotion Thresholds

- Big-swing promote: `delta_vs_control <= -0.0060`
- Small-sweep promote: `delta_vs_control <= -0.0008`

