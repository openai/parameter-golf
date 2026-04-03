# Crawler Leg 1 Hypothesis

Date: 2026-03-29

## Mission
Rebuild signal on the crawler path with DeltaNet fully quarantined.

## Hard Rules
1. `DELTA_NET_HEADS=0` for every run in this leg.
2. NGRAM evaluation stays off while rebuilding core architecture signal.
3. Track model-only metrics first (`final_int6_roundtrip_exact`, `final_int6_sliding_window_exact`).

## Why This Leg Exists
- Recent A/B indicates DeltaNet interaction is currently harmful/untrusted for crawler behavior.
- We need a clean crawler-only baseline and ablation stack before reintroducing any delta memory mechanism.
- Bandit is now SOTA and serves as external reference while crawler-only leg re-stabilizes.

## Crawler-Only Priority Queue
1. Loop count sweep (`CRAWLER_LOOPS`: 3/4/5)
2. Instruction bottleneck sweep (`INST_DIM`: 0/16/32/64)
3. Shared-block width sweep (`CRAWLER_MLP_MULT`: 3.0/4.0/5.0)
4. Flat/crawler depth split sweep (`NUM_FLAT_LAYERS`, `NUM_CRAWLER_LAYERS`)
5. Quant policy sweep for shared block (`CRAWLER_QUANT_INT8`: 0/1)

## Exit Criteria For Leg 1
- Stable crawler-only runbook with reproducible metrics.
- At least one crawler-only config that clearly improves baseline BPB or speed.
- DeltaNet remains disabled until a separate sandbox proves non-harmful interaction.
