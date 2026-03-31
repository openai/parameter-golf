# Bandit_Wagon_V — Results

## Architecture

BW4 + COMPILE_FULLGRAPH=1. Single validated change.

- `COMPILE_FULLGRAPH=1`
- `CRAWLER_MLP_CHOKE_DIM=0` (no choke)
- `CRAWLER_LOOP_ROPE_SCALES=9,1,1`
- All other flags identical to BW4

## Tier 1 Gate — COMPILE_FULLGRAPH=1 (2000 steps, seed=444, 8×H100)

| Metric | BW4 baseline | BW5 fullgraph | Delta |
|--------|-------------|---------------|-------|
| step_avg | 74.80ms | **74.52ms** | **-0.28ms** |
| graph breaks | — | **0** | clean |
| roundtrip eval | 13,061ms | **4,707ms** | **2.77× faster** |
| sliding window eval | 73,489ms | **64,274ms** | **12.5% faster** |
| quant_gap (2k steps) | -0.0119 | **-0.0188** | more negative |

Step_avg still trending down at step 2000 (74.82 → 74.70 → 74.58 → 74.51ms).
May settle further below 74.52ms in full production run.

## Production Results

| Seed | Steps | raw_bpb | int6_sw_bpb | quant_gap | bytes | vs BW4 |
|------|-------|---------|-------------|-----------|-------|--------|
| 444 | 8035 | 1.1987 | **1.18672385** | -0.0120 | 9,024,399 (~8.61MB) | **-0.00058** |
| 300 | TBD | TBD | TBD | TBD | TBD | TBD |

Step_avg production: 74.68ms (trending down, clipped by wallclock).
SWA kicked in at step 7650.
Sliding window eval: 72,592ms (full val set, 8×H100).

## Reference

| Config | int6_sw_bpb | bytes |
|--------|-------------|-------|
| Leg 3 SOTA | 1.18746 | 8.84MB |
| BW4 seed=444 | 1.18731 | 8.97MB |
| **BW5 seed=444** | **1.18672** | **8.61MB** |
| BW5 seed=300 | TBD | TBD |
