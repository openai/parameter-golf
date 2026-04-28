# Bandit_Wagon_III — Results

## Architecture

pyramid-512 choke + 9,1,1 battery on Crawler Leg 3 base.

- `CRAWLER_MLP_CHOKE_SHAPE=pyramid`
- `CRAWLER_MLP_CHOKE_DIM=512`
- `CRAWLER_MLP_CHOKE_GROUPS=8`
- `CRAWLER_LOOP_ROPE_SCALES=9,1,1`
- `NUM_FLAT_LAYERS=4` / `NUM_CRAWLER_LAYERS=1` / `CRAWLER_LOOPS=3`
- `CRAWLER_MLP_MULT=6.0` / `INST_DIM=32`
- `MLP_LEAKY_SLOPE=0.5` / `CRAWLER_MLP_LEAKY_SLOPE=0.5`
- `XSA_LAST_N=11`
- `SKIP_GPTQ=1` / `SKIP_EMA=1`
- `MAX_WALLCLOCK_SECONDS=600` / `WARMDOWN_ITERS=2000`

## Run 1 — seed=444, 8×H100, 2026-03-31

| Metric | Value |
|--------|-------|
| Steps | 7548 (wallclock cap at 600s) |
| SWA start | step 7150 |
| step_avg | 79.50ms |
| raw_bpb | 1.1980 |
| int6_sw_bpb | **1.20684096** |
| quant_gap | +0.0088 |
| bytes | 10,067,990 (~10.07MB) |
| val set | 62,021,632 tokens |
| log | `results/BW3_s444_20260331_061333.log` |

### Notes

- val_bpb at step 0: 4.1048 (62M token val set — different from original BWCD reference pod at 58M)
- quant_gap +0.0088: larger than the +0.0001 seen in 500-step ablations; SWA at step 7150 likely shifted weight distributions
- int6_sw_bpb 1.20684 vs Crawler Leg 3 SOTA 1.18720 — behind by 0.0196 at matched wallclock
- Cannon ablations (BWE) pending — per-loop output calibration may close gap

## Reference

| Config | int6_sw_bpb | bytes |
|--------|-------------|-------|
| Crawler Leg 3 SOTA | 1.18720 | 8.84MB |
| **BW3 seed=444** | **1.20684** | **10.07MB** |
| BWCD-02 (1-shard proxy) | 1.43531 | — |
