# Bandit_Wagon_IV — Results

## Hypothesis

BW3 showed that pyramid-512 choke + 9,1,1 battery (1.20684) was behind Leg 3 SOTA (1.18720).
Diagnosis: pyramid-512 adds ~1.57M params, slowing convergence under 600s wallclock.
The battery (9,1,1) is free — zero extra params. Test battery alone on Leg 3 base.

## Architecture

Leg 3 base + 9,1,1 battery. **No pyramid choke.**

- `CRAWLER_MLP_CHOKE_DIM=0` (choke disabled)
- `CRAWLER_LOOP_ROPE_SCALES=9,1,1`
- `NUM_FLAT_LAYERS=4` / `NUM_CRAWLER_LAYERS=1` / `CRAWLER_LOOPS=3`
- `CRAWLER_MLP_MULT=6.0` / `INST_DIM=32`
- `MLP_LEAKY_SLOPE=0.5` / `CRAWLER_MLP_LEAKY_SLOPE=0.5`
- `XSA_LAST_N=11`
- `SKIP_GPTQ=1` / `SKIP_EMA=1`
- `MAX_WALLCLOCK_SECONDS=600` / `WARMDOWN_ITERS=2000`

## Reference

| Config | int6_sw_bpb | bytes | Notes |
|--------|-------------|-------|-------|
| Leg 3 SOTA | 1.18720 | 8.84MB | no choke, no battery |
| BW3 seed=444 | 1.20684 | 10.07MB | pyramid-512 + battery |
| **BW4 target** | **< 1.18720** | **~8.84MB** | **battery only** |

## Results

| Seed | Steps | raw_bpb | int6_sw_bpb | quant_gap | bytes | vs Leg 3 |
|------|-------|---------|-------------|-----------|-------|----------|
| 444 | TBD | TBD | TBD | TBD | TBD | TBD |
| 300 | TBD | TBD | TBD | TBD | TBD | TBD |
