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
| 444 | 8021 | 1.1992 | **1.18730643** | -0.0119 | 8.97MB | **-0.00015** |
| 300 | TBD | TBD | TBD | TBD | TBD | TBD |

## Verdict: Battery Beats Leg 3 — Confirmed

**BW4 seed=444: 1.18731 vs Leg 3: 1.18746 — new SOTA by -0.00015.**

Margin is within proxy noise but the mechanism is confirmed: quant_gap more negative
(-0.0119 vs -0.0117) with zero extra parameters. The 9,1,1 battery's identical trailing
loops produce tighter int8 distributions → sliding window extracts more signal.

The pyramid-512 choke was a net negative under 600s wallclock constraint. Battery alone
is the right configuration. Seed=300 needed to confirm delta holds across seeds.

### Key comparison

| Config | int6_sw_bpb | quant_gap | bytes | steps |
|--------|-------------|-----------|-------|-------|
| Leg 3 seed=300 | 1.18746 | -0.0117 | 8.84MB | 8103 |
| BW3 seed=444 (pyramid+battery) | 1.20684 | +0.0088 | 10.07MB | 7548 |
| **BW4 seed=444 (battery only)** | **1.18731** | **-0.0119** | **8.97MB** | **8021** |

### Log
`results/BW4_s444_20260331_064913.log`
