# Bandit_Wagon_V_Pyramid — Gate Results

## Architecture

BW5 + `CRAWLER_MLP_CHOKE_DIM=512` (pyramid shape).
Single validated change vs BW5 control.

- `CRAWLER_MLP_CHOKE_DIM`: 0 (flat) vs 512 (pyramid)
- `CRAWLER_MLP_CHOKE_SHAPE`: flat vs pyramid
- `CRAWLER_MLP_CHOKE_GROUPS=8`
- All other flags identical to BW5

---

## Gate: Single GPU, 500 steps, seed=444

Base: BW5 (CHOKE_DIM=0, COMPILE_FULLGRAPH=1, ROPE_SCALES=9,1,1)
Variable: CRAWLER_MLP_CHOKE_DIM (0=flat vs 512=pyramid)

Note: 1GPU gate uses grad_accum=8. step_avg here ≠ 8GPU step_avg.
Per-micro-batch overhead = +27.22ms / 8 ≈ **+3.4ms** — likely within 8×H100 budget.

| ARM | CHOKE_DIM | model_params | step_avg | raw_bpb | int6_rt_bpb | int6_sw_bpb | size_bytes |
|-----|-----------|-------------|----------|---------|-------------|-------------|------------|
| BWVP-00 | 0 (flat) | 14,462,508 | 583.99ms | 1.4432 | 1.46801971 | 1.44668780 | 6,750,039 |
| BWVP-01 | 512 (pyramid) | 16,035,372 | 611.21ms | **1.4339** | **1.45855878** | **1.43681894** | 7,497,734 |
| delta | | +1,572,864 | **+27.22ms** | **-0.0093** | **-0.00946** | **-0.00987** | +747,695 |

### Quality: STRONG PASS
- raw_bpb: **-0.0093** — one of the largest proxy deltas in this series
- int6_sw_bpb: **-0.00987**, int6_rt_bpb: **-0.00946**
- All quality metrics clearly positive

### Speed: NEEDS 8GPU CONFIRMATION
- +27.22ms on 1GPU (grad_accum=8) → ~+3.4ms per micro-batch
- On 8×H100, true overhead likely 3–5ms — within budget. Must confirm.

### Size: +747KB — expected from 1.57M extra params (choke bottleneck + expansion per loop)

## Verdict: QUALITY PASSES STRONGLY. Proceed to gate_8gpu.sh.

---

## Gate: 8×H100, 2000 steps, seed=444

*Run only if 1GPU gate passes.*

| ARM | CHOKE_DIM | step_avg | val_bpb | int6_rt_bpb | int6_sw_bpb | size_bytes |
|-----|-----------|----------|---------|-------------|-------------|------------|
| BWVP-00 | 0 (flat) | | | | | |
| BWVP-01 | 512 (pyramid) | | | | | |
| delta | | | | | | |

## Verdict: PENDING
