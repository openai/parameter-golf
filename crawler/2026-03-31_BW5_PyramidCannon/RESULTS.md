# BW5_PyramidCannon — Gate Results

## Architecture

BW5 + pyramid (CHOKE_DIM=512) + scalar cannon. Two-variable combined test.
Prerequisites: cannon 8GPU speed gate passed, pyramid 1GPU quality gate passed.

- `CRAWLER_MLP_CHOKE_DIM=512`, `CRAWLER_MLP_CHOKE_SHAPE=pyramid`
- `CRAWLER_CANNON_TYPE=scalar`
- All other flags identical to BW5

---

## Gate: Single GPU, 500 steps, seed=444

Base: BW5 flat+none control. Test: pyramid+scalar cannon combined.
Note: 1GPU gate uses grad_accum=8. step_avg ÷ 8 ≈ real per-micro-batch overhead.

| ARM | model_params | step_avg | raw_bpb | int6_rt_bpb | int6_sw_bpb | size_bytes |
|-----|-------------|----------|---------|-------------|-------------|------------|
| BWVPC-00 control (flat+none) | 14,462,508 | 585.42ms | 1.4435 | 1.46507573 | 1.44361588 | 6,736,451 |
| BWVPC-01 pyramid+scalar cannon | 16,035,372 | 609.26ms | **1.4344** | **1.45744500** | **1.43527438** | 7,498,346 |
| delta | +1,572,864 | **+23.84ms** | **-0.0091** | **-0.00763** | **-0.00834** | +761,895 |

### Quality: PASSES
- raw_bpb: **-0.0091**
- int6_sw_bpb: **-0.00834**
- int6_rt_bpb: **-0.00763**
- All quality metrics clearly positive

### Speed: PASSES
- +23.84ms on 1GPU (grad_accum=8) → ~+3ms per micro-batch on 8×H100
- Within budget

### Size: +762KB — consistent with pyramid alone (+748KB), cannon adding negligible size

### Cannon's incremental contribution: INCONCLUSIVE
Cross-run control repin variance (~0.003 BPB) swamps the signal.
Cannot cleanly isolate cannon's effect on top of pyramid at 500 steps.
What is clear: the combined pair passes and the quality signal is real.

## Verdict: 1GPU GATE PASSES. Proceed to gate_8gpu.sh.

---

## Gate: 8×H100, 2000 steps, seed=444

*Run after 1GPU gate passes. Results below when complete.*

| ARM | step_avg | val_bpb | int6_rt_bpb | int6_sw_bpb | size_bytes |
|-----|----------|---------|-------------|-------------|------------|
| BWVPC-00 control (flat+none) | | | | | |
| BWVPC-01 pyramid+scalar cannon | | | | | |
| delta | | | | | |

## Verdict: PENDING
