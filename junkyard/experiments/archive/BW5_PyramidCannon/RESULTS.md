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

| ARM | step_avg | raw_bpb | int6_rt_bpb | int6_sw_bpb | size_bytes |
|-----|----------|---------|-------------|-------------|------------|
| BWVPC-00 control (flat+none) | 74.40ms | 1.3069 | 1.31209610 | 1.28787686 | 9,415,826 |
| BWVPC-01 pyramid+scalar cannon | 79.33ms | 1.3283 | 1.34492218 | 1.32227987 | 10,408,358 |
| delta | +4.93ms | **+0.0214** | **+0.02283** | **+0.03440** | +992,532 |

Train loss crossover: pyramid+cannon wins at step 500 (2.4767 vs 2.4926) but falls behind by step 1000 (2.3639 vs 2.3598) and keeps diverging to step 2000 (2.1825 vs 2.1370).

## Verdict: DOES NOT PROMOTE

**Hard failure.** int6_sw_bpb regression of +0.03440 at 2000 steps is decisive.

**Root cause:** 1.57M cold choke params are a training burden that compounds over time.
The 1GPU 500-step proxy captured early structural advantage only — proxy was badly misleading here.

**Pyramid concept notes for future:**
- Smaller choke dim (128 or 256) — less cold param burden
- Warm initialization of bottleneck weights
- Dedicated LR schedule for choke layers
- Or: investigate whether pyramid helps only at very long training runs (>>8000 steps)
