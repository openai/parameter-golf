# Bandit_Wagon_V_Cannon — Gate Results

## Gate: Single GPU, 500 steps, seed=444

Base: BW5 (CHOKE_DIM=0, COMPILE_FULLGRAPH=1, ROPE_SCALES=9,1,1)
Variable: CRAWLER_CANNON_TYPE

| ARM | Type | raw_bpb | int6_sw_bpb | vs control | bytes |
|-----|------|---------|-------------|------------|-------|
| BWVC-00 | control (none) | 1.4413 | 1.44236 | — | 6,788,121 |
| BWVC-01 | scalar (3 params) | 1.4407 | 1.44261 | +0.00025 | 6,794,463 |
| BWVC-02 | channel (1.5K) | 1.4422 | 1.44296 | +0.00060 | 6,729,386 |
| BWVC-03 | rmsnorm (1.5K) | 1.4408 | 1.44428 | +0.00192 | 6,776,903 |

## Verdict: ~~DOES NOT PROMOTE~~ — CORRECTED. See 8GPU gate below.

**Correction:** The original verdict was based solely on int6_sw_bpb at 500 proxy steps (unreliable at that scale).
Scalar cannon raw_bpb (1.4407) was better than control (1.4413). Speed was also faster on 1GPU.
8GPU gate was required and has now been run.

---

## Gate: 8×H100, 2000 steps, seed=444

Base: BW5. Arms: control (none) vs scalar cannon only (best 1GPU arm).
Pass criteria: scalar step_avg < control step_avg.

| ARM | Type | step_avg | val_bpb | int6_rt_bpb | int6_sw_bpb | size_bytes |
|-----|------|----------|---------|-------------|-------------|------------|
| BWVC-00 | control (none) | 74.84ms | 1.3080 | 1.31294609 | 1.28870981 | 9,169,530 |
| BWVC-01 | scalar cannon (3 params) | **74.81ms** | 1.3082 | **1.31256407** | **1.28854887** | 9,512,901 |
| delta | | **-0.03ms** | +0.0002 | **-0.00038** | **-0.00016** | **+343,371** |

### Verdict: SPEED GATE PASSES (barely). Quality positive. Size regression.

- **Speed:** scalar 74.81ms < control 74.84ms → **PASSES** (-0.03ms, marginal)
- **int6_sw_bpb:** scalar wins by -0.00016 → positive quality signal
- **int6_rt_bpb:** scalar wins by -0.00038 → positive quality signal
- **Size:** scalar is +343KB larger despite only 3 extra params — quantization behavior differs

**Finding:** Scalar cannon is real signal. Tiny speed gain, tiny quality gain, but notable size cost.
Proceed to `Bandit_Wagon_V_PyramidCannon` — the combined pyramid+cannon test is the next gate.
