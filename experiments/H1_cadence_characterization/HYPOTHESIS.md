# H1: Cadence Characterization on 4x2 (RC-0)

## Question
What is cadence doing to BPB in a balanced 4f+2cx2 recursive system?

## Prediction
Cadence 2 (C/N alternating) is near-optimal because:
- Cadence 1 (all C): doubles compute but ref never gets N-step outbound gradient.
  The PD channel is always in "write" mode, never "read." Expect worse BPB per wall-second.
- Cadence 2: balanced read/write on the PD channel. N-steps let the ref's gradient
  propagate back through the crawler without competing with the C-step consensus update.
- Cadence 3-4: starves the ref of C-step updates. The deliberation mechanism goes dormant.
  Expect delib_scale to plateau or decay.

We expect a U-shaped curve: BPB worst at cadence 1 (compute waste) and cadence 4
(PD starvation), best at cadence 2 or 3.

## Architecture (held constant)
```
NUM_FLAT_LAYERS=4  NUM_CRAWLER_LAYERS=2  CRAWLER_LOOPS=2
MODEL_DIM=640  NUM_HEADS=10  NUM_KV_HEADS=5  MLP_MULT=4
XSA_LAST_N=2  VE_LAYERS=0,1
```

## Arms

| Arm | DIAG_FIXED_CADENCE | C-step ratio | Parent |
|-----|-------------------|--------------|--------|
| cad1 | 1 | 100% (all C) | RC-0 |
| cad2 | 2 | 50% (C/N) | RC-0 (control) |
| cad3 | 3 | 33% (C/N/N) | RC-0 |
| cad4 | 4 | 25% (C/N/N/N) | RC-0 |

## Scale
0.25 (150s wallclock, 625 warmdown, TTT/distill OFF)

## Diagnostic Focus
1. `delib_scale` trajectory — does PD stay alive across cadences?
2. `fast_val_bpb` at wall-clock matched checkpoints
3. `train_loss` split by `is_crawl` — are C-steps helping or hurting?
4. Total steps achieved (cadence 1 will get fewer)
5. `quant_gap` — does cadence affect quantization friendliness?

## Results (2026-03-24, 8xH100 SXM)

| Arm | Steps | step_avg | val@500 | final_val | post_ema | sliding_bpb | quant_gap |
|-----|-------|----------|---------|-----------|----------|-------------|-----------|
| cad1 | 702 | 213ms | 1.3842 | 1.3736 | 1.4790 | **1.5092** | 0.136 |
| cad2 | 810 | 185ms | 1.3841 | 1.3409 | 1.4103 | **1.4222** | 0.081 |
| cad3 | 854 | 176ms | 1.3839 | 1.3328 | 1.3875 | **1.3941** | 0.061 |
| cad4 | 878 | 171ms | 1.3838 | 1.3249 | 1.3780 | **1.3836** | 0.059 |

### cad0 Full Scale (600s, production diag script, TTT+distill ON)
| Steps | val@500 | val@3828 | post_ema | sliding_bpb | quant_gap |
|-------|---------|----------|----------|-------------|-----------|
| 3828 | 1.4017 | 1.1853 | 1.1794 | **1.1603** | 0.004 |

Note: cad0 diag-script run was confounded (different script, fewer steps). See production run below.

### cad0 Full Scale — PRODUCTION SCRIPT (apples-to-apples vs Run 8)
| | Run 8 (cad2) | cad0 (no C) | Delta |
|---|---|---|---|
| Script | production | **production** | **same** |
| Steps | 7,076 | **7,856** | +11% |
| step_avg | ~85ms | **76ms** | faster |
| Peak memory | 33,182 MiB | **22,854 MiB** | **-31%** |
| post_ema | 1.1535 | **1.1487** | -0.005 |
| **sliding_window** | **1.1355** | **1.1325** | **-0.003** |
| quant_gap | 0.0075 | 0.0070 | -0.0005 |

Learning curve (cad0 production, no C-steps):
```
step  500: 1.4032    step 4000: 1.2366
step 1000: 1.3234    step 4500: 1.2315
step 1500: 1.2984    step 5000: 1.2286
step 2000: 1.2678    step 5500: 1.2204
step 2500: 1.2537    step 6000: 1.2102
step 3000: 1.2449    step 6500: 1.1946
step 3500: 1.2405    step 7000: 1.1809
                     step 7500: 1.1622
                     step 7856: 1.1512
```

## Status
COMPLETED — recursion/cadence mechanism is deprecated for the primary race path.

## Verdict

**PREDICTION REFUTED. Recursion is net overhead at all tested scales.**

At 0.25 scale (150s, ~800 steps):
- val@500 identical across cadences — C-steps neutral per step
- More steps in same wallclock → better final BPB
- Quant gap shrinks monotonically: 0.136 → 0.059
- Winner: cad4 (1.3836 sliding)

At 1.0 scale (600s, ~7800 steps, PRODUCTION SCRIPT):
- **cad0 (no C-steps) beats Run 8 by 0.003 BPB** (1.1325 vs 1.1355)
- 11% more steps (no C-step compute overhead)
- 31% less memory (no double-firing activation storage)
- Quant gap slightly better (0.0070 vs 0.0075)
- Recursion does NOT compound at 7000 steps

**The C-step double-firing mechanism provides zero measurable benefit.**
The architecture's value comes from weight sharing, trigram embedding, XSA,
VE injection, GPTQ, SWA, TTT burst, and self-distillation — not recursion.

Next step: isolate and validate each remaining component.
