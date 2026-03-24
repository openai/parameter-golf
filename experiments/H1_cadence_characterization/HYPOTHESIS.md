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

Note: cad0 full-scale used diagnostic script (156ms/step), NOT production script.
Run 8 (cadence 2, production script) got 7076 steps in 600s at ~85ms/step.
Comparison is confounded by step count difference. Clean A/B pending.

## Verdict

**PREDICTION PARTIALLY REFUTED.** No U-shape found at 0.25 scale.

At 0.25 scale (150s), less recursion is monotonically better:
- val@500 is identical across all cadences (1.3838-1.3842) — C-steps are neutral per step
- More steps in same wallclock → better final BPB
- Quant gap shrinks monotonically: 0.136 → 0.081 → 0.061 → 0.059
- Winner: **cad4** (1.3836 sliding)

Open question: does recursion compound at full scale (7000 steps)?
Production-script cad0 vs Run 8 test pending.
