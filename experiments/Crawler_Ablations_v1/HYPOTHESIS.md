# Crawler_Ablations_v1 Hypothesis

Date: 2026-03-29
Pod: C.33800697 (1×H100, Vast.ai)

## Mission
Quantify the effect of post-training policies (GPTQ calibration strategy, EMA, compile mode, int8 quant scope) on final int6 sliding-window BPB. Single-variable ablation against a clean baseline.

## Hard Rules
1. `DELTA_NET_HEADS=0` — DeltaNet quarantined for all arms.
2. `NGRAM_EVAL_ORDER=0` — ngram eval off.
3. `NITRUST_ENABLE=0` — Nitrust disabled for clean comparison.
4. All arms: 600s wallclock, seed 1337, 1 GPU.
5. Key metric: `final_int6_sliding_window_exact` (lower = better).

## Arms

| Arm | Override | Hypothesis |
|-----|----------|------------|
| A_baseline | (none) | Pure baseline reference |
| B_loop_aware_gptq | LOOP_AWARE_GPTQ=1 | 2-phase Hessian calibration accounts for quantized-flat activations seen by crawler layers |
| C_ema_on | SKIP_EMA=0 | EMA weights are smoother — may compress better under GPTQ |
| D_int8_off | CRAWLER_QUANT_INT8=0 | Extend GPTQ to crawler layers (30 layers vs 24) for smaller submission |
| E_compile_fullgraph | COMPILE_FULLGRAPH=1 | Fullgraph compile fits more steps in 600s → better-trained model |
| F_gptq_and_ema | LOOP_AWARE_GPTQ=1 SKIP_EMA=0 | Combined effect of B+C |

## Exit Criteria
- Rank all 6 arms by final BPB.
- Promote any arm with delta > 0.010 improvement to next production baseline.
- Kill any arm that regresses > 0.020.
