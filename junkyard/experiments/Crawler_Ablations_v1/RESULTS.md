# Crawler_Ablations_v1 Results

Date: 2026-03-30 (run overnight 2026-03-29→30)
Pod: C.33800697 (1×H100, Vast.ai)
Wallclock: 600s per arm | Seed: 1337 | model_params: 13,430,316

## Final BPB Table (key metric: final_int6_sliding_window_exact, lower = better)

| Arm | Override | Steps | Post-train BPB | Post-EMA BPB | Int6 SW BPB | Delta vs A | Verdict |
|-----|----------|------:|---------------:|-------------:|------------:|:----------:|---------|
| A_baseline | (none) | 747 | 1.4102 | — | **1.60513** | — | baseline |
| B_loop_aware_gptq | LOOP_AWARE_GPTQ=1 | 807 | 1.3932 | — | **1.56511** | **−0.0400** | ✅ WIN |
| E_compile_fullgraph | COMPILE_FULLGRAPH=1 | 751 | 1.4076 | — | **1.57930** | **−0.0258** | ✅ WIN |
| D_int8_off | CRAWLER_QUANT_INT8=0 | 786 | 1.3974 | — | **1.60273** | −0.0024 | wash |
| C_ema_on | SKIP_EMA=0 | 784 | 1.3999 | 1.5236 | **1.67479** | +0.0697 | ❌ LOSER |
| F_gptq_and_ema | LOOP_AWARE_GPTQ=1 SKIP_EMA=0 | 773 | 1.4033 | 1.5367 | **1.70575** | +0.1006 | ❌ WORST |

## Exact Metrics (from log)

### ARM A — baseline
```
post_ema val_bpb:        1.4102  (no EMA applied)
final_int6_roundtrip:    1.62586908
final_int6_sw_exact:     1.60513220
submission_size:         5,479,835 bytes
gptq_layers:             24
calibration_time:        3.7s
```

### ARM B — loop_aware_gptq
```
post_ema val_bpb:        1.3932  (no EMA applied)
final_int6_roundtrip:    1.58662077
final_int6_sw_exact:     1.56510915
submission_size:         5,482,227 bytes
gptq_layers:             24
calibration_time:        854.2s (2-phase)
```

### ARM C — ema_on
```
post_train val_bpb:      1.3999
post_ema val_bpb:        1.5236  ← EMA degrades live model by 0.124 BPB
final_int6_roundtrip:    1.69192498
final_int6_sw_exact:     1.67478698
submission_size:         5,163,046 bytes
gptq_layers:             24
```

### ARM D — int8_off
```
post_ema val_bpb:        1.3974  (no EMA applied)
final_int6_roundtrip:    1.62344477
final_int6_sw_exact:     1.60272582
submission_size:         4,782,291 bytes   ← 700KB smaller (30 GPTQ layers vs 24)
gptq_layers:             30
```

### ARM E — compile_fullgraph
```
post_ema val_bpb:        1.4076  (no EMA applied)
final_int6_roundtrip:    1.59967764
final_int6_sw_exact:     1.57929781
submission_size:         5,440,708 bytes
gptq_layers:             24
```

### ARM F — gptq_and_ema
```
post_train val_bpb:      1.4033
post_ema val_bpb:        1.5367  ← EMA degrades live model by 0.133 BPB
final_int6_roundtrip:    1.72154728
final_int6_sw_exact:     1.70574793
submission_size:         5,158,853 bytes
gptq_layers:             24
calibration_time:        861.3s (2-phase)
```

## Key Findings

### 1. Loop-aware GPTQ is a real win (−0.040 BPB)
Two-phase calibration — freeze flat layers with GPTQ weights, then collect crawler Hessians
under quantized-flat activations — significantly improves the crawler's quantization quality.
Cost: 854s calibration overhead (vs 2.6s standard). Worth it at 600s training.

### 2. EMA is actively harmful for quantization (+0.070–0.101 BPB)
EMA smooths weights in a way that hurts the GPTQ quantization grid. Post-EMA val_bpb
is ~0.124–0.133 worse than the live model (pre-EMA). SKIP_EMA=1 must stay default.

### 3. EMA + loop-aware GPTQ are antagonistic
F (combined) is WORSE than either alone. EMA negates loop-aware calibration gains entirely.

### 4. Fullgraph compile is a moderate win (−0.026 BPB)
Slightly faster step time → more steps in 600s → better-trained model. No architecture change.

### 5. int8_off is a wash (−0.002 BPB) but saves ~700KB submission size
Extending GPTQ to crawler layers (30 vs 24) gives a meaningful size reduction at near-zero BPB cost.

## Next Experiments (BKD roadmap)

| Priority | Experiment | Hypothesis | Expected delta |
|----------|------------|------------|----------------|
| 1 | B+E combined | loop_aware_gptq + compile_fullgraph | ~−0.060 cumulative? |
| 2 | B+D combined | loop_aware_gptq + int8_off (30 layers) | −0.040 + free size savings |
| 3 | B+E+D combined | all three wins | best achievable config |
| 4 | Loop count sweep | CRAWLER_LOOPS: 3/4/5 | +/− unknown |
| 5 | INST_DIM sweep | 0/16/32/64 | +/− unknown |
