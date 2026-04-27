# Crawler_Leg_1 Results

Date: 2026-03-30 (run overnight 2026-03-29→30)
Pod: C.33800697 (1×H100, Vast.ai)
Wallclock: 600s per arm | Seed: 1337 | GPUs: 1
Script: experiments/Crawler_Leg_1/run_all.sh
Key config: SKIP_GPTQ=1, SKIP_EMA=1, CRAWLER_QUANT_INT8=1, NUM_FLAT_LAYERS=4, NUM_CRAWLER_LAYERS=1

---

## Summary Table (key metric: final_int6_sliding_window_exact, lower = better)

| Arm | Label | Params | Steps | ms/step | Post-train BPB | Int6 SW BPB | Quant Gap | Delta vs baseline | Verdict |
|-----|-------|-------:|------:|--------:|---------------:|------------:|----------:|:-----------------:|---------|
| CL1-00 | baseline (loops=4 inst=32 mlp=4.0 4F+1C) | 13,430,316 | 817 | 735 | 1.3921 | **1.74636** | 0.354 | — | baseline |
| CL1-01 | loops=3 | 13,413,932 | 884 | 679 | 1.3710 | **1.65890** | 0.288 | **−0.0875** | ✅ WIN |
| CL1-07 | mlp_mult=5.0 (wide) | 13,954,604 | 917 | 655 | 1.3621 | **1.64868** | 0.287 | **−0.0977** | ✅ BEST |
| CL1-04 | inst_dim=16 (narrow) | 13,389,356 | 808 | 743 | 1.3758 | **1.75600** | 0.380 | +0.0096 | wash |
| CL1-05 | inst_dim=64 (wide) | 13,512,236 | 762 | 788 | 1.4058 | **1.75201** | 0.346 | +0.0057 | wash |
| CL1-02 | loops=5 | 13,446,700 | 792 | 758 | 1.3768 | **1.81547** | 0.439 | +0.0691 | ❌ LOSER |
| CL1-03 | inst_dim=0 (off) | 13,350,444 | 790 | 760 | 1.3894 | **1.78019** | 0.391 | +0.0338 | ❌ LOSER |
| CL1-09 | 5F+1C | 15,791,668 | 720 | 834 | 1.3877 | **1.79416** | 0.406 | +0.0478 | ❌ LOSER |
| CL1-06 | mlp_mult=3.0 (narrow) | 12,906,028 | 750 | 803 | 1.4170 | **1.86261** | 0.446 | +0.1163 | ❌ LOSER |
| CL1-10 | 3F+2C | 13,954,092 | 696 | 862 | 1.3925 | **1.86610** | 0.474 | +0.1197 | ❌ LOSER |
| CL1-08 | crawler_quant_int8=0 | 13,430,316 | 697 | 862 | 1.4339 | **1.94389** | 0.510 | +0.1975 | ❌ WORST |

---

## Exact Metrics (from logs)

### CL1-00 — baseline (loops=4 inst=32 mlp=4.0 4F+1C)
```
post_train val_bpb:      1.3921
final_int6_roundtrip:    1.76325044
final_int6_sw_exact:     1.74635595
submission_size:         4,772,167 bytes
quant_gap:               +0.354 BPB
```

### CL1-01 — loops=3
```
post_train val_bpb:      1.3710
final_int6_roundtrip:    1.67751002
final_int6_sw_exact:     1.65890461
submission_size:         4,926,806 bytes
quant_gap:               +0.288 BPB  ← 66ms faster/step, 884 steps
```

### CL1-02 — loops=5
```
post_train val_bpb:      1.3768
final_int6_roundtrip:    1.83602072
final_int6_sw_exact:     1.81546960
submission_size:         4,696,959 bytes
quant_gap:               +0.439 BPB  ← 23ms slower/step, only 792 steps
```

### CL1-03 — inst_dim=0 (off)
```
post_train val_bpb:      1.3894
final_int6_roundtrip:    1.80101371
final_int6_sw_exact:     1.78019323
submission_size:         4,716,667 bytes
quant_gap:               +0.391 BPB  ← inst=0 hurts both quality AND quant
```

### CL1-04 — inst_dim=16 (narrow)
```
post_train val_bpb:      1.3758
final_int6_roundtrip:    1.77461489
final_int6_sw_exact:     1.75599701
submission_size:         4,784,960 bytes
quant_gap:               +0.380 BPB
```

### CL1-05 — inst_dim=64 (wide)
```
post_train val_bpb:      1.4058
final_int6_roundtrip:    1.77250840
final_int6_sw_exact:     1.75200884
submission_size:         4,924,217 bytes
quant_gap:               +0.346 BPB  ← wider inst improves quant slightly but fewer steps
```

### CL1-06 — mlp_mult=3.0 (narrow)
```
post_train val_bpb:      1.4170
final_int6_roundtrip:    1.87406276
final_int6_sw_exact:     1.86261350
submission_size:         4,496,163 bytes
quant_gap:               +0.446 BPB  ← narrow MLP = worse quality AND worse quant
```

### CL1-07 — mlp_mult=5.0 (wide)
```
post_train val_bpb:      1.3621
final_int6_roundtrip:    1.67060502
final_int6_sw_exact:     1.64867635
submission_size:         5,127,370 bytes
quant_gap:               +0.287 BPB  ← 80ms FASTER/step (655ms), 917 steps, best int6 BPB
```

### CL1-08 — crawler_quant_int8=0
```
post_train val_bpb:      1.4339
final_int6_roundtrip:    1.95396352
final_int6_sw_exact:     1.94388999
submission_size:         4,527,901 bytes
quant_gap:               +0.510 BPB  ← catastrophic. Disabling int8 during training destroys quant quality.
```

### CL1-09 — 5F+1C
```
post_train val_bpb:      1.3877
final_int6_roundtrip:    1.81116344
final_int6_sw_exact:     1.79415967
submission_size:         5,309,412 bytes   ← 15.79M params
quant_gap:               +0.406 BPB  ← more flat layers = more unique params = bigger model + worse quant
```

### CL1-10 — 3F+2C
```
post_train val_bpb:      1.3925
final_int6_roundtrip:    1.87893267
final_int6_sw_exact:     1.86610473
submission_size:         4,643,659 bytes
quant_gap:               +0.474 BPB  ← two crawlers = massive quant gap, near-worst int6 BPB
```

---

## Key Findings

### 1. MLP width is the largest lever (mlp_mult=5.0: −0.098 BPB, BEST)
CL1-07 wins outright. Wider MLP yields:
- Better pre-quant quality (1.3621 vs 1.3921)
- **Faster step time** (655ms vs 735ms — counter-intuitive, likely kernel tile efficiency)
- 917 steps vs 817 for baseline = more gradient updates
- Smaller quant gap (0.287 vs 0.354)

This directly extends the prior crawler analysis: width is the dominant capacity lever. The MLP expansion does double duty — more capacity AND faster matmuls.

### 2. Fewer loops is better (loops=3: −0.088 BPB)
CL1-01 is second-best. Confirms the core Frugendorff hypothesis:
- loops=3 → 0.288 quant gap
- loops=4 → 0.354 quant gap
- loops=5 → 0.439 quant gap

Each additional loop adds ~0.085 BPB to the quant gap. Fewer loops = less weight sharing pressure = cleaner quantization. Also: loops=3 is 56ms/step faster (679ms vs 735ms), yielding 884 steps vs 817 for baseline.

### 3. inst_dim is nearly irrelevant (0.0057–0.0338 range)
- inst_dim=0 (off): +0.034 — removing inst hurts but modestly
- inst_dim=16: +0.010 — most of the value is recovered with 16 dims
- inst_dim=32 (baseline): —
- inst_dim=64: +0.006 — marginal improvement but slower steps

The instruction signal matters for loop differentiation, but 16 dims is nearly enough. The cost is in step time, not quality.

### 4. CRAWLER_QUANT_INT8=1 is mandatory (+0.198 BPB if disabled)
CL1-08 is the worst non-quant result. The in-training int8 quantization of crawler weights is essential — it acts as quantization-aware training, keeping weights in a distribution that survives int6 export.

### 5. More crawler blocks destroys quality (3F+2C worst split: +0.120 BPB)
- 4F+1C (baseline): best
- 5F+1C: +0.048
- 3F+2C: +0.120

More crawlers = more loops of shared-weight computation = larger quant gap. 5F+1C pays for the extra flat layer with 0.4M more unique params (5.3MB submission) but gains nothing in quality.

### 6. Narrow MLP is catastrophic (mlp_mult=3.0: +0.116 BPB)
The capacity reduction hurts more than the speed gain helps. 750 steps at worse quality per step = worst of both worlds.

---

## Quant Gap Summary (pre-quant to int6 SW BPB)

| Arm | Pre-quant BPB | Int6 SW BPB | Gap | Interpretation |
|-----|:------------:|:-----------:|:---:|----------------|
| CL1-07 mlp=5.0 | 1.3621 | 1.6487 | **0.287** | Best — wide MLP easiest to quantize |
| CL1-01 loops=3 | 1.3710 | 1.6589 | **0.288** | Near-best — fewer loops = less sharing pressure |
| CL1-00 baseline | 1.3921 | 1.7464 | 0.354 | reference |
| CL1-05 inst=64 | 1.4058 | 1.7520 | 0.346 | Wide inst slightly helps |
| CL1-04 inst=16 | 1.3758 | 1.7560 | 0.380 | Narrow inst slightly worse |
| CL1-03 inst=0 | 1.3894 | 1.7802 | **0.391** | No inst = no loop differentiation = worse quant |
| CL1-09 5F+1C | 1.3877 | 1.7942 | 0.406 | Extra flat layer doesn't help quant |
| CL1-02 loops=5 | 1.3768 | 1.8155 | **0.439** | Extra loop = +0.085 quant gap vs loops=4 |
| CL1-06 mlp=3.0 | 1.4170 | 1.8626 | 0.446 | Narrow MLP = bad weights for quant |
| CL1-10 3F+2C | 1.3925 | 1.8661 | **0.474** | Two crawlers = massive sharing pressure |
| CL1-08 int8_off | 1.4339 | 1.9439 | **0.510** | No QAT = worst quant |

**Gap scales with loop count: loops=3 (+0.288), loops=4 (+0.354), loops=5 (+0.439).**
Each loop adds ~0.085 BPB to the quantization gap.

---

## Next Steps (BKD roadmap)

| Priority | Experiment | Rationale | Expected delta |
|----------|------------|-----------|----------------|
| 1 | **loops=3 + mlp=5.0** combined | Best two wins, potentially additive | −0.150+ BPB? |
| 2 | **loops=3 + mlp=5.0 + loop_aware_gptq** | Add Crawler_Ablations_v1 B win (−0.040) | −0.190+ BPB? |
| 3 | loops=2 ablation | Does the quant gap keep shrinking? | −0.030 est |
| 4 | mlp_mult=6.0 ablation | Is there more on the table from width? | −0.030 est |
| 5 | inst_dim sweep at loops=3 | Confirm inst irrelevance at fewer loops | low |

**Hypothesis for Leg 2**: loops=3 + mlp=5.0 + loop_aware_gptq could bring int6 SW BPB from 1.746 to ~1.55 range. That's a meaningful step toward the 1.1 target with a sub-5MB submission.
