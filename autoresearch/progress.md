# Autoresearch Progress

## Session Info
- **Started:** 2026-03-22
- **Branch:** autoresearch/mar26
- **Best val_bpb so far:** 1.3039 raw / 1.3100 roundtrip (11L MLP3x partial RoPE all-int5, 15.0MB)
- **Total experiments:** ~100
- **Status:** running — testing leaky_relu_sq activation (SOTA #1 uses this)

## Current Experiment: Hadamard Rotation
- **GPUs:** 0,1,2,3 (4× A6000)
- **Change:** HADAMARD_ROTATE=True — block-diagonal Hadamard rotation before int5 quantization
- **Expected:** ~1.3039 raw, ~1.3090 roundtrip (-0.001 BPB), ~14.4MB (-0.6MB)

## Recent Results (Mar 26-27)
| Experiment | val_bpb | roundtrip | size_MB | status |
|---|---|---|---|---|
| 11L MLP3x frac=0.4 int6/int5 | 1.3071 | 1.3116 | 16.97 | OVER |
| 11L MLP3x frac=0.2 int6/int5 | 1.3080 | 1.3132 | 16.18 | OVER |
| 11L MLP3x frac=0.0 all-int5 | 1.3072 | 1.3137 | 15.38 | FITS |
| 11L MLP3x FFT int4 | 1.3080 | 1.3349 (+0.027) | 12.9 | LOSSY |
| 11L partial RoPE 16/64 | 1.3039 | 1.3100 | 15.0 | **FITS** NEW BEST |
| partial RoPE + GPTQ-lite | 1.3018 | 1.3073 | 16.4 | OVER — GPTQ hurts compressibility |
| BigramHash(2048,128) | 1.3101 | 1.3164 | 15.2 | WORSE (+0.006 regression) |
| softcap=30 (3GPU) | 1.3394 | 1.3460 | 15.9 | inconclusive — per-step better |
| ROPE_DIM=32 | 1.3052 | 1.3118 | 14.8 | WORSE than ROPE_DIM=16 |
| SWA(0.10,30) 3GPU | 1.3406 | 1.3450 | 16.1 | WORSE — SWA degrades 0.008 |
| warmdown=9000 | 1.3047 | 1.3108 | 15.0 | WORSE than warmdown=7000 |
| Hadamard rotation | 1.3018 | 1.3075 | 16.0 | OVER — quality better but +1MB size |
| EMA(0.997) | 1.3197 | 1.3239 | 14.8 | WORSE — all-warmdown means last step is best |
| leaky_relu_sq | 1.3135 | 1.3193 | 14.8 | WORSE — 214ms/step (+18%), fewer steps kill gains |
| ROPE_DIM=8 | 1.3073 | 1.3138 | 15.2 | WORSE — 16 confirmed optimal |
| **MATRIX_LR=0.12** | running | — | — | — |

## Key Findings
1. **SiLU >> ReluSquared.** relu_sq is ~23ms/step slower, fewer total steps negate per-step gains.
2. **11L with INT6 > 10L.** INT6 quant compresses 11L to 15.3MB. Better quality per step offsets slower speed.
3. **grad_accum=1 is fastest.** Single forward/backward per step.
4. **Matrix LR 0.10 optimal.** Higher/lower both worse at full run.
5. **Batch 98K optimal.** 65K too noisy, 131K too few steps.
6. **Warmdown 7000 optimal.** 3000/5000 too short, 10000 too aggressive.
7. **Pretrained GPT-2 PCA embeddings.** +0.004 improvement (1.3181 without vs 1.3143 with).
8. **Muon WD 0.04 optimal.** Tested 0.02 and 0.06 both worse.
9. **MUON_MOMENTUM=0.90 > 0.95 > 0.85.** Sweet spot.
10. **Orthogonal init helps.** ~0.001 consistent improvement.
11. **SWA doesn't help.** With warmdown schedule, SWA degrades by ~0.008.
12. **BigramHash doesn't help.** 1.3200 vs 1.3143 baseline, adds 0.6MB.
13. **Wavelet init doesn't help.** 1.3207 vs 1.3143 baseline PCA init.
14. **12L doesn't fit.** INT6=16.6MB, tied=16.3MB, INT5 fits but quality loss too high.
15. **SOFTCAP=8 optimal.** Tested 5,6,9,15,20,30,100.
16. **Screening unreliable for small diffs.** 2-min screen correlates poorly for <0.02.
17. **MLP3x + tied + SmearGate.** Works well on new arch. SmearGate helps 0.012 BPB on MLP3x (was neutral on MLP2x).
18. **SOTA Muon settings don't work.** lr=0.025/mom=0.99 gave 1.3556 (needs more steps). Our lr=0.10/mom=0.90 better with fewer steps.
19. **11L MLP3x needs aggressive compression.** frac=0.4 gives 16.97MB, frac=0.2 gives 16.18MB. Need frac=0.0 or different approach.
20. **zstd-22 better than zlib-9.** ~8% compression improvement on quantized weights.
21. **Warmdown 9000 worse than 7000.** 1.3047 vs 1.3039 raw, 1.3108 vs 1.3100 roundtrip.
22. **Hadamard rotation: quality wins but size loses.** Roundtrip 1.3075 (best ever!) but model +1MB (16.0MB OVER). Rotation destroys zstd-friendly patterns.
23. **EMA(0.997) hurts.** All-warmdown schedule means last step is always best. EMA averages the final ~333 suboptimal states, giving +0.006 BPB. Only useful if there's a stable plateau phase.
24. **U-Net skip connections + resid_mix already in our arch.** Confirmed present from the beginning.

## SOTA Gap Analysis (2026-03-26)
**SOTA: 1.1194 BPB, Our best: 1.3071 (raw) / 1.3143 (under 16MB), Gap: 0.19 BPB**
Main gap driver: hardware (3×A6000 ~2560 steps vs 8×H100 ~7400 steps = 2.9x fewer tokens)

## Technique Status
- [x] MLP 3x + tied embeddings — ADOPTED (11L MLP3x)
- [x] SmearGate — ADOPTED (+0.012 on MLP3x)
- [x] INT6_QUANT — ADOPTED
- [x] Orthogonal init — ADOPTED
- [x] zstd compression — ADOPTED
- [x] Nonuniform quantization — ADOPTED (needed to fit 10L MLP3x)
- [x] relu_sq / leaky_relu_sq — WORSE (too slow)
- [x] SOTA Muon lr/mom — WORSE on our hardware
- [x] SWA — WORSE with warmdown
- [x] BigramHash — WORSE on old arch, untested on MLP3x
- [ ] Partial RoPE — untested
- [ ] XSA — untested
- [ ] EMA + SWA combo — untested
- [ ] GPTQ-lite — untested
- [ ] TTT — untested

## Trajectory
- Baseline: 2.2276 (1 GPU, default config)
- LR sweep: 2.2276 -> 1.7136
- Batch sweep: 1.7136 -> 1.5054
- 4 GPUs: 1.5054 -> 1.3811
- LR retune: 1.3811 -> 1.3595
- Untied embeddings: 1.3595 -> 1.3532
- Muon WD: 1.3532 -> 1.3446
- Pretrained embeddings: 1.3446 -> 1.3426
- SiLU activation: 1.3426 -> 1.3373
- Batch 96K + scalar LR: 1.3373 -> 1.3348
- 10 layers: 1.3348 -> 1.3314
- 11L INT6: 1.3314 -> 1.3161
- Orthogonal init: 1.3161 -> 1.3161
- Muon momentum 0.90: 1.3161 -> 1.3144
- Muon warmup 250: 1.3144 -> **1.3143**
- MLP3x + tied + SmearGate: 1.3143 -> **1.3071** (pre-roundtrip, 11L, over 16MB)
