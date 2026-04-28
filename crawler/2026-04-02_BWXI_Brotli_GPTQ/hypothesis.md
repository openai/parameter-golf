# BW XI — Hypothesis (Best-Foot-Forward)

## Parent
Bandit Wagon X 9F: `records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/`
- 1.13867894 int6_sw_bpb, 15,239,617 bytes, 110.19ms/step, 5446 steps

## Changes vs BWX (5 stacked signals)

### 1. Compression: zstd → brotli (approved baseline)
- Brotli quality=11 replaces zstd level=22
- BW20 gate: clean run, no blowups
- Expected: ~5-15% smaller artifact, frees 16MB headroom

### 2. Loop-aware GPTQ (LOOP_AWARE_GPTQ=1, SKIP_GPTQ=0)
- 2-phase Hessian calibration: phase 1 collects all layers, phase 2 re-collects
  crawler Hessians after flat layers are quantized
- BW10 full run: −0.00380 BPB vs BW5 champion
- MUST be loop-aware, not standard — shared weights are hostile to naive quant
  (Frugendorff catastrophe: 1.38 → 5.7 BPB with naive quant)
- BW12/BW13: consistent −0.002 signal at gate scale

### 3. QK_GAIN_INIT=4.0 (sharper initial attention)
- Per-head q_gain scalar initialized at 4.0 (default 1.5)
- Neural 2k proxy: −0.00149 BPB + ~1.05MB compression gain
- External signal: ~−0.006 BPP across 45 runs, 3 codebases
- Zero code change, just env var. Model free to train away from init.
- HIGH-CONFIDENCE but untested on crawler — first crawler test here

### 4. CRAWLER_LOOPS=2 (down from 3)
- BW17 DGX-Spark RAPID: −0.054 int6_sw_bpb (directional, small-token run)
- Fewer loops = faster steps = more steps in 600s budget
- Fewer loops = smaller quant gap (less shared-weight amplification)
- DIRECTIONAL signal — first full-scale test here

### 5. Warmdown stays at 2000 (confirmed best)
- Research synthesis: 2000 > 3500 > 5000 (Rat Rod warmdown study)
- Already in BWX, keeping it

## Architecture summary
- 9F flat layers, 1 crawler layer, **2 loops** (was 3)
- Tap-off (TAP_DIM=0) — BW12 showed tap-off beats tap-on by −0.002
- No anchor — BW13 showed anchor regresses on tap-off
- INST_DIM=32, MODEL_DIM=512
- QK_GAIN_INIT=4.0
- Loop-aware GPTQ post-training (128 samples × 2048 seq)
- Brotli quality=11 compression
- COMPILE_FULLGRAPH=1

## Expected outcome
- int6_sw_bpb: ~1.132-1.136 (GPTQ −0.002, QK4 −0.001 to −0.006, loops directional)
- Artifact: ~13-15MB (brotli savings offset GPTQ overhead)
- Step speed: faster than BWX (fewer loops = less compute per step)
- More total steps in 600s budget

## Risk factors
- QK4 untested on crawler — could be null
- Loops=2 unconfirmed at full scale — RAPID signal was directional only
- GPTQ + brotli + 9F size interaction untested at full scale
