# BW XI — Hypothesis

## Parent
Bandit Wagon X 9F: `records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/`
- 1.13867894 int6_sw_bpb, 15,239,617 bytes, 110.19ms/step, 5446 steps

## Changes vs BWX

1. **Compression: zstd → brotli** (approved baseline change from BW20 gate)
   - Brotli quality=11 replaces zstd level=22
   - Expected: smaller artifact (~5-15% compression gain on dense weight blobs)
   - This frees headroom under the 16MB cap

2. **GPTQ enabled** (SKIP_GPTQ=0 → standard GPTQ 128×2048 calibration)
   - Confirmed signal: −0.002 int6_sw_bpb across BW12 and BW13
   - Adds ~700KB to artifact size (offset by brotli savings)
   - Post-training Hessian-aware quantization — no training change

## Why this combination
BWX at 15.24MB is tight on the 16MB cap. Brotli compression shrinks the
artifact enough to absorb GPTQ's size increase while still staying legal.
GPTQ has the most consistent signal of any post-training improvement tested
(−0.002 in every run).

## Expected outcome
- int6_sw_bpb: ~1.136-1.137 (−0.002 from GPTQ)
- Artifact: ~14-15MB (brotli savings minus GPTQ overhead)
- Step speed: identical (compression and GPTQ are post-training)

## Run
Full 8×H100 600s production run, seed=444, then seed=300 confirmation.
