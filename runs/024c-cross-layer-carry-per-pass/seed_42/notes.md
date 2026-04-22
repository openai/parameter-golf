# 024c Execution Notes

## Launch
- Pod: ekdxerdvv8ibde (NA-1, 4×H100)
- Commit: ba957e1 on exp/recur-alpha-buffer
- Launched after 024b completed
- DATA_DIR corrected to /workspace/parameter-golf/data (same fix as 024b)
- PHASED_TTT_ENABLED=0, TTT_ENABLED=0 for screen mode

## Throughput
- Pre-loop: same as 024b — +43K tok/s vs 021c
- Loop activated at step 2257 (within [2000,2400] ✓)
- Post-loop decline: nearly identical to 024b — 4.11M → 3.31M by step 5000
- The extra 12 params (24 vs 12 total) added zero throughput cost

## Alpha/beta trajectory
- Init: beta=[[1,1,1],[1,1,1]], alpha=[[[0×9],[0×9]]] ✓
- First nonzero (step ~2300): all values ~0.17–0.19, symmetric across passes
- Rapid divergence by step 2500 — passes learning opposite strategies:
  - Pass 2: beta=[0.73, 1.27, 1.57] — L3 attenuated; alpha[L5]=[+0.29, +0.23, −0.50] — strong self-subtraction
  - Pass 3: beta=[1.52, 1.40, 1.11] — all boosted; alpha more uniformly positive
- Beta fully converged by step ~3500: pass2=[0.945, 1.297, 1.625], pass3=[1.984, 1.688, 1.156]
- Alpha still slowly drifting at end (grad_norm ~0.005–0.009) but structure stable
- Final pass2 alpha[L5]=[+0.001, +0.279, −0.326] — nearly zeroed out own carry, relies on L4
- Final pass3 alpha pattern: generally smaller magnitudes, more distributed

## Results
- Steps: 5030 ✓ (≥5000)
- val@4000: 1.1205 (vs 024b: 1.1196) — slightly worse
- val@final: 1.0704 (vs 024b: 1.0699) — slightly worse
- pre-quant EMA: 1.06999 — behind 024b (1.06960) by 0.00039
- quantized: 1.07952 — behind 024b (1.07907) by 0.00045
- Wall time: ~40 min pod total
- Cost: ~$4

## Key finding
Per-pass parameterization learns genuinely different strategies per pass (pass 2 = aggressive negation, pass 3 = gentle positive blend), but this expressiveness did not translate to better EMA quality vs shared alpha (024b). 024b's simpler shared weights appear to regularize better at this scale.

## Anomalies
None — clean run throughout. Beta converged faster than alpha. The dramatic per-pass divergence is real and interesting but doesn't help quality at 4×H100 screen scale.
