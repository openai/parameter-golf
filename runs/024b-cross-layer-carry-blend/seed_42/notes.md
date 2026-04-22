# 024b Execution Notes

## Launch
- Pod: 88u8sha9h1cify (NA-1, 4×H100)
- Commit: 89367c5 on exp/recur-alpha-buffer
- Launched parallel to 024 (detached-lerp)
- DATA_DIR corrected to /workspace/parameter-golf/data (spec said /workspace/data which doesn't exist)
- PHASED_TTT_ENABLED=0, TTT_ENABLED=0 for screen mode (spec had TTT enabled — disabled for apples-to-apples vs 024)

## Throughput
- Pre-loop (~0–2256): +38–51K tok/s faster than 021c (4.24M vs 4.20M)
- Loop activated at step 2256 (within [2000,2400] window ✓)
- Post-loop: cumulative tok/s declined steadily from 4.11M → 3.30M by step 5000
- Interval rate at end ~2.7M tok/s — carry dict overhead is real but step count still ≥5000

## Alpha/beta trajectory
- Init: beta=[1,1,1], alpha=[[0,0,0]×3] ✓ — identical to baseline
- First nonzero (step ~2300): all alpha positive ~0.17, symmetric
- By step 3000: diagonal dominance emerged — each layer mostly uses its own carry
- Converged by ~step 4000. Final values (step 5000):
  - beta=[1.60, 1.88, 1.99] — all well above 0.1 (no collapse)
  - alpha[L3]=[+0.252, −0.021, −0.012] — own-carry dominant, slight neg cross
  - alpha[L4]=[+0.067, −0.348, +0.003] — strongly subtracts own carry (residual correction)
  - alpha[L5]=[+0.139, +0.241, +0.027] — uses both L3 and L4 carries
- grad_norm at convergence: ~0.003–0.009

## Results
- Steps: 5017 ✓ (≥5000)
- val@4000: 1.1196 (vs 024: 1.1185, vs 021c: 1.1177) — slight regression
- pre-quant EMA: 1.06960 — beats 024 (1.07106) by 0.00146
- quantized: 1.07907 — beats 024 (1.08057) by 0.00150
- Wall time: ~40 min pod total (~20 min training)
- Cost: ~$4

## Anomalies
- Cumulative tok/s declined continuously post-loop. The carry dict (storing 3 layer outputs) adds persistent memory/compute overhead. Step count still made threshold.
- val@4000 slightly worse than 024, but final EMA better — EMA averaging over more diverse trajectory helped.
