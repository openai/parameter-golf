# Spec 026 Execution Notes

## Launch
- Pod: jqnj0zret8c13u (AP-JP-1, 8×H100, $23.92/hr)
- Commit: 950af24 on exp/recur-alpha-buffer (025b frozen cross-layer carry)
- Volume mounts at /workspace (not /runpod — verified on pod)
- DATA_DIR: /workspace/data (JP path, confirmed correct)
- TORCHINDUCTOR_CACHE_DIR: /tmp/torch_inductor_cache_026_8h_jp (on /tmp, not NFS)
- PHASED_TTT_ENABLED=1, PHASED_TTT_PREFIX_DOCS=2000, PHASED_TTT_NUM_PHASES=3
- RECUR_ALPHA_ENABLED=1 (frozen constants, requires_grad=False)

## Frozen alpha/beta verification
- beta=[1.59375, 1.8828125, 1.9921875] — 024b converged values baked in
- alpha[L3]=[+0.252, -0.021, -0.012], alpha[L4]=[+0.067, -0.348, +0.003], alpha[L5]=[+0.139, +0.241, +0.027]
- grad_norm=0.000000 throughout entire run — confirmed requires_grad=False
- Values bit-identical at every logged step

## Loop activation
- Activated at step 2157 (frac=0.350) ✓ within [2000, 2300] window
- encoder:[0,1,2,3,4,5,3,4] decoder:[5,3,4,5,6,7,8,9,10]

## Throughput
- Pre-loop: ~8.14M tok/s (vs 021e: ~8.14M — identical, frozen adds zero overhead)
- Post-loop: declining cumulative average, 7.64M at step 2500 → 6.43M at step 4800
- Same trajectory as 021e — frozen carry dict same cost as learnable

## Train loss vs 021e
| step | 026 | 021e | Δ |
|------|-----|------|---|
| 100 | 3.6458 | 3.6590 | -0.013 |
| 500 | 2.5881 | 2.5837 | +0.004 |
| 1000 | 2.8202 | 2.8195 | +0.001 |
| 2000 | 2.6719 | 2.6754 | -0.004 |
| 2500 | 2.5588 | 2.5641 | -0.005 |
| 3000 | 2.5667 | 2.5711 | -0.004 |
| 4000 | 2.4138 | 2.4146 | -0.001 |

Consistent small advantage post-loop for 026.

## Results
- Training stopped: step 4863 (internal 600s wallclock cap)
- val@4000: 1.1128 (vs 021e 1.1134, Δ=-0.0006)
- val@final: 1.0695
- pre-quant EMA: 1.06893 (vs 021e 1.06944, Δ=-0.00051)
- quantized: 1.07827
- **post-TTT: 1.06582** (vs 021e 1.06622, Δ=-0.00040; vs SOTA #1736 1.06610, Δ=-0.00028)
- TTT: 3 phases, boundaries [666, 1333, 2000] docs, eval_time=470s
- Wall time: ~30 min, cost: ~$12

## Verdict
Beats 021e by 0.00040 and beats SOTA #1736 by 0.00028. Falls in [1.062, 1.066] bucket → 3-seed confirmation required (seeds 43/44 on same pod).

## Anomalies
None — clean run throughout. Frozen routing hypothesis confirmed: baking in 024b's converged alpha/beta eliminates discovery cost and delivers a consistent training advantage over learnable alpha (021e) at matched steps.

## Artifacts
- train.log — complete ✓
- diag_nvsmi.csv — complete ✓
- final_model.int6.ptz — rsynced to local ✓ (15.9MB)
- final_model.pt — on pod container disk (stopped, not deleted); 135MB
