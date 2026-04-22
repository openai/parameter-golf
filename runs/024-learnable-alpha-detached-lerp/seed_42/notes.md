# Spec 024 — seed 42 execution notes

## Summary

Detached-lerp blend + init-ones α. Run finished cleanly (DONE marker). Screen mode (TTT_ENABLED=0 + PHASED_TTT_ENABLED=0); EMA and GPTQ ran by default.

**Final step: 4975** (vs 021c 5004, −29 steps → ~0.6% residual throughput overhead).
**val_bpb@4000: 1.1185** (vs 021c 1.1177, Δ +0.0008).
**Pre-quant post-EMA val_bpb: 1.07106** (vs 021c 1.06952, Δ +0.00154).

## Deviations from spec

| item | spec | actual | reason |
|---|---|---|---|
| DATA_DIR | `/workspace/data` | `/workspace/parameter-golf/data` | spec path doesn't exist on current volume state |
| MAX_WALLCLOCK_SECONDS | unset | 1200 | needed to match "4H wallclock" convention (021c ref) |
| PHASED_TTT_ENABLED | 1 | 0 | screen-mode, per user's interview choice |
| TTT_ENABLED | default 1 | 0 | screen-mode, per user's interview choice |

## Throughput timeline (tok/s vs 021c_variant_4h_ne1)

Cumulative tok/s (code reports cumulative avg, not interval — per user memory, post-activation interval rate is noise here):

| step | 021c | 024 | Δ | |
|---|---|---|---|---|
| 500 | 4.197M | 4.210M | +13K | pre-activation |
| 2000 | 4.199M | 4.207M | +8K | pre-activation |
| 2300 | 4.142M | 4.153M | +10K | loop activated ~2250 |
| 3000 | 3.684M | 3.720M | +36K | early post-activation |
| 4000 | 3.427M | 3.407M | -20K | mid-training |
| 4900 | 3.301M | 3.279M | -22K | end |

Cumulative slip widens as cumulative avg pulls down the rate. Raw step-rate behavior looks clean.

## α values

- Init: all 1.0 (lerp identity = full transformation)
- First non-trivial grad at step 2250 (loop activation)
- Converged pattern around step 3500:
  - pass 1 (enc, layers 3/4/5): **[~1.18, ~1.24, ~1.41]** — amplifying, strongest at L5
  - pass 2 (dec, layers 3/4/5): **[~1.03, ~0.95, ~0.85]** — near-identity, damp at L5
- grad_norm steady around 0.0025–0.005 after convergence

Pattern is consistent with prior learnable-α runs' direction (pass 1 amps, pass 2 trims).

## Training loss vs 021c — matched-step Δ (selected)

| step | 021c | 024 | Δ |
|---|---|---|---|
| 500 | 2.6678 | 2.6732 | +0.0054 |
| 1000 | 2.7722 | 2.7752 | +0.0030 |
| 2000 | 2.6600 | 2.6601 | +0.0001 |
| 3000 | 2.5783 | 2.5819 | +0.0036 |
| 4000 | 2.3989 | 2.3986 | -0.0003 |
| 4900 | 2.3340 | 2.3332 | -0.0008 |

All within ±0.006 — noise-level.

## Artifacts on /workspace (NA-1, not git)

- `/workspace/runs/024-learnable-alpha-detached-lerp/seed_42/final_model.pt` (135MB, post-EMA pre-GPTQ)
- `/workspace/runs/024-learnable-alpha-detached-lerp/seed_42/final_model.int6.ptz` (15.9MB, GPTQ int6 quantized submission)

## Anomalies

- **Pod container exited on its own** when training script finished (template has no supervisord daemon). SSH became unreachable. Had to provision a 1×H100 pod on the same volume to rsync artifacts. ~$0.20 churn cost. Not a logic bug; just how this pod image behaves.
- 4×H100 **unavailable in JP** at launch time — fell back to NA-1 per user choice.

## Cost

- Main run pod (4×H100 NA, $11.96/hr): ~29 min = $5.78
- Rsync pod (1×H100 NA, $2.99/hr): ~2 min = $0.10
- Total: ~$5.88

## Handback

Spec 024's accept decision tree bucket: **4900–4999 = "Partial recovery — investigate before promoting"**. Research session to decide promote/iterate/kill.
