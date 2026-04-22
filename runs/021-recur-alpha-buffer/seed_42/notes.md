# Run notes: 021-recur-alpha-buffer seed_42

**Pod:** 2seyx32eh5ggyv (4×H100 SXM US-NE-1)
**Commit:** cb5cd78 (recur_alpha as register_buffer frozen at 017 endpoint)
**Date:** 2026-04-21

## Hardware substitution
Spec called for 8×H100 JP. JP + NE-1 × H100 SXM/NVL, H200, A100 SXM were all dry for ~4 hours during launch window. Used 4×H100 NE-1 with `MAX_WALLCLOCK_SECONDS=1200` (doubled from default 600) to match total GPU-seconds of an 8×H100 run. All fraction-based schedules (`ENABLE_LOOPING_AT`, `warmdown_frac`) scale automatically with the doubled cap.

## Config deviation from spec
- `ENABLE_LOOPING_AT=0.35` (spec listed 0.17, corrected by user during launch to the code default). This made the test a cleaner "buffer-α only" variable instead of a combined "buffer-α + early-activation" test.
- All other env vars per spec 021.

## Training outcome
- Stopped at step 4736 / 20000 (wallclock cap 1196s)
- Loop activated at step 2112 (frac 0.350)
- Pre-quant post-EMA val_bpb: **1.07094776** (val_loss 2.34376)
- val_bpb@step_4000: 1.1086

## Key diagnostic result — BUFFER-α HYPOTHESIS CONFIRMED ON PRODUCTION
| metric | 019 (literal-α) | **021 (buffer-α)** |
|---|---|---|
| Type B mystery spikes | 12 | **0** |
| Post-loop tok/s stability | choppy 5.01–5.33M | flat 2.70M ±0.01 |
| Post-val recompile cluster | yes | no |

Per-100-step inst tok/s breakdown matches 020b's diagnostic profile exactly:
- Pre-loop: 3.97M ±0.02 (0.5% jitter)
- Post-loop steady: 2.70M ±0.01 (0.4% jitter)
- Only dips correspond to Type A dataloader shard-loads (every 127 steps) and val fire at step 4000

## Per-GPU throughput matches 008 exactly
- 008 per-GPU post-loop: 0.685M tok/s
- 021 per-GPU post-loop: 0.675M tok/s (1.5% diff, within noise)
No 4×H100 hardware overhead vs 8×H100.

## Step-matched val_bpb comparison
| step | 008 (8×H, #1736) | 019 (8×H, lit-α) | **021 (4×H, buf-α)** |
|---|---|---|---|
| 4000 | 1.1110 | 1.1071 | **1.1086** |
| ~4700-4800 final | 1.0697 @4828 | 1.0709 @4697 | **1.0714 @4736** |
| pre-quant post-EMA | 1.06922 | 1.07063 | **1.07095** |

021's pre-quant post-EMA is 0.00032 behind 019 and 0.00173 behind 008. At matched total GPU-seconds, 4×H100 pays a small batch-size tax (~0.001–0.002 bpb) for smaller effective batch per step. 021 essentially *matches* the 8×H baseline family on half the GPUs.

## Post-training crash (GPTQ → serialize → brotli)
Same brotli failure as 020b on NE-1 pod. Training, pre-quant post-EMA val eval, and GPTQ quantization all completed successfully; only the brotli compression step in `serialize()` crashed. Pre-quant number is the valid scientific signal. Post-TTT and post-GPTQ numbers not captured.

**Fix for next NE-1 run:** `pip install brotli` in preflight.

## TORCHINDUCTOR_CACHE_DIR note
`/tmp/torch_inductor_cache_021` used (container-local). Works fine on NE-1.

## Artifacts rsync'd
- `train.log` (22KB)
- `diag_nvsmi.csv` (529KB)
- `final_model.pt` (130MB)
- `7d2a2d89-96cb-4f37-b7ad-a9854bdd5db0.txt` (torchrun trace)

No `final_model.int6.ptz` — brotli crash prevented serialization.
