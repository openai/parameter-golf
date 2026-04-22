# Spec 029 mini — 2×H100 JP, seed 314

**Pod:** `3yquayffufrgjd` @ AP-JP-1, 2×H100 SXM, $5.98/hr, image `runpod/parameter-golf:latest`
**Commit:** `c3a99b3` (exp/029-full-stack) — 025b frozen + LoRA warm-start-A + depth curriculum + alpha_info fix + pre-warm
**Seed:** 314
**Launched:** 2026-04-23 16:20 UTC
**Stopped (training):** 16:37 UTC — `stopping_early: wallclock_cap` at step 674, train_time 396.6s (MAX_WALLCLOCK=400s)
**Pod stopped by user:** 16:40 UTC after confirming GPTQ val_bpb not needed
**Total pod wall:** ~20 min, ~$2

## Purpose

Validate three things before committing to 4×H / 8×H cost:
1. Depth curriculum markers actually fire (`loop_warmup:depth_upgraded`, `layer_loop:enabled`, `loop_depth:upgraded`)
2. Pre-warm in `c3a99b3` eliminates mid-run recompiles
3. No NaN / silent divergence on the full-stack config

## Result: VALIDATION PASSED ✅

All spec-required markers fired at the right places:

| marker | expected | observed |
|---|---|---|
| `loop_warmup:depth_upgraded looping_depth:4` | pre-step-0 (during startup warmup) | ✅ emitted after `loop_warmup_step: 20/20` |
| `layer_loop:enabled` | step ≈ 100 (10% of train time) | ✅ `step:108 frac:0.101 depth:3` |
| `loop_depth:upgraded` | step ≈ 200 (20%) | ✅ `step:180 frac:0.201 depth:4` |
| NaN / inf in train_loss | never | ✅ none observed |
| Recompile count post-step-0 | 0 (pre-warm hypothesis) | ✅ stable at 32 — all pre-step-0 |

Key: **pre-warm works as designed**. Recompile counter climbed during startup (2→32) across three depth states × four cu_seqlens buckets, then flatlined once training started. `TORCH_LOGS=recompiles` confirms zero recompiles between step 0 and step 674.

## Training trajectory (informational; 400s is too short for meaningful bpb)

```
step    train_loss   tok/s     train_time
100     3.5255       2.14M     0.6m   (depth=1)
200     3.1982       1.70M     1.5m   ← layer_loop:enabled at 108, depth=3
300     2.8996       1.51M     2.6m
400     2.7026       1.43M     3.7m   ← loop_depth:upgraded at 180, depth=4
500     2.5828       1.38M     4.8m
600     2.6250       1.35M     5.8m
674     stopped (wallclock cap)
```

Loss trajectory is healthy — characteristic drop after each phase transition. Tok/s falls as deeper loops activate (2.14M → 1.35M), which is expected on 2×H with NUM_LOOPS=3 and depth up to 4.

Final val@674 = 1.1959 (meaningless at this step count).
Pre-quant EMA val_bpb = 1.29251 (also meaningless — undertrained).

## Incidents

1. **Pre-compile was slow** — ~15 min from torchrun start to first `warmup_step:` line. 12+ Dynamo graphs to compile (3 depth states × 4 cu_seqlens buckets). Acceptable once; mitigation for future repeat runs on same commit would be persisting `TORCHINDUCTOR_CACHE_DIR` to `/workspace/.torch_inductor_cache_029` (deferred — spec didn't request it).

2. **JP volume mount** — pod mounted volume at `/workspace`, not `/runpod` (contra prior memory). Fixed with `ln -sfn /workspace /runpod`. Memory `reference_jp_volume_mount.md` updated.

## Artifacts

- `train.log` — full stdout (61 KB)
- `ae1a25cd-…txt` — rank 0 log
- `launch.out` — empty (setsid detached cleanly)

## Decision

Validation passed → **advance to 4×H screen** (already launched in parallel on NE-1 pod `tbk3t7a9o5tifp`).
