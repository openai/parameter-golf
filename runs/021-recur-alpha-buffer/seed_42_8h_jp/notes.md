# Run notes: 021-recur-alpha-buffer seed_42 (8×H100 JP)

**Pod:** b2bb9p8ux2fth7 (8×H100 SXM AP-JP-1, $23.92/hr)
**Commit:** cb5cd78 (recur_alpha as register_buffer frozen at 017 endpoint)
**Date:** 2026-04-21

## Relationship to earlier 4×H NE-1 run
The earlier partial run lives at `runs/021-recur-alpha-buffer/seed_42/`. This 8×H JP run is the authoritative full-pipeline result (training + TTT + GPTQ + serialize all landed, brotli installed before launch).

## Training outcome
- Stopped at step 4883 / 20000 (wallclock cap 596s = 10min)
- Loop activated at step 2156 (frac 0.350, earlier than Loop45's ~step 3500)
- Val@step 4000: val_bpb 1.1136
- Val@step 4883 (in-training, post-EMA): val_bpb 1.0701
- Pre-quant post-EMA val_bpb: **1.06963** (beats 019's 1.07063)
- Post-GPTQ (pre-TTT) val_bpb: **1.07913** (beats 019's 1.07989)
- **Post-TTT val_bpb: 1.06900** (loses to 019's 1.06744 and #1736's 1.06610)
- Submission size: 15.95 MB

## Throughput per-minute (pre-loop vs post-loop)
Pre-loop (steps ≤2156): ~8.13M tok/s — slightly faster than 008/017/019 (8.06–8.13M)
Post-loop: ramps down to 6.5M tok/s at step 4500, same profile as 017

021 was ~0.2-0.5m wallclock AHEAD of refs at every checkpoint due to faster pre-loop and slightly longer training window. Still hit stopping_early at step 4883 (55 more than 008, 186 more than 019).

## The scientific finding
**Buffer-α trades TTT headroom for pre-TTT val_bpb.** Frozen α means TTT's LoRA adapters can't modulate the α blending — they have to route all adaptation through weight correction. Literal/learned α stays flexible during TTT. Numeric evidence:

| metric | 019 (literal-α) | 021 (buffer-α) |
|---|---|---|
| pre-quant post-EMA val_bpb | 1.07063 | **1.06963** (better by 0.001) |
| TTT delta | −0.01245 | **−0.01013** (019 TTT is more effective) |
| post-TTT val_bpb | **1.06744** (better) | 1.06900 |

So 021 wins the *intermediate* milestone but loses the *endpoint*. Net: modest regression on the real metric.

## Throughput confirmation
0 Type B mystery spikes in this run (consistent with 020b + 021-4H-NE-1). Clean tok/s profile through all 4883 steps.

## TTT timing (new data)
- TTT_lora compile warmup: 133.6s
- Phase 1/3: 215s (111 chunks)
- Phase 2/3: 331s - 215s = 116s (185 chunks)
- Phase 3/3: ~139s (more chunks, finished at 470.7s total TTT time)
- Total TTT + compile: ~605s

## Costs
Total pod runtime: ~30 min × $23.92/hr = **~$12**. Plus ~$0.30 probe overhead from pod discovery. Total run cost ~$12.

## Environment / deps
- brotli installed via `pip install --break-system-packages brotli` in preflight
- No crash in serialize — full submission produced
- TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_021 (container-local, fine)

## Artifacts rsync'd
- train.log (41KB)
- diag_nvsmi.csv (1MB)
- final_model.int6.ptz (16MB — the submission)
- 37a90c58-*.txt (196KB — torchrun trace)
- final_model.pt NOT rsynced (130MB, intentionally skipped)
