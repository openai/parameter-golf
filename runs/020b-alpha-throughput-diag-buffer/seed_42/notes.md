# Run notes: 020b-alpha-throughput-diag-buffer seed_42

**Pod:** u6wzbaz86b7aw6 (NA 4×H100 SXM, /workspace)
**Commit:** 3cfc372
**Date:** 2026-04-21

## Training outcome
- Completed 2193 steps (wallclock cap on 4×H100)
- Loop activated at step 509 (`ENABLE_LOOPING_AT=0.17`)
- Val fired at step 1500 (`VAL_LOSS_EVERY=1500`) — val_bpb 1.1671
- Final pre-quant post-EMA val_bpb: 1.10598 (not comparable to #1736 due to 4×H100 step deficit)

## Post-training crash
GPTQ phase failed: `ModuleNotFoundError: No module named 'brotli'`
`final_model.pt` was written (130MB). No int6.ptz artifact. Not needed for this diagnostic run.

## Key diagnostic result
**Buffer-α hypothesis CONFIRMED.**

| metric | 020 (literal-α) | 020b (buffer-α) |
|---|---|---|
| Type B GPU mystery spikes | 12 | **0** |
| Post-val recompile cluster | N/A (no val fired) | **0** (post-val window = steady state) |
| Type A dataloader spikes | 16 | 12 (same pattern) |
| Post-loop median step time | 277.6ms | 292.8ms |

Post-val window (steps 1501-1700): mean 293.8ms vs steady 292.8ms — identical.
The 2 elevated steps in that window are Type A dataloader events (dl_us >40ms), not recompile events.

## TORCHINDUCTOR_CACHE_DIR note
Must be `/tmp/...` (container-local). NFS mount `/workspace` causes stale file handle crash.
Applied fix on this run: `TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_020b`.
