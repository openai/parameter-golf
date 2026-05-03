# Curated supporting logs

Five logs that back the headline numbers in the parent `README.md`. The original screening / ablation batches contained ~100 logs across four Runpod sessions; these five are the ones a reviewer needs to verify the architectural claims.

| File | What it shows | Pointer in main README |
|---|---|---|
| `old_residual_240step_seed42.log` | 240-step seed-42 run with the **pre-UT7** recurrent-state-contraction residual. Same config as the delta run below except for `UT_RESIDUAL_DELTA=0`. Final `val_bpb=1.84038776`, `total=14,279,048` bytes. | `## How This Got Here` → "UT7: delta-residual fix" |
| `delta_residual_240step_seed42.log` | 240-step seed-42 run with `UT_RESIDUAL_DELTA=1, BRANCH_SCALE_INIT=0.6`. Same config, same seed, same data as the old-residual log. Final `val_bpb=1.58599604`, `total=14,506,646` bytes. **Direct A/B: 0.254 nat improvement from the residual change alone.** | `## How This Got Here` → "UT7: delta-residual fix"; `## Negative Results` → "Recurrent state contraction" |
| `rank256_noTTT_screen_60step.log` | 60-step master log of the rank-256 no-TTT screening from the 04-30 UT6 RLMA256 batch. Contains the screening that established a no-TTT baseline before pivoting to UT7. Final `val_bpb=2.11560405` for the first 60-step run (config does not yet have delta-residual). | `## How This Got Here` → "UT6 + RLMA256 (no TTT)" |
| `rank320_screen_seed42.log` | The 1×H100 screening run that selected `ADAPTER_RANK=320` for production. Architecture matches the submitted config exactly. Final `val_bpb=1.29703488`, `total=15,800,799` bytes — these match `train_seed42.log` in the parent folder, confirming the production seed run is reproducible from this screen. | `## Cap-Hardening Sweep` (calibration run) |
| `clip_sweep_seed314.log` | Independent quantization-clip sweep on **seed 314**, covering `clip_k ∈ {10, 11.5, 12.85, 14, 16}`. Confirms the seed-42 cap-hardening selection on a different seed: `clip_k=12.85` lands at `val_bpb=1.31170050`, `total=13,861,663` bytes (lower bytes than production because this screen uses a shorter schedule). | `## Cap-Hardening Sweep` |

All five logs were produced on Runpod 1×H100 (80 GB HBM3) instances with the same `train_gpt.py` shipped in the parent folder. Original Runpod batch directories (`runpod_ut_delta_20260430/`, `runpod_ut_long_20260430/`, `runpod_ut_next_20260430/`, `runpod_ut_sched_20260430/`) are kept locally but omitted from this PR for size; available on request.
