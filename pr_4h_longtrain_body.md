# [Non-Record] 6h Long-Train Scaling + TTT Hyperparameter Sweep

> **Current best 360-minute post-TTT BPB:** **1.03387** (`v7_noqv_rank96`, **single seed**, 4xH100 NVL)

## Summary

**Formal non-record submission** studying BPB as a function of training duration (10 min -> 6h) and systematically sweeping TTT/LoRA hyperparameters on the final 6h quantized artifact.

### At a glance

| Metric | Value | Notes |
|--------|-------|-------|
| Best 360-min post-TTT BPB | **1.03387340** | `v7_noqv_rank96` on the final 360-min artifact (**single seed**) |
| Matched 360-min pre-quant EMA BPB | **1.03340201** | eval-only follow-up from saved resume checkpoint |
| Matched 360-min quantized sliding BPB | **1.04273086** | same artifact, no TTT |
| 6h quantization tax | **+0.00932885 BPB** | quantized minus matched pre-quant EMA |
| Best TTT recovery at 6h | **0.00885746 BPB (~95%)** | `v7_noqv_rank96`; recovery fraction = 0.00885746 / 0.00932885 = 94.94% |
| Final artifact size | **15,926,271 bytes** | `final_model.int6.360min.ptz` |
| Run shape | **two RunPod sessions for the artifact path; third later pod for matched pre-quant recovery** | downloaded 300-min snapshot -> 4-GPU continuation -> later eval-only follow-up |

### Key findings

1. Post-TTT BPB improves from **1.06003** (10-min reference, PR #1934 3-seed mean) to **1.03387** (6h single-seed, `v7_noqv_rank96`). This is a descriptive endpoint comparison across durations and seeds, not a controlled scaling estimate.
2. A matched 360-min comparator gives **pre-quant EMA 1.03340201 -> quantized 1.04273086 -> post-TTT 1.03387340** (`v7`), so GPTQ adds **+0.00932885 BPB** at 6h and best TTT recovers **0.00885746 BPB** of that tax.
3. In this single-seed run, the best 6h post-TTT result remains only **+0.00047139 BPB** above the matched 6h pre-quant EMA.
4. Additional matched 240-min and 300-min controls show the same pattern: EMA helps, GPTQ adds a modest tax, and TTT recovers most or all of that tax.
5. Artifact size is effectively constant across this family of runs; quality improves more than bytes do.
6. **Removing Q and V LoRA targets** (`v7`: K+MLP+O+lm_head only) beats both the original full-target control (`v0`) and the lighter single-phase variant (`v12`).

## Acknowledged PR lineage for this stack

These are the PRs most directly responsible for the training recipe, optimizer substrate, continuation semantics, and TTT control/sweep used here.

| PR | Why it matters here |
|----|---------------------|
| **PR #1934** | Original record-track recipe that this long-train study extends in non-record form |
| **PR #1950** | Compliance-audited reproduction of PR #1934; exact base training recipe used here |
| **PR #1979** | 1-hour long-train precursor; provides the 60-min comparator and the original `v0_control_pr1979` TTT settings |
| **PR #461** | Original legal score-first TTT framework that all post-TTT comparisons here still follow |
| **PR #1767** | TTT alpha / warm-start / weight-decay improvements carried into the control TTT recipe |
| **PR #1855** | QK-gain and TTT-rank exploration that informed the long-train control and later sweep directions |
| **PR #1344** | Polar Express per-iteration Newton-Schulz coefficients concept for Muon |
| **PR #1787** | Parameter-golf integration of Polar Express Muon coefficients used by this training stack |

## Training scaling results

All durations use the same PR #1950 / PR #1934 recipe. To avoid mixing live-training metrics with matched eval-only comparators, the live checkpoint trajectory and the post-TTT horizon table are separated below.

| Duration | Source | Export / endpoint step | Live training val_bpb near export | Artifact |
|----------|--------|------------------------|-----------------------------------|----------|
| 60 min | PR #1979 | 16,001 | 1.0615 | 15,944 KB |
| 240 min | standalone 4h run | ~30K final stop/export | ~1.0600 live | 15,933 KB |
| **360 min** | resumed 6h chain | 49,765 | 1.0599* | **15,926 KB** |

*Last logged live validation BPB near the 360-min export; the matched 360-min EMA / quantized / post-TTT comparator chain is reported later.

### Live training trajectory around saved/exported checkpoints

This table reports the **last logged live training metrics near each saved/exported checkpoint**, not matched EMA/quantized/post-TTT evals. The 60/120/180/240 rows come from the standalone 4h run; the 300/360 rows come from the resume chain that produced the final 6h artifact.

| Checkpoint minute | Source run | Saved/exported step | Last logged train_loss near checkpoint | Last logged live val_loss | Last logged live val_bpb |
|-------------------|------------|---------------------|----------------------------------------|---------------------------|--------------------------|
| 60 | standalone 4h run | 10,488 | 2.4241 (step 10,000) | 2.5649 (step 8,000) | 1.1720 |
| 120 | standalone 4h run | 17,480 | 2.5575 (step 17,000) | 2.4924 (step 16,000) | 1.1389 |
| 180 | standalone 4h run | 23,418 | 2.4389 (step 23,000) | 2.4474 (step 20,000) | 1.1183 |
| 240 | standalone 4h run | ~30K final stop/export | 2.3156 (step 29,500) | 2.3199 (step 29,888) | 1.0600 |
| 300 | downloaded seed snapshot for continuation | 36,452 | 2.4071 (step 36,000) | 2.3792 (step 36,000) | 1.0871 |
| 360 | resumed 6h continuation | 49,765 | 2.2774 (step 48,000) | 2.3197 (step 48,000) | 1.0599 |

## How the 6h artifact and later follow-ups were actually produced

The **final 360-minute artifact itself** was produced in two RunPod sessions. A **third later pod** was used only for matched pre-quant follow-up recovery.

| Phase | Pod | Persistent checkpoint/export state | What it was used for |
|-------|-----|------------------------------------|----------------------|
| Initial live training run | `y3ulfm7pb5kqyt` | Downloaded `results/8h_longtrain_final/resume_snapshot_step_36452/` containing `resume_manifest.json` + `resume_rank{0..3}_step36452.pt`; manifest reports `step=36452`, `training_time_ms=18000630.06`, `world_size=4`, `exported_minutes=[60,120,180,240,300]` | Authoritative 300-minute restart point pulled back to HPC before the original pod expired |
| Resumed 6h-horizon continuation | `mu4c253h9yoiy3` | Wrote `results/resumed_6h_horizon_continuation_step36452/final_model.int6.360min.ptz` and `checkpoint_360min.json` (`train_steps=49765`, `train_wallclock_seconds=21600.15`, `artifact_bytes=15926271`); log also shows resume saves at 330 min (`step=43125`) and 360 min (`step=49765`) | Produced the 360-minute submission artifact and the original 6h post-TTT control result |
| Later pre-quant follow-up safety capture | `h2fkfy6usuw72n` | Downloaded `results/prequant_360min_from_step36452/resume_snapshot_step_43062/` with manifest + all 4 rank files; manifest reports `step=43062`, `training_time_ms=19800085.99`, `world_size=4` | Fallback 330-minute restart snapshot captured while recovering the matched 360-minute pre-quant EMA comparator stored in `results/prequant_360min_from_step36452/prequant_eval_summary.live.json` |

What was done, exactly:

1. The original 4-GPU live pod was allowed to run until a full 300-minute resume snapshot existed, then **all four rank-local checkpoint files plus the manifest** were downloaded to HPC under `results/8h_longtrain_final/resume_snapshot_step_36452/`.
2. The continuation resumed from that downloaded snapshot on **4 GPUs only**. The continuation log confirms `RESUME: restored step=36452, training_time=18000.6s, exported_minutes=[60, 120, 180, 240, 300]`.
3. The seed run was already a 6-hour training-wallclock run (`training_wallclock=21600` in `results/8h_longtrain_final/launcher_state.json`). The resumed pod used a longer hard stop than 6h, but explicitly kept `SCHEDULE_HORIZON_SECONDS=21600`, so LR warmdown and schedule-dependent behavior still followed the original 6-hour horizon. This is a **faithful continuation of the 6h schedule**, not a fresh longer-horizon rerun.
4. The submission artifact for this PR is the 360-minute export from the resumed pod: `results/resumed_6h_horizon_continuation_step36452/final_model.int6.360min.ptz`.
5. The later NCCL timeout in the continuation log happened **after** the 360-minute export and 360-minute resume save were written, so it does **not** invalidate the artifact used here.
6. The 330-minute step differs slightly between the main continuation (`43125`) and the later pre-quant follow-up snapshot (`43062`) because those are **different resumed pods** launched from the same 300-minute seed snapshot for different purposes.

## Post-TTT BPB over time

This table is the easiest way to see how the post-TTT endpoint moves with training duration. Only 240/300/360 have matched artifact/checkpoint controls in this session; 120 and 180 were not separately evaluated with TTT.

| Training horizon | Source / comparator | TTT config | post_ttt_bpb | Notes |
|------------------|---------------------|------------|--------------|-------|
| 10 min | PR #1934 reference | record submission config | 1.06003 | 3-seed mean reference point |
| 60 min | PR #1979 | original long-train control | 1.03988 | 8xH100, 60-min precursor |
| 240 min | matched 240-min artifact | `v0_control_pr1979` | 1.03539272 | nearly returns to matched 240-min pre-quant EMA (1.03545673) |
| 300 min | matched 300-min checkpoint | original control recipe | 1.04210727 | from resume-decomposition follow-up on the same saved checkpoint |
| 360 min | matched 360-min artifact | `v0_control_pr1979` | 1.03471322 | original 6h control used in the first sweep |
| 360 min | matched 360-min artifact | `v12_rank96_phase1_prefix1000` | 1.03421043 | single-phase / lower-global-compute variant |
| 360 min | matched 360-min artifact | `v7_noqv_rank96` | **1.03387340** | best result: Q/V LoRA removed, K+MLP+O+lm_head only |

## TTT/LoRA sweep on the 360-min quantized artifact

| Variant | LoRA rank/alpha | LR | Batch | post_ttt_bpb | Peak memory | Status |
|---------|------------------|----|-------|--------------|-------------|--------|
| `sliding_window_control` | — | — | — | 1.04273086 | 5.3 GB | baseline |
| `v0_control_pr1979` | 96 / 144 | 1e-4 | 64 | 1.03471322 | 47.8 GB | control |
| `v12_rank96_phase1_prefix1000` | 96 / 144 | 1e-4 | 64 | 1.03421043 | 47.7 GB | better than control |
| `v7_noqv_rank96` | 96 / 144 (K+MLP+O+lm_head only) | 1e-4 | 64 | **1.03387340** | 43.6 GB | **best** |
| `v1_rank128_alpha192` | 128 / 192 | 1e-4 | 64 | 1.03877 | — | worse |
| `v2_rank128_lr3e4` | 128 / 192 | 3e-4 | 64 | 1.09049 | — | regression |
| `v3_local_batch_chunk` | 128 / 192 | 3e-4 | 128 | — | — | failed (no clean traceback; likely memory pressure / unstable config) |
| `v4_global2_largechunk` | 128 / 192 | 3e-4 | 128 | — | — | failed (no clean traceback; likely memory pressure / unstable config) |
| `v5_prefix3000` | 128 / 192 | 3e-4 | 128 | — | — | failed (no clean traceback; likely memory pressure / unstable config) |
| `v6_prefix3000_phase4_optional` | 128 / 192 | 3e-4 | 128 | — | — | failed (no clean traceback; likely memory pressure / unstable config) |

Interpretation:

- The sliding-window control isolates the TTT contribution on the same 360-minute artifact.
- `v7` improves on the control while using **~4.2 GiB less peak memory** than the full-target `v0` recipe.
- `v12` is interesting because it nearly matches the original 3-phase control while using much less global-TTT compute.

## Matched decomposition and comparator chain

| Stage | BPB | Delta |
|-------|-----|-------|
| Matched 6h pre-quant EMA | 1.03340201 | baseline |
| Quantized 6h artifact (sliding eval) | 1.04273086 | +0.00932885 vs matched pre-quant EMA |
| Post-TTT (`v0_control_pr1979`) | 1.03471322 | -0.00801764 vs quantized, +0.00131121 vs matched pre-quant EMA |
| Post-TTT (`v7_noqv_rank96`) | **1.03387340** | **-0.00885746 vs quantized, +0.00047139 vs matched pre-quant EMA** |

Additional matched controls:

- **240 min:** pre-quant EMA **1.03545673** -> quantized **1.04485881** (+0.00940208 tax) -> post-TTT **1.03539272**
- **300 min:** live **1.08215117** -> EMA **1.04945326** -> quantized **1.05603004** (+0.00657678 tax) -> post-TTT **1.04210727**
- **360 min:** the original control (`v0`) reaches **1.03471322**, while the later Q/V-ablation follow-up (`v7`) improves further to **1.03387340**

## Scientific hypotheses tested

1. **H1: Longer training improves post-TTT BPB** -> supported descriptively
2. **H2: Longer training meaningfully reduces compressed artifact size** -> not supported
3. **H3: Higher LoRA rank improves TTT on this 6h artifact** -> not supported
4. **H4: Higher LR improves TTT at rank 128** -> rejected
5. **H5: Larger local batch / chunk improves TTT** -> untested because those variants failed
6. **H6: GPTQ degrades BPB on matched checkpoints** -> supported at 240, 300, and 360 minutes
7. **H7: Q/V LoRA targets are necessary for best 6h TTT** -> rejected by `v7_noqv_rank96`

## Infrastructure additions used by this PR

- Resumable rank-local checkpoints with manifest-driven restore
- `SCHEDULE_HORIZON_SECONDS` to decouple stop horizon from LR / schedule horizon during continuation
- `sweep-only-artifact` mode for standalone TTT evaluation on an existing quantized artifact
- HTTP-based artifact upload/download around RunPod proxy instability
- Per-variant isolated TTT sweep execution with JSON / CSV summaries

## Compliance

- NOT record-track compliant (training exceeds the 600s wallclock budget)
- Training recipe intentionally held fixed relative to PR #1950 / PR #1934
- Score-first TTT retained; no validation tokens are seen before scoring
- Artifact remains under the 16 MB limit
- TTT/LoRA is RAM-only at eval time and does not alter the serialized artifact

## Hardware and cost

| Phase | Hardware | Notes |
|-------|----------|-------|
| 1h precursor | 8xH100 SXM | PR #1979 baseline |
| 4h standalone run | 4xH100 NVL | 60/120/180/240 checkpoint study |
| 6h continuation | 4xH100 NVL | downloaded 300-min snapshot -> 360-min resumed artifact |
| TTT sweep + follow-ups | 4xH100 NVL | 240-min TTT-only, 300-min decomposition, 360-min pre-quant recovery, v7/v12 follow-up sweep |

Estimated total cost across the long-train stack and follow-ups is on the order of **~$160**.
