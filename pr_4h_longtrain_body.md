# [Non-Record] 6h Long-Train Scaling + TTT Hyperparameter Sweep  val_bpb=1.03387

## Summary

**Non-record experiment** studying BPB as a function of training duration (10 min → 6h) and systematically sweeping TTT/LoRA hyperparameters on the final 6h quantized artifact.

**Key findings:**
1. Post-TTT BPB improves from 1.060 (10 min, 3-seed mean) to **1.03387** (6h single-seed, post-TTT v7 no-Q/V ablation) — note: this is a descriptive endpoint comparison across different durations/seeds, not a controlled scaling estimate
2. A matched 360min comparator gives **pre-quant EMA 1.03340201 -> quantized 1.04273086 -> post-TTT 1.03387** (v7), so GPTQ adds **+0.00932885 BPB** at 6h and best TTT recovers **0.00886 BPB** of that tax
3. In this single-seed run, the final 6h post-TTT result (v7) remains only **+0.00047 BPB** above the matched 6h pre-quant EMA (TTT recovers ~95% of the 6h quantization tax; v0 control recovers ~86%)
4. Additional matched 240min and 300min controls show the same pattern: pre-quant EMA 1.03546→quant 1.04486→post-TTT 1.03539 (240min); live 1.08215→EMA 1.04945→quant 1.05603→post-TTT 1.04211 (300min)
5. Artifact size is effectively constant (15,926–15,953 KB across all durations)
6. **Removing Q and V LoRA** targets (v7: K+MLP+O+lm_head only) with 3-phase TTT achieves **1.03387 BPB** — beating both the v0 full-target control (1.03471) and the single-phase v12 variant (1.03421)
7. At rank 128/alpha 192, raising LR from 1e-4 to 3e-4 worsened BPB by ~0.052
8. Single-phase TTT with only 1000 prefix docs (v12) gives **1.03421** — nearly matching the full 3-phase 2000-prefix control with substantially less global-SGD compute (1 phase × 1000 docs vs 3 phases × 2000 docs)

### Training Scaling Results

All durations use the identical PR #1950/1934 recipe. Metrics differ by evaluation stage:

| Duration | Steps | training_val_bpb | quantized_bpb | post_ttt_bpb | Artifact |
|----------|-------|-----------------|---------------|-------------|----------|
| 10 min (3-seed, PR #1934) | ~4000 | — | — | 1.06003† | 15,953 KB |
| 60 min (8×H100, PR #1979) | ~8000 | 1.0615 | — | 1.03988 | 15,944 KB |
| 240 min (4×H100) | ~30K | — | 1.0449 | — | 15,933 KB |
| **360 min (4×H100)** | 49765 | 1.0599* | **1.04273** | **1.03387** | **15,926 KB** |

*training_val_bpb at step ~48000 (last logged); †3-seed mean from record submission  
TTT gain = quantized_bpb - post_ttt_bpb = 1.04273 - 1.03387 = **0.00886 BPB** (v7 no-Q/V ablation)

### How the 6h artifact was actually produced (two RunPod sessions)

The 360-minute artifact in this PR was **not** trained in one uninterrupted pod. We split the run across two 4xH100 NVL RunPod sessions using the repo's manifest-driven rank-local resume path, then ran later eval-only follow-ups from the saved snapshots.

| Phase | Pod | Persistent checkpoint/export state | What it was used for |
|-------|-----|------------------------------------|----------------------|
| Initial live training run | `y3ulfm7pb5kqyt` | Downloaded `results/8h_longtrain_final/resume_snapshot_step_36452/` containing `resume_manifest.json` + `resume_rank{0..3}_step36452.pt`; manifest reports `step=36452`, `training_time_ms=18000630.06`, `world_size=4`, `exported_minutes=[60,120,180,240,300]` | This is the authoritative 300-minute restart point that was pulled back to HPC before the original pod expired |
| Resumed 6h-horizon continuation | `mu4c253h9yoiy3` | Wrote `results/resumed_6h_horizon_continuation_step36452/final_model.int6.360min.ptz` and `checkpoint_360min.json` (`train_steps=49765`, `train_wallclock_seconds=21600.15`, `artifact_bytes=15926271`); pod log also shows resume saves at 330 min (`step=43125`) and 360 min (`step=49765`) | This continuation produced the 360-minute submission artifact and the original 6h post-TTT result |
| Later pre-quant follow-up safety capture | `h2fkfy6usuw72n` | Downloaded `results/prequant_360min_from_step36452/resume_snapshot_step_43062/` with manifest + all 4 rank files; manifest reports `step=43062`, `training_time_ms=19800085.99`, `world_size=4` | This was a fallback 330-minute restart snapshot captured while recovering the matched 360-minute pre-quant EMA comparator |

What was done, exactly:

1. We first let the original 4-GPU live pod run until a full 300-minute resume snapshot existed, then downloaded **all four rank-local checkpoint files plus the manifest** to HPC under `results/8h_longtrain_final/resume_snapshot_step_36452/`.
2. We resumed from that downloaded snapshot on **4 GPUs only**. The continuation log confirms `RESUME: restored step=36452, training_time=18000.6s, exported_minutes=[60, 120, 180, 240, 300]`.
3. The resumed pod used a longer hard stop than 6h, but kept `SCHEDULE_HORIZON_SECONDS=21600`, so LR warmdown / schedule-dependent behavior still followed the original 6-hour horizon. This is a **faithful continuation of the 6h schedule**, not a fresh longer-horizon rerun.
4. The submission artifact for this PR is the 360-minute export from the resumed pod: `results/resumed_6h_horizon_continuation_step36452/final_model.int6.360min.ptz`. That export completed before the later NCCL timeout seen in the continuation log, so the timeout does **not** invalidate the 360-minute artifact used here.
5. The 330-minute step differs slightly between the main continuation (`43125`) and the later pre-quant follow-up snapshot (`43062`) because those are **different resumed pods** launched from the same 300-minute seed snapshot for different purposes.
6. Reproducibility note: to reproduce this PR artifact exactly, reproduce the **same two-stage chain** — initial run to the downloaded 300-minute snapshot, then a 4-GPU continuation from `resume_snapshot_step_36452` with schedule horizon fixed at 21600s. An uninterrupted single-pod 360-minute run is **not** what generated the artifact in this PR.

### TTT/LoRA Sweep (on 360-min quantized artifact)

| Variant | LoRA Rank/Alpha | LR | Batch | post_ttt_bpb | Peak Memory | Status |
|---------|----------------|------|-------|-------------|-------------|--------|
| **sliding_window_control** | — | — | — | 1.04273 | 5.3 GB | ✓ baseline |
| v0_control (PR #1979) | 96/144 | 1e-4 | 64 | 1.03471 | 47.8 GB | ✓ |
| **v7_noqv_rank96** | 96/144 (K+MLP+O+lm_head) | 1e-4 | 64 | **1.03387** | 43.6 GB | ✓ **best** |
| v12_rank96_phase1_prefix1000 | 96/144 | 1e-4 | 64 | 1.03421 | 47.7 GB | ✓ |
| v1_rank128_alpha192 | 128/192 | 1e-4 | 64 | 1.03877 | — | ✓ |
| v2_rank128_lr3e4 | 128/192 | 3e-4 | 64 | 1.09049 | — | ✓ regression |
| v3_local_batch_chunk | 128/192 | 3e-4 | 128 | — | — | failed |
| v4_global2_largechunk | 128/192 | 3e-4 | 128 | — | — | failed |
| v5_prefix3000 | 128/192 | 3e-4 | 128 | — | — | failed |
| v6_prefix3000_phase4 | 128/192 | 3e-4 | 128 | — | — | failed (optional) |

The sliding_window_control runs quantized model evaluation with no TTT adaptation, providing the proper baseline to isolate the TTT contribution (0.00886 BPB gain with best variant v7).

**Q/V LoRA ablation insight:** Removing Q and V LoRA targets reduces peak memory by 4.2 GiB (43.6 vs 47.8 GB) and achieves a lower BPB (1.03387 vs 1.03471). This is consistent with the possibility that Q/V LoRA introduces optimization interference or excess capacity when applied alongside K+MLP+O, though the single-seed result is not diagnostic of the mechanism. A single-phase variant (v12) with only 1000 prefix docs also slightly beats the 3-phase control despite substantially less global-SGD compute.

## ML Changes from Reference

**Training:** No training-side ML change from PR #1950/1934. The resumed 6h run keeps the same architecture, optimizer, loss, tokenizer/data setup, and schedule semantics; the code additions are infrastructural (resume checkpoints, periodic exports, schedule-horizon continuation).

**Eval-only adaptation changes:** Relative to the PR #1979 control, this submission also runs a RAM-only TTT/LoRA sweep over adaptation hyperparameters including LoRA rank/alpha, LoRA LR, local batch/chunk size, global TTT epochs/chunk tokens/batch seqs/warmup, phased prefix/phase count, and **LoRA target selection** (Q/V ablation). These changes affect only evaluation, never the serialized 16 MB artifact. The best result came from **v7** — removing Q and V LoRA targets while keeping K+MLP+O+lm_head (3 phases, 2000 prefix docs), achieving **1.03387** BPB.

## Background & Related PRs

- **PR #1979** — Our 1h long-train study (post-TTT BPB 1.0399, 8×H100)
- **PR #1950** — Compliance-audited reproduction of PR #1934 (base recipe)
- **PR #1934** — Record-track 3-seed submission (post-TTT val_bpb 1.06003)
- **PR #461** — Original score-first legal TTT framework
- **PR #1767** — TTT alpha/warm-start/weight-decay improvements
- **PR #1855** — QK_GAIN_INIT=6.0 + TTT_LORA_RANK exploration

## Scientific Hypotheses Tested

1. **H1: Longer training improves post-TTT BPB** ✅ Confirmed (1.060 → 1.035 over 6h; note: 10-min is 3-seed mean, 6h is seed-42 only)
2. **H2: Longer training reduces artifact size** ❌ Rejected (−27 KB ≈ 0.17%)
3. **H3: Higher LoRA rank improves TTT** ❌ Not supported (+0.004 BPB with rank 128 vs 96)
4. **H4: Higher LR improves TTT at rank 128** ❌ Rejected (v1→v2: +0.052 BPB regression at 3e-4)
5. **H5: Larger batch/chunk improves TTT** ❌ Untestable (all batch-128 variants failed)
6. **H6: GPTQ quantization degrades BPB** ✅ Confirmed at matched 240min / 300min / 360min checkpoints (+0.00940208 / +0.00657678 / +0.00932885 BPB EMA->quantized)

## Decomposition of BPB Improvement Pipeline

| Stage | BPB | Δ from previous |
|-------|-----|-----------------|
| Training val (live model, non-EMA, earlier step) | 1.0599 | — |
| Matched 6h pre-quant EMA | 1.03340201 | not a like-for-like delta vs prior row |
| Quantized 6h artifact (post-EMA serialize, sliding eval) | 1.04273086 | +0.00932885 vs matched pre-quant EMA |
| Score-first TTT (v7: K+MLP+O rank 96, no Q/V) | 1.03387340 | −0.00885746 from quantized, +0.00047139 vs matched pre-quant EMA |

Matched 240min control: pre-quant EMA **1.03545673** -> quantized **1.04485881**
(+0.00940208 tax) -> post-TTT **1.03539272** (−0.00946609 from quantized, within
0.00006401 of pre-quant EMA).

Matched 300min decomposition on the same checkpoint: live **1.08215117** -> EMA
**1.04945326** (−0.03269791) -> quantized **1.05603004** (+0.00657678 tax) ->
post-TTT **1.04210727** (−0.01392277 from quantized, −0.00734599 vs EMA).

Matched 360min follow-up: pre-quant EMA **1.03340201** -> quantized
**1.04273086** (+0.00932885 tax) -> post-TTT **1.03470849**
(−0.00802237 from quantized, +0.00130648 vs pre-quant EMA).

## Infrastructure Additions

- Resumable rank-local checkpoints (atomic writes, manifest-driven)
- SCHEDULE_HORIZON_SECONDS for faithful schedule continuation beyond original horizon
- sweep-only-artifact mode for standalone TTT evaluation
- HTTP-based file upload through RunPod proxy (replaces SSH which is blocked)
- Per-variant isolated subprocess execution with timeout

## Compliance

- ⚠️ NOT record-track compliant (training exceeds 600s)
- ✅ Identical training recipe to PR #1950/1934
- ✅ Score-first TTT (no validation tokens seen before scoring)
- ✅ Artifact < 16 MB (15,926,271 bytes)
- ✅ TTT/LoRA is RAM-only at eval time, does not affect artifact size

## Hardware & Cost

- Phase 1 (1h): 8×H100 SXM ($7.48/hr) — 60 min
- Phase 2 (4h): 4×H100 NVL ($5.98/hr × 2 pods) — ~4h + 6h resumed
- TTT Sweep: 4×H100 NVL ($11.96/hr) — 90 min
- Follow-up controls (240min TTT-only, 300min decomposition, 360min pre-quant): 4×H100 NVL — ~3.5h
- Total estimated cost: ~$160
