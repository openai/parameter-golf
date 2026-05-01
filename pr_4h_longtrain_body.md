# [Non-Record] 6h Long-Train Scaling + TTT Hyperparameter Sweep

## Summary

**Non-record experiment** studying BPB as a function of training duration (10 min → 6h) and systematically sweeping TTT/LoRA hyperparameters on the final 6h quantized artifact.

**Key findings:**
1. Post-TTT BPB improves from 1.060 (10 min, 3-seed mean) to **1.03471** (6h single-seed, post-TTT)
2. A matched 360min comparator gives **pre-quant EMA 1.03340201 -> quantized 1.04273086 -> post-TTT 1.03470849**, so GPTQ adds **+0.00932885 BPB** at 6h and TTT recovers **0.00802237 BPB** of that tax
3. The final 6h post-TTT result remains only **+0.00130648 BPB** above the matched 6h pre-quant EMA, i.e. TTT recovers about **86%** of the 6h quantization tax
4. Additional matched 240min and 300min controls show the same causal structure: EMA helps, GPTQ adds a modest tax, and TTT often recovers most or all of it
5. Artifact size is effectively constant (±27 KB) across all durations
6. PR #1979 control TTT parameters (rank 96, alpha 144, lr 1e-4) were **best among tested variants**
7. At rank 128/alpha 192, raising LR from 1e-4 to 3e-4 worsened BPB by ~0.052

### Training Scaling Results

All durations use the identical PR #1950/1934 recipe. Metrics differ by evaluation stage:

| Duration | Steps | training_val_bpb | quantized_bpb | post_ttt_bpb | Artifact |
|----------|-------|-----------------|---------------|-------------|----------|
| 10 min (3-seed, PR #1934) | ~4000 | — | — | 1.06003† | 15,953 KB |
| 60 min (8×H100, PR #1979) | ~8000 | 1.0615 | — | 1.03988 | 15,944 KB |
| 240 min (4×H100) | ~30K | — | 1.0449 | — | 15,933 KB |
| **360 min (4×H100)** | 49765 | 1.0599* | **1.04273** | **1.03471** | **15,926 KB** |

*training_val_bpb at step ~48000 (last logged); †3-seed mean from record submission  
TTT gain = quantized_bpb - post_ttt_bpb = 1.04273 - 1.03471 = **0.00802 BPB**

### TTT/LoRA Sweep (on 360-min quantized artifact)

| Variant | LoRA Rank/Alpha | LR | Batch | post_ttt_bpb | Peak Memory | Status |
|---------|----------------|------|-------|-------------|-------------|--------|
| **sliding_window_control** | — | — | — | 1.04273 | 5.3 GB | ✓ baseline |
| **v0_control (PR #1979)** | 96/144 | 1e-4 | 64 | **1.03471** | 47.8 GB | ✓ best |
| v1_rank128_alpha192 | 128/192 | 1e-4 | 64 | 1.03877 | — | ✓ |
| v2_rank128_lr3e4 | 128/192 | 3e-4 | 64 | 1.09049 | — | ✓ regression |
| v3_local_batch_chunk | 128/192 | 3e-4 | 128 | — | — | failed |
| v4_global2_largechunk | 128/192 | 3e-4 | 128 | — | — | failed |
| v5_prefix3000 | 128/192 | 3e-4 | 128 | — | — | failed |
| v6_prefix3000_phase4 | 128/192 | 3e-4 | 128 | — | — | failed (optional) |

The sliding_window_control runs quantized model evaluation with no TTT adaptation, providing the proper baseline to isolate the TTT contribution (0.00802 BPB gain).

v3–v6 failed with exit code 1 (no explicit OOM traceback available, but batch doubling from 64→128 would exceed 80 GB given v0 peak at 47.8 GB).

## ML Changes from Reference

**Training:** No training-side ML change from PR #1950/1934. The resumed 6h run keeps the same architecture, optimizer, loss, tokenizer/data setup, and schedule semantics; the code additions are infrastructural (resume checkpoints, periodic exports, schedule-horizon continuation).

**Eval-only adaptation changes:** Relative to the PR #1979 control, this submission also runs a RAM-only TTT/LoRA sweep over adaptation hyperparameters including LoRA rank/alpha, LoRA LR, local batch/chunk size, global TTT epochs/chunk tokens/batch seqs/warmup, and phased prefix/phase count. These changes affect only evaluation, never the serialized 16 MB artifact. The best result still came from the original PR #1979 control.

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
| Score-first TTT (rank 96) | 1.03470849 | −0.00802237 from quantized, +0.00130648 vs matched pre-quant EMA |

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
