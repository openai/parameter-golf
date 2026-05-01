# [Non-Record] 6h Long-Train Scaling + TTT Hyperparameter Sweep

## Summary

**Non-record experiment** studying BPB as a function of training duration (10 min → 6h) and systematically sweeping TTT/LoRA hyperparameters on the final 6h quantized artifact.

**Key findings:**
1. Post-TTT BPB improves from 1.060 (10 min, 3-seed mean) to **1.03471** (6h single-seed, post-TTT)
2. GPTQ quantization improves BPB: 1.0599 (training val) → **1.04273** (quantized) — a 0.017 gain suggesting quantization acts as regularization
3. True TTT contribution is **0.00802 BPB** (quantized → post-TTT)
4. Artifact size is effectively constant (±27 KB) across all durations
5. PR #1979 control TTT parameters (rank 96, alpha 144, lr 1e-4) were **best among tested variants**
6. At rank 128/alpha 192, raising LR from 1e-4 to 3e-4 worsened BPB by ~0.052

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

**Training:** No ML changes from PR #1950/1934. Identical architecture, hyperparameters, loss function, optimizer, and schedule. Only infrastructure additions (resumable checkpoints, periodic artifact export, wallclock extension).

**TTT/LoRA parameters (eval-only, RAM-only):** The TTT parameters used are identical to those established in PR #1979 (which inherited from PR #461 score-first framework + PR #1767 improvements + PR #1855 rank exploration). These parameters are applied only at eval time and do not affect the 16 MB artifact. The sweep tested variations but found the control best.

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
6. **H6: GPTQ quantization degrades BPB** ❌ Rejected — GPTQ *improves* BPB by 0.017 (1.0599 → 1.0427), suggesting quantization regularizes the model

## Decomposition of BPB Improvement Pipeline

| Stage | BPB | Δ from previous |
|-------|-----|-----------------|
| Training val (live model) | 1.0599 | — |
| INT6 GPTQ quantization | 1.04273 | −0.01717 (quantization gain) |
| Score-first TTT (rank 96) | 1.03471 | −0.00802 (TTT gain) |
| **Total pipeline gain** | | **−0.02519** |

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

- Phase 1 (1h): 8×H100 NVL ($7.48/hr) — 60 min
- Phase 2 (4h): 4×H100 NVL ($5.98/hr × 2 pods) — ~4h + 6h resumed
- TTT Sweep: 4×H100 NVL ($11.96/hr) — 90 min
- Total estimated cost: ~$85
