# PR #1950 Long-Train Artifact Scaling + TTT Sweep

**Non-record track** — training exceeds the 600s wallclock budget.

> **No training-side ML change on top of PR #1950.** This experiment keeps the
> PR #1950 / PR #1934 training recipe fixed, extends wallclock for non-record
> study, and adds an eval-only TTT/LoRA sweep around the PR #1979 control.

## Result Summary

| Metric | 1h (8×H100) | 4h (4×H100) | 6h (4×H100) |
|--------|-------------|-------------|-------------|
| Training steps | 16,001 | 30,688 | 49,765 |
| Training val_bpb | 1.0615 | — | 1.0599* |
| Quantized BPB (pre-TTT) | — | 1.0449 | **1.04273** |
| Post-TTT BPB | 1.03988 | — | **1.03387** |
| TTT gain | — | — | 0.00886 |
| Artifact bytes | 15,944,203 | 15,932,638 | 15,926,271 |

*training_val_bpb at step ~48000 (last logged, non-EMA, earlier step; not a like-for-like GPTQ comparator); †3-seed mean from record submission

**Conclusions:**
1. Post-TTT BPB improves with training duration (1.060 at 10 min → 1.034 at 6h; note: 10-min is 3-seed mean, 6h is seed-42 only)
2. Artifact size is effectively constant (within ~27 KB, 0.17%) — compression is at entropy floor
3. **Removing Q/V LoRA targets** (v7) while keeping K+MLP+O achieves the best BPB
   (1.03387) with 4.2 GiB less memory than the full-target control
4. Single-phase TTT with only 1000 prefix docs (v12) nearly matches 3-phase control
   (1.03421 vs 1.03471) in ~60% less global-SGD time
5. At rank 128/alpha 192, raising LR from 1e-4 to 3e-4 worsened BPB by ~0.052
6. Batch-128 variants failed (likely memory-related; peak was 47.8 GB at batch 64)
7. Matched 240min, 300min, and 360min checkpoint controls all show GPTQ tax; at 6h
   the tax is +0.00932885 BPB and TTT (v7) recovers ~95% of it, ending only +0.00047
   above the matched pre-quant EMA

## TTT/LoRA Hyperparameter Sweep

Sweep conducted on the 360-min (6h) quantized artifact using `TTT_EVAL_ONLY=1`:

| Variant | LoRA Rank | LR | Batch | Chunk | BPB | Status |
|---------|-----------|------|-------|-------|-----|--------|
| **sliding_window** | — | — | — | — | 1.04273 | ✓ baseline |
| v0_control | 96 | 1e-4 | 64 | 48 | 1.03471 | ✓ |
| **v7_noqv_rank96** | 96 (K+MLP+O+lm_head) | 1e-4 | 64 | 48 | **1.03387** | ✓ **best** |
| v12_phase1_prefix1000 | 96 | 1e-4 | 64 | 48 | 1.03421 | ✓ |
| v1_rank128 | 128 | 1e-4 | 64 | 48 | 1.03877 | ✓ |
| v2_rank128_lr3e4 | 128 | 3e-4 | 64 | 48 | 1.09049 | ✓ regression |
| v3_batch128 | 128 | 3e-4 | 128 | 64 | — | failed* |
| v4_global2 | 128 | 3e-4 | 128 | 64 | — | failed* |
| v5_prefix3000 | 128 | 3e-4 | 128 | 64 | — | failed* |
| v6_phase4 | 128 | 3e-4 | 128 | 64 | — | failed* |

The sliding_window control runs the quantized artifact with no TTT adaptation,
providing the proper baseline: **TTT gain = 1.04273 − 1.03387 = 0.00886 BPB** (v7).

*Variants v3–v6 failed with exit code 1 (likely memory-related: v0 peak was 47.8 GB
at batch_size=64, so batch_size=128 would approach or exceed H100 80 GB capacity).

**Fixed parameters across all variants:** TTT_WEIGHT_DECAY=1.0, TTT_BETA1=0,
TTT_BETA2=0.999, TTT_OPTIMIZER=adam, TTT_WARM_START_A=1, FUSED_CE_ENABLED=1,
GLOBAL_TTT_LR=0.001, PHASED_TTT_PREFIX_DOCS=2000, PHASED_TTT_NUM_PHASES=3.

**Key finding:** Removing Q and V LoRA targets (v7) — keeping only K+MLP+O+lm_head —
gives the best BPB (1.03387) with 4.2 GiB less peak memory (43.6 vs 47.8 GB). This
suggests the Q/V pathway may introduce mild overfitting on prefix documents. A
single-phase variant (v12) with only 1000 prefix docs also nearly matches the full
3-phase 2000-prefix control (1.03421 vs 1.03471) in ~60% less global-SGD time.
Higher LoRA rank provides minimal benefit (+0.004 BPB)
while at rank 128, raising LR from 1e-4 to 3e-4 worsened BPB by ~0.052 (v1→v2).
TTT adaptation is RAM-only at eval time and
does not change the 16 MB artifact size.

## Research Questions

1. **Does longer training (10 min to 6h) improve BPB?** YES — monotonically.
2. **Does longer training reduce artifact size?** NO — compression is at entropy floor.
3. **Can TTT/LoRA parameters be improved?** YES — removing Q/V LoRA targets (v7) improves by 0.00084 BPB over the full-target control.

## Base Recipe

PR #1950 (compliance-audited reproduction of PR #1934):
- 11-layer transformer, dim=512, 8 attn heads / 4 KV heads (GQA), 4× MLP
- SmearGate (window=12), SparseAttnGate, fused CE
- INT6 GPTQ quantization + INT7 embeddings + LQER asymmetric rank-4 (top-3)
- Per-group lrzip compression
- Phased score-first TTT (3 phases, 2000 prefix docs, LoRA rank 96)
- Baseline (10 min record-track): **val_bpb ≈ 1.06003**, artifact ≈ 15.97 MB

## Training Scaling Results

### Phase 1: 1h on 8×H100 SXM

| Minute | Steps  | Artifact (bytes) | Δ vs 10 min | Notes |
|--------|--------|-----------------|-------------|-------|
| 10     | 6,348  | 15,953,292      | baseline    | In-loop export |
| 20     | 7,193  | 15,952,677      | −615        | In-loop export |
| 30     | 7,899  | 15,956,638      | +3,346      | In-loop export |
| 45     | 12,135 | 15,955,847      | +2,555      | In-loop export |
| **60** | 16,001 | 15,944,203      | **−9,089**  | Post-stop export |

### Phase 2: 4h on 4×H100 NVL (separate run)

| Minute | Steps  | Artifact (bytes) | val_bpb | Notes |
|--------|--------|-----------------|---------|-------|
| 60     | 10,488 | 15,947,774      | 1.1720  | In-loop export |
| 120    | 17,480 | 15,944,413      | 1.1389  | In-loop export |
| 180    | 23,418 | 15,944,789      | 1.1183  | In-loop export |
| 240    | 30,688 | 15,932,638      | 1.0449  | Final GPTQ export |

### Phase 3: 6h on 4×H100 NVL (resumed from 300 min)

| Minute | Steps  | Artifact (bytes) | val_bpb | Notes |
|--------|--------|-----------------|---------|-------|
| 360    | 49,765 | 15,926,271      | 1.0599  | LONGTRAIN export at schedule endpoint |

Training resumed at step 36452 (300 min) and continued to 360 min (6h schedule
horizon) using `SCHEDULE_HORIZON_SECONDS=21600`. LR was at minimum for the entire
continuation segment. val_bpb plateaued at ~1.060 from step 44000 onwards.

### Final Model Quality (360-min artifact)

| Eval Stage | val_bpb | Notes |
|------------|---------|-------|
| Training val (step ~48000) | 1.0599 | Live model, non-quantized |
| Pre-quant EMA follow-up | **1.03340201** | Matched 360min EMA eval from resumed continuation |
| Quantized (sliding window, no TTT) | **1.04273** | INT6 GPTQ artifact only |
| Post-TTT (phased, 3 phases, rank 96) | **1.03471** | On quantized artifact |
| TTT gain (quantized → post-TTT) | **0.00802** | True isolated TTT contribution |

The matched 360min comparator now resolves the main question directly:
pre-quant EMA **1.03340201** -> quantized **1.04273086** (+0.00932885 tax) ->
post-TTT **1.03470849** (recovering 0.00802237 of that tax, leaving +0.00130648
vs pre-quant EMA). The matched 1h and 4h EMA→quantized comparisons in this repo
already showed similar tax levels (+0.00975 at 1h, +0.00940 at 4h). The 240min
TTT-only control likewise lands at 1.03539272 from the 240min quantized artifact
(1.04485881), essentially matching the 240min pre-quant EMA measurement of
1.03545673. A matched 300min decomposition gives live 1.08215117 -> EMA 1.04945326
-> quantized 1.05603004 -> post-TTT 1.04210727, i.e. EMA provides the large gain,
GPTQ adds +0.00657678 BPB tax, and TTT more than recovers that tax on that checkpoint.

## Experiment Design

### Training Scaling (Phases 1–3)

The modified `train_gpt.py` adds a `NON_RECORD_LONGTRAIN=1` mode that:

1. Trains for up to `MAX_WALLCLOCK_SECONDS` (default 3600s = 60 min)
2. At configurable milestones (default: 10, 20, 30, 45, 60 min):
   - Synchronizes all distributed ranks via `dist.broadcast` (decision) + `dist.barrier`
   - Pauses training timer
   - Applies EMA weights to a model copy
   - Runs full GPTQ quantization + lrzip compression (serialize)
   - Records artifact size, step count, and timing metadata
   - Restores non-EMA weights and resumes training
3. After training completes at the wallclock cap, runs standard serialize + TTT eval

Phase 3 (6h continuation) uses `SCHEDULE_HORIZON_SECONDS=21600` to preserve
the original 6h LR schedule semantics during continuation beyond the initial
4h run. Training was resumed from a checkpoint captured at 300 min (step 36452).

### TTT/LoRA Sweep (Phase 4)

The sweep uses `TTT_EVAL_ONLY=1` mode which:
1. Loads the quantized INT6 GPTQ artifact from disk
2. Applies LoRA-based test-time training adaptation
3. Runs phased score-first evaluation (3 phases, 2000 prefix docs)
4. Reports final BPB and timing metrics

Each variant runs in an isolated subprocess with its own output directory
to prevent state contamination. LoRA parameters are RAM-only at eval time
and do **not** modify the 16 MB artifact.

### Key Technical Notes

- **Distributed sync:** Rank 0 broadcasts the "export now" decision to prevent NCCL
  desync across ranks (avoids timeout from timer drift between ranks).
- **torch.compile invalidation:** Each checkpoint export triggers a `torch._dynamo.reset()`
  which causes substantial post-export throughput loss from graph recompilation.
- **Resumable checkpoints:** Rank-local saves with manifest-driven validation.
  Refuses resume if world_size, architecture, or optimizer config changed.
- **Memory limits on larger TTT batches:** TTT_BATCH_SIZE=128 likely exceeds H100 80GB capacity
  with this model (peak 47.8 GB for batch=64 → estimated 77+ GB for batch=128). Variants
  v3–v6 failed with exit code 1; no explicit OOM trace was captured.

## Interpretation

Per our pre-registered decision framework:

| Condition | Threshold | Actual | Result |
|-----------|-----------|--------|--------|
| Artifact shrink ≥ 300 KB | Recommend larger model | −27 KB (6h vs 10 min) | ❌ Not met |
| Artifact shrink 50–300 KB | Report scaling benefit | −27 KB | ❌ Not met |
| BPB improves, no size change | Quality-only benefit | ✓ | ✅ **This case** |
| TTT params can be improved | Lower post-TTT BPB | Control is best | ❌ Not met |

**Decision:** Longer training improves BPB quality substantially but does NOT free
artifact budget for a larger model. The compression pipeline (INT6 GPTQ + per-group
lrzip) reaches its entropy floor within the first 10 minutes of training.
The existing TTT parameters (from PR #1979 / PR #461 / PR #1767) were best among tested variants.

## Hardware & Cost

| Phase | Hardware | Runtime | Est. Cost |
|-------|----------|---------|-----------|
| 1h scaling | 8×H100 SXM | ~101 min | ~$36 |
| 4h scaling | 4×H100 NVL | ~300 min | ~$60 |
| 6h continuation | 4×H100 NVL | ~65 min | ~$13 |
| TTT sweep | 4×H100 NVL | ~60 min | ~$12 |
| Follow-up controls (240 TTT-only, 300 decompose, 360 pre-quant) | 4×H100 NVL | ~205 min | ~$41 |
| **Total** | | | **~$160** |

## Files

| File | Purpose |
|------|---------|
| `train_gpt.py` | Modified PR #1950 script with LONGTRAIN + resume + TTT_EVAL_ONLY |
| `train.log` | Rank-0 training log from 8×H100 run (seed 42, 1h) |
| `pgolf_stdout.txt` | Combined stdout (1h run) |
| `submission.json` | Experiment metadata |
| `results/checkpoint_*.json` | Per-milestone artifact size and step data |
| `results/followups/*.json` | Matched 240/300/360 comparator follow-up summaries |
| `results/followups/followup_controls_summary.csv` | Aggregated follow-up control table |
| `results/ttt_sweep/ttt_sweep_results.csv` | TTT sweep results (all 7 variants) |
| `results/ttt_sweep/ttt_sweep_summary.json` | Sweep summary with best variant |
| `results/ttt_sweep/ttt_sweep_manifest.json` | Sweep configuration manifest |
| `results/scaling_results.csv` | Tabular data for 1h checkpoints |
| `results/experiment_summary.json` | 1h summary with conclusions |
| `scripts/run_longtrain_scaling.sh` | Launcher script with all env vars |
| `notes/IMPLEMENTATION_NOTES.md` | Implementation details and safety analysis |

## Reproducing

### Training (4×H100 NVL, 6h)

```bash
python3 scripts/run_longtrain_scaling.py \
  --num-gpus 4 --duration-hours 6 --max-wallclock 21600 \
  --export-minutes 60,120,180,240,360 --enable-resume \
  --resume-save-minutes "270,300,330,360" \
  --iterations 200000 --max-minutes 400
```

### TTT Sweep (on existing artifact)

```bash
python3 scripts/run_longtrain_scaling.py \
  --sweep-only-artifact results/<run>/final_model.int6.360min.ptz \
  --num-gpus 4 --max-minutes 150 --ttt-max-minutes-per-variant 20 \
  --results-dir results/ttt_sweep_360min
```

### Local dry-run

```bash
python3 scripts/run_longtrain_ttt_sweep.py \
  --dry-run --artifact /path/to/final_model.int6.ptz
```

## Related Work

| PR | Contribution |
|----|-------------|
| **#1950** | Compliance-audited base recipe (this experiment's foundation) |
| **#1934** | Record-track 3-seed submission (val_bpb 1.06003) |
| **#1979** | 1h long-train study; post-TTT BPB 1.0399 |
| **#461** | Score-first legal TTT framework (phased evaluation) |
| **#1767** | TTT alpha/warm-start/weight-decay improvements |
| **#1855** | QK_GAIN_INIT + TTT_LORA_RANK exploration |

## Compliance Statement

- ⚠️ **NOT record-track compliant** — training time exceeds 600s.
- ✅ Evaluation scoring is unchanged from PR #1950.
- ✅ No PPM-D, n-gram cache, or eval-time scoring changes.
- ✅ No validation tokens accessed before scoring (score-first TTT).
- ✅ No external network calls during train/eval.
- ✅ Artifact fits within 16 MB (15,926,271 bytes < 16,000,000).
- ✅ TTT/LoRA parameters are RAM-only at eval time — do not affect artifact size.
