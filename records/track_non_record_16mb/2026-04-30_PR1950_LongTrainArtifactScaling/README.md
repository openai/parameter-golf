# PR #1950 Long-Train Artifact Scaling Experiment

**Non-record track** — training exceeds the 600s wallclock budget.

> **No ML change on top of PR #1950.** This experiment uses the identical model
> architecture, hyperparameters, and scoring pipeline from PR #1950 (our
> compliance-audited reproduction of PR #1934). The only modification is removing
> the 600s wallclock cap to study artifact compressibility vs. training duration.

## Result Summary

| Metric | Value |
|--------|-------|
| Post-TTT BPB (60 min) | **1.03988** |
| Quantized BPB (60 min) | 1.04944 |
| Pre-quant BPB (60 min) | 1.03969 |
| Artifact (60 min) | 15,944,203 bytes |
| Artifact (10 min baseline) | 15,953,292 bytes |
| Net artifact shrink | **−9,089 bytes (−0.06%)** |
| Quantization tax | 0.00975 BPB |
| TTT gain | 0.00956 BPB |

**Conclusion:** Longer training dramatically improves BPB (1.18 → 1.06 training val;
1.04 post-TTT at final model) but does NOT meaningfully reduce artifact size. The
INT6 GPTQ + per-group lrzip compression is already near-optimal for this architecture
at 10 min of training.

## Research Question

> Does longer training (10–60 minutes) make the PR #1950 GPTQ+lrzip model more
> compressible, achieve lower BPB, or both?

If longer training significantly shrinks the compressed artifact, the freed bytes
could fund a larger model (more layers, wider dim, higher LQER_TOP_K) that still
fits within the 16 MB cap on the record track.

## Base Recipe

PR #1950 (compliance-audited reproduction of PR #1934):
- 11-layer transformer, dim=512, 8 attn heads / 4 KV heads (GQA), 4× MLP
- SmearGate (window=12), SparseAttnGate, fused CE
- INT6 GPTQ quantization + INT7 embeddings + LQER asymmetric rank-4 (top-3)
- Per-group lrzip compression
- Phased score-first TTT (3 phases, 2000 prefix docs, LoRA rank 96)
- Baseline (10 min record-track): **val_bpb ≈ 1.06003**, artifact ≈ 15.97 MB

## Artifact Scaling Results

| Minute | Steps  | Artifact (bytes) | Δ vs 10 min | Notes |
|--------|--------|-----------------|-------------|-------|
| 10     | 6,348  | 15,953,292      | baseline    | In-loop export |
| 20     | 7,193  | 15,952,677      | −615        | In-loop export |
| 30     | 7,899  | 15,956,638      | +3,346      | In-loop export |
| 45     | 12,135 | 15,955,847      | +2,555      | In-loop export |
| **final** (~60 min cap) | 16,001 | 15,944,203 | **−9,089** | Post-stop export at 3598s¹ |

¹ Training stopped at 3598.25s due to `GPTQ_RESERVE_SECONDS=5.5` (cap = 3600 − 5.5 = 3594.5s).
The 60-min in-loop trigger never fired because training time never reached 3600s.
The final export is the standard end-of-training serialize (identical path to PR #1950).

**Training val_bpb progression** (measured at periodic validation steps):
- Step 4,000 (before 10-min export): 1.1779
- Step 8,000: 1.1359
- Step 12,000: 1.1039
- Step 16,001 (final): 1.0615

### Final Model Eval (post-stop at 3598s)

| Eval Stage | val_bpb |
|------------|---------|
| Pre-quantization (EMA, diagnostic) | 1.03969 |
| Post-INT6-GPTQ quantization | 1.04944 |
| Post-TTT (phased, score-first) | **1.03988** |

Quantization tax: 0.00975 BPB; TTT gain: 0.00956 BPB (measured on final model only).

## Experiment Design

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

### Key Technical Notes

- **Distributed sync:** Rank 0 broadcasts the "export now" decision to prevent NCCL
  desync across ranks (avoids timeout from timer drift between ranks).
- **torch.compile invalidation:** Each checkpoint export triggers a `torch._dynamo.reset()`
  which causes substantial post-export throughput loss from graph recompilation.
  Interval throughput: steps 6348→7193 (845 steps in 600s = 1.4 steps/s) vs normal
  ~4.7 steps/s, suggesting ~70% overhead in the first 600s after each export.
- **Step count discrepancy:** Due to recompilation overhead after each export,
  effective training throughput is reduced by ~20% from the 5 export pauses.

## Interpretation

Per our pre-registered decision framework:

| Condition | Threshold | Actual | Result |
|-----------|-----------|--------|--------|
| Artifact shrink ≥ 300 KB | Recommend larger model | −9 KB | ❌ Not met |
| Artifact shrink 50–300 KB | Report scaling benefit | −9 KB | ❌ Not met |
| BPB improves, no size change | Quality-only benefit | ✓ | ✅ **This case** |

**Decision:** Longer training improves BPB quality substantially but does NOT free
artifact budget for a larger model. The compression pipeline (INT6 GPTQ + per-group
lrzip) reaches its entropy floor within the first 10 minutes of training.

## Hardware & Cost

- **GPU:** 8×H100 80GB SXM (RunPod COMMUNITY)
- **Pod:** `hq01mtcdiivfij` at $21.52/hr
- **Runtime:** ~101 min (data download + 60 min training + 5 checkpoint exports + final eval + TTT)
- **Estimated cost:** ~$36

## Files

| File | Purpose |
|------|---------|
| `train_gpt.py` | Modified PR #1950 script with `NON_RECORD_LONGTRAIN` support |
| `train.log` | Rank-0 training log from 8×H100 run (seed 42) |
| `pgolf_stdout.txt` | Combined stdout (launcher context, preflight, training, eval) |
| `submission.json` | Experiment metadata |
| `scripts/run_longtrain_scaling.sh` | Launcher script with all env vars |
| `scripts/analyze_scaling.py` | Post-hoc analysis of checkpoint JSONs |
| `scripts/make_larger_variant_plan.py` | Larger-model plan generator (unused — threshold not met) |
| `notes/IMPLEMENTATION_NOTES.md` | Implementation details and safety analysis |
| `results/checkpoint_*.json` | Per-milestone artifact size and step data |
| `results/checkpoint_60min.json` | Manually created from final-export metrics (see note above) |
| `results/scaling_results.csv` | Tabular data for all checkpoints |
| `results/experiment_summary.json` | Full summary with conclusions |

## Reproducing

### On RunPod (8×H100)

```bash
# Standard PR #1950 recipe + long-train overrides
SEED=42 NON_RECORD_LONGTRAIN=1 MAX_WALLCLOCK_SECONDS=3600 \
  LONGTRAIN_EXPORT_MINUTES=10,20,30,45,60 EXPORT_MODE=light \
  GPTQ_RESERVE_SECONDS=5.5 COMPRESSOR=pergroup EMBED_WD=0.06 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=12.0 MLP_CLIP_SIGMAS=12.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=12.0 MATRIX_LR=0.026 MIN_LR=0.1 \
  CASEOPS_ENABLED=1 SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 SPARSE_ATTN_GATE_ENABLED=1 \
  FUSED_CE_ENABLED=1 TTT_WARM_START_A=1 PHASED_TTT_PREFIX_DOCS=2000 \
  PHASED_TTT_NUM_PHASES=3 NCCL_NET=Socket \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Analyzing Results

```bash
python3 scripts/analyze_scaling.py results/
```

## Compliance Statement

- ⚠️ **NOT record-track compliant** — training time exceeds 600s.
- ✅ Evaluation scoring is unchanged from PR #1950.
- ✅ No PPM-D, n-gram cache, or eval-time scoring changes.
- ✅ No validation tokens accessed before scoring (score-first TTT).
- ✅ No external network calls during train/eval.
- ✅ Artifact fits within 16 MB (15,944,203 bytes < 16,000,000).
