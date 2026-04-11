# Non-Record Submission: SwiGLU + Warmdown Fix + Quarter Batch (1×RTX 5090)

This is a non-record submission documenting a systematic 10-experiment exploration on a **single RTX 5090**, iterating from the stock baseline toward better val_bpb under the 16MB artifact constraint.

The final post-quant score (**1.3281 val_bpb**) does not beat the 8×H100 baseline (1.2244) due to hardware throughput limitations (~3,773 steps vs ~13,780 on 8×H100), but the individual improvements — particularly the **warmdown schedule bug fix** — are hardware-agnostic and should transfer directly to multi-GPU runs.

## Summary of Changes (cumulative, all kept)

1. **SwiGLU activation** replacing ReLU² — better gating mechanism, widely adopted in modern LLMs
2. **Warmdown schedule bug fix** — stock config decays LR from step 1; fixed via time-fraction approach
3. **Reduced MLP hidden (640)** — trades params for artifact budget headroom
4. **Quarter batch size (131K tokens)** — 4× more optimizer steps in the same wall-clock time
5. **Gradient accumulation (2 steps)** — doubles effective batch without increasing per-step memory

## Key Discovery: Warmdown Schedule Bug

The stock `train_gpt.py` sets `warmdown_iters=1200`, but with a 600s wallclock cap the implied warmdown window exceeds total training time. This means the learning rate decays from step 1 — the model never trains at full LR.

**Fix:** Replace iteration-based warmdown with a time-fraction approach (`warmdown_frac=0.2`), so warmdown occupies the last 20% of wall-clock time. This alone gave **-0.006 bpb** improvement.

## Full Experiment Log

| Exp | Description | val_bpb | Delta | Artifact (MB) | Status |
|-----|-------------|---------|-------|---------------|--------|
| 001 | Baseline (stock config) | 1.3633 | — | 12.3 | keep |
| 002 | SwiGLU MLP | 1.3592 | -0.0041 | 15.1 | keep |
| 003 | Warmdown fix (time-fraction 20%) | 1.3536 | -0.0056 | 17.9 | discard (>16MB) |
| 004 | SwiGLU(768) + warmdown fix | 1.3496 | -0.0096 | 15.4 | keep |
| 005 | Half batch (262K tokens) | 1.3336 | -0.0160 | 16.6 | discard (>16MB) |
| 006 | Half batch + MLP hidden 704 | 1.3359 | -0.0137 | 15.8 | keep |
| 007 | Quarter batch (131K) + MLP hidden 640 | 1.3305 | -0.0054 | 15.3 | keep |
| 008 | + Gradient accumulation ×2 | **1.3281** | -0.0024 | 15.3 | **best** |
| 009 | + Weight decay 0.01 | 1.3284 | +0.0002 | 15.3 | discard |
| 010 | Layer recurrence ×2 | 1.3791 | +0.0510 | 15.1 | discard |

**Total improvement over baseline: -0.0352 bpb** (1.3633 → 1.3281)

## Negative Results Worth Noting

- **Weight decay** (exp009): No benefit at this scale/duration. The regularization effect is negligible for short training runs.
- **Layer recurrence** (exp010): Doubling depth by reusing weights halves the number of training steps in fixed wall-clock time, which more than offsets any capacity gain. Worst result since baseline (+0.051 bpb).

## Configuration (Best Run — exp008)

```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_HIDDEN=640 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024
WARMDOWN_FRAC=0.2 GRAD_ACCUM_STEPS=2
```

Key metrics:
- `val_bpb` (post-quant): **1.32814313**
- Artifact size: **15,327,112 bytes** (~670KB headroom)
- Model params: 16,470,088
- Steps completed: 3,773
- Peak memory: 10,225 MiB
- GPU: 1×RTX 5090, 600s wallclock

## Hardware Note

All experiments ran on a single RTX 5090 with a 10-minute wallclock cap. The throughput gap vs 8×H100 (~3.6× fewer steps) explains the score gap vs the baseline leaderboard entry. The architectural and schedule improvements documented here are hardware-agnostic and intended to be validated on 8×H100 as a next step.

## Included Files

- `train_gpt.py` — code snapshot of the best configuration so far (008)
- `results.tsv` — full experiment results table
- `submission.json` — leaderboard metadata
