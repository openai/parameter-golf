# Agent Context: Recurrent Depth Experiments

## Goal

Beat the current SOTA of **1.1194 bpb** on the Parameter Golf 10-min / 8xH100 / 16MB track. The significance threshold is **0.005 nats** improvement (~1.1164 bpb or lower).

## Architecture

11-layer transformer with depth recurrence: layers 4-6 are the "core" block, reused multiple times. Progressive training ramps passes from 1→2→3→4 during training, then evaluates with 4 passes. ResidualScale (learnable per-pass scalars) and Jacobian proxy loss keep recurrence contractive.

Key modules:
- `train_gpt_recurrent.py` — main training/eval script (~100KB)
- `feedback.py` — ErrorFeedbackModule (diagonal, rank 2, 2560 params)
- `stability.py` — RecurrentStabilizer + ResidualScale

## Current Best Result (1-GPU, progressive_1to4)

- **TTT bpb: 1.1147** (1.8820 nats) — beats SOTA by 0.009 nats
- Pre-TTT (quantized, 4-pass): 1.1526 bpb
- Artifact size: **16,222,054 bytes — OVER the 16,000,000 limit by 222KB**
- Model compressed (int6+lzma preset 6): 16,122,576 bytes
- Code: 99,478 bytes
- Log: `logs/progressive_1to4.txt`

## The Size Problem

The 16MB limit is **decimal 16,000,000 bytes** (confirmed in repo README.md line 171: "The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes").

The compressed model ALONE (16.1MB) exceeds the limit. LZMA preset 7/8/9 tested — they make it **worse** (the data is already at LZMA's sweet spot at preset 6). The size overshoot is due to weight entropy from progressive training producing less compressible weight distributions compared to the baseline.

The SOTA baseline (same architecture, non-recurrent) compressed to 15.99MB. Our model has identical parameter count (26,927,712) but the progressive training changes weight distributions.

## Three Run Scripts (all 8xH100, 600s wallclock)

### 1. `run_submission.sh` — Winning config
- LATE_QAT_THRESHOLD=0.15 (~200 QAT steps on 8 GPUs)
- feedback-mode diagonal
- Best bpb but artifact oversized

### 2. `run_earlyqat.sh` — Earlier QAT for smaller artifact
- LATE_QAT_THRESHOLD=0.25 (~400 QAT steps on 8 GPUs)
- feedback-mode diagonal
- Hypothesis: more QAT steps = lower weight entropy = better compression

### 3. `run_nofeedback.sh` — No feedback + early QAT
- LATE_QAT_THRESHOLD=0.25
- feedback-mode none
- Hypothesis: feedback module was NEVER used at eval/TTT time (bug — `eval_val_sliding_ttt` never passes `feedback_fn` to forward calls). So the model trained WITH feedback corrections it won't have at eval. Removing feedback from training should make the model learn to be stable without corrections, potentially improving eval quality AND removing 2560 unused training params from the optimizer.

## Key Training Config (shared across all runs)

```
ITERATIONS=6500            # wallclock cap stops it at ~6100-6200 steps on 8 GPUs
WARMDOWN_ITERS=2500        # time-based when wallclock active
PASSES_SCHEDULE="0:1,4500:2,5500:3,6000:4"
NUM_PASSES=1               # initial passes
EVAL_PASSES=4              # override at eval time
CORE_START=4 CORE_END=7    # layers 4-6 are the recurrent core
TRAIN_BATCH_TOKENS=786432  # matches 8-GPU effective batch
SWA_ENABLED=1 SWA_EVERY=50
--residual-scale-init 0.5
--jacobian-proxy-weight 0.1
--no-interpass-rmsnorm
```

## lr_mul is Time-Based on 8 GPUs

When `MAX_WALLCLOCK_SECONDS=600` is set, the `lr_mul` function switches from step-based to time-based warmdown. `WARMDOWN_ITERS` controls the warmdown duration as `warmdown_iters * step_avg_ms` in real time. SWA triggers at `scale < 0.2`, Late QAT at `scale < threshold`. These auto-adapt to step speed.

Estimated 8-GPU timeline:
- Steps 0-4499: 1-pass, ~87ms/step
- Steps 4500-5499: 2-pass, ~116ms/step
- Steps 5500-5999: 3-pass, ~140ms/step
- Steps 6000+: 4-pass, ~178ms/step
- QAT 0.25 triggers ~step 5750, QAT 0.15 ~step 5950
- SWA starts ~step 5800
- Training ends ~step 6150 (wallclock cap)

## Pre-compilation of Progressive Passes

`torch.compile` traces are cached during warmup. The last N warmup steps cycle through each pass count variant so all compiled graphs are ready before the timed training loop. No recompilation overhead during training.

## The Feedback Bug

`eval_val_sliding_ttt` calls `base_model.forward_logits(x_batch)` (scoring) and `base_model(x, y)` (TTT training) WITHOUT passing `feedback_fn`. Both `forward()` and `forward_logits()` accept `feedback_fn=None` as default. The `_forward_hidden` method applies feedback at line 1001-1002:

```python
if feedback_fn is not None:
    x = x + feedback_fn(x, k)
```

So during eval/TTT, this is always skipped. The model was trained expecting corrections between passes but never gets them at inference. This is why `run_nofeedback.sh` exists as an experiment.

The feedback weights ARE maintained through EMA (lines 1987-1991) and exist in memory, but they are NOT in the exported artifact (`export_sd` only contains `base_model.state_dict()` minus mtp_heads).

## Quantization Pipeline

1. EMA weights loaded into base_model
2. `num_passes` overridden from training value to `EVAL_PASSES=4`
3. ResidualScale padded for extra passes (init 0.5 for new passes)
4. `export_sd` captured (re-captured AFTER ResidualScale resize — this was a critical bug fix)
5. State dict unbanked (3D parameter banks → individual 2D weight matrices)
6. `mixed_quantize_int6` quantizes weights: int6 for mlp/attn categories, fp16/fp32 for small params
7. `torch.save` → `lzma.compress(preset=6)` → `final_model.int6.ptz`
8. Decompressed and loaded into fresh `eval_model` for TTT

## Files on Disk

- `final_model.pt` — uncompressed model (106MB), can be re-quantized offline
- `final_model.int6.ptz` — compressed artifact (16.1MB), from the progressive_1to4 run
- `logs/progressive_1to4.txt` — the winning run log (1.1147 bpb)
- `logs/full5600.txt`, `full5600_v2.txt` — earlier constant 2-pass runs
- `logs/full_11L_int8core.txt` — failed 11L int8-core experiment
- `logs/smoke_mixedprec*.txt` — early mixed-precision smoke tests
- `2026-03-26_RecurrentSOTA_Feedback_BACKUP/` — backup of the folder before progressive changes

## What to Watch For in New Runs

1. **Artifact size** — must be under 16,000,000 bytes total (model + code). The key line is `Total submission size int6+lzma: XXXXX bytes`.
2. **TTT bpb** — must beat 1.1194 by at least 0.005 nats. Look for `legal_ttt_exact`.
3. **QAT step count** — look for `late_qat:enabled step:XXXX`. More QAT steps = better compression but potentially worse loss.
4. **SWA start** — look for `swa:start step:XXXX`.
5. **Wallclock stop** — look for `stopping_early: wallclock_cap`.

## Competition Submission Format

- All counted code must live in a single `train_gpt.py` script (per README.md). Currently we have 3 files — feedback.py and stability.py should be inlined before final submission.
- Need 3 seeds for statistical significance (p < 0.01).
- `submission.json` needs to be filled with 3-seed mean bpb and bytes_total.
- The PR goes to https://github.com/nestamidavaine/parameter-golf (fork of openai/parameter-golf).
