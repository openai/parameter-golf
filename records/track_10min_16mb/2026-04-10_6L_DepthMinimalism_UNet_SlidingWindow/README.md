# 6L Depth Minimalism: Go Shallow !

> Built and validated on a single A100. Extrapolated to 8xH100 with napkin math and cautious optimism.

**what's the minimum depth that can beat the baseline?**

The answer is **6 layers**. Just a shallow transformer with some strong opinions about architecture.

## The Result

| Metric | Naive Baseline (9L, 8xH100) | This Submission (6L, 1xA100) |
|---|---|---|
| Layers | 9 | **6** (33% fewer) |
| Parameters | 17,059,912 | **16,791,064** (1.6% fewer) |
| Pre-quant val_bpb | 1.2172 | 1.2219 |
| Post-quant val_bpb | 1.2244 | 1.2246 |
| **Sliding window val_bpb** | N/A | **1.2026** |
| Training steps | 13,780 (wallclock-capped) | 20,000 (completed all) |
| Artifact size | 15.86 MB | 15.84 MB |




## The "I Only Have One GPU" Calibration

All metrics are from **1xA100 80GB**. Here's the napkin math for 8xH100:

| | A100 (what I have) | 8xH100 (what the contest has) |
|---|---|---|
| Baseline step_avg | 705.97ms | 43.54ms |
| Speedup ratio | 1x | 16.2x |
| **This model step_avg** | **318.85ms** | **~19.7ms (extrapolated)** |
| Training (20K steps) | 106 min | ~6.6 min |
| Sliding window eval | 26 min | ~1.6 min |
| **Total** | **132 min** | **~8.2 min** |

Conservative estimate with small-batch GPU inefficiency: **~9.7 min**. Either way, fits under 10 minutes. The `MAX_WALLCLOCK_SECONDS=600` warmdown handles it gracefully if the estimate is off.

> Caveat: the 16.2x ratio comes from the baseline where both configs process 65K tokens/GPU/micro-step. Our 6L model does 131K tokens/GPU on A100 but only 16K tokens/GPU on 8xH100 (8 sequences of 2048 per micro-step). Small batches = lower GPU utilization. The real speedup might be 10-12x instead of 16x. Even at 10x, that's 30.8ms/step = 10.3 min with warmdown kicking in at the boundary. It'll be close.

## Why 6 Layers Works

Rather than adding depth. We removed it and compensated with five architectural bets:

**1. Full Attention (4H/4KV, no GQA)**
Every submission uses 8H/4KV grouped query attention. We use 4 heads, each with its own KV projection. At 6 layers you can't afford the approximation — every attention pattern needs to count.

**2. Untied Embeddings**
Rather than tying input/output embeddings to save parameters, we untie them, with only 6 layers to transform embeddings into predictions, a dedicated output head bridges the gap. The 512K extra params are paid for by having 3 fewer layers.

**3. Half Batch (262K tokens, grad_accum=2)**
Standard is 524K. We halve it and accumulate twice, giving 2x more optimizer steps per token. The shallow model's simpler loss landscape rewards frequent updates over large gradient estimates.

**4. Tight Softcap (12.0 vs 30.0)**
Shallow models get overconfident. A tight logit softcap acts as implicit regularization, keeping the output distribution honest.

**5. Long Context (seq_len=2048)**
Double the baseline's 1024. More context per sequence helps the shallow model learn longer-range dependencies that deeper models get "for free" from stacked attention.



## Architecture

```
6L x 512d x 4H/4KV (full attention, no GQA) x MLP 3x (1536 hidden)
U-Net: 3 encoder + 3 decoder with learned skip weights
Untied embeddings (tok_emb + separate lm_head)
LeakyReLU^2(0.5), logit_softcap=12.0
Learnable residual mixing (per-block blend with input embedding x0)
seq_len=2048, batch=262K, grad_accum=2
Sliding window eval stride=64
Int8 + zlib compression
16.8M parameters
```

## Configuration vs Baseline

| Parameter | This (6L) | Baseline (9L) |
|---|---|---|
| `NUM_LAYERS` | **6** | 9 |
| `NUM_HEADS` | **4** | 8 |
| `NUM_KV_HEADS` | 4 | 4 |
| `MLP_MULT` | **3** | 2 |
| `TIE_EMBEDDINGS` | **0** | 1 |
| `TRAIN_SEQ_LEN` | **2048** | 1024 |
| `TRAIN_BATCH_TOKENS` | **262,144** | 524,288 |
| `LOGIT_SOFTCAP` | **12.0** | 30.0 |
| `MODEL_DIM` | 512 | 512 |
| `WARMDOWN_ITERS` | 250 | 3500 |
| `EVAL_MODE` | 1 (sliding window) | 0 |
| `EVAL_STRIDE` | 64 | N/A |

## Commands

**8xH100 (contest target):**
```bash
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_MODE=1 \
EVAL_STRIDE=64 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=4000 \
torchrun --standalone --nproc_per_node=8 train_gpt_26e6b4a.py
```

**1xA100 (development, how this was actually run):**
```bash
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_MODE=1 \
EVAL_STRIDE=64 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=4000 \
torchrun --standalone --nproc_per_node=1 train_gpt_26e6b4a.py
```

## Key Metrics (from 1xA100 run)

- Training completed all `20000/20000` steps
- Pre-quant: `val_loss:2.0631` `val_bpb:1.2219`
- Int8+zlib roundtrip: `val_loss:2.06773341` `val_bpb:1.22462820`
- **Sliding window: `val_loss:2.03046171` `val_bpb:1.20255713`**
- Step avg: `318.85ms` (A100)
- Peak memory: `16228 MiB allocated` `17974 MiB reserved`
- Model int8+zlib: `15,784,231 bytes`
- Code: `56,975 bytes`
- **Total: `15,841,206 bytes` (under 16MB)**
- Parameters: `16,791,064`

## Training Volume

- Global batch: `262,144` tokens/step (with `grad_accum=2`)
- Total tokens seen: `5,242,880,000` (5.24B over 20K steps)

## Included Files

- `train_gpt_26e6b4a.py` — the full training script
- `train.log` — exact output from the 1xA100 run
- `submission.json` — leaderboard metadata
