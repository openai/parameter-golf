# Stacked Hyperparameter Tuning + Eval2048

**Track:** Non-record (RTX 5090, 20 train shards)

**val_bpb: 1.3358** (post-quant int8+zlib) | **15.8MB artifact**

## TL;DR

Ran 40+ experiments overnight via an autoresearch loop. Didn't touch the architecture. Just found that the baseline ships with a broken LR schedule and 5 config fixes stack to -0.027 bpb.

## The broken warmdown

`WARMDOWN_ITERS=1200` at 600s wallclock means the cosine warmdown fires from step 1. At ~620ms/step you get ~968 steps total, so 1200 > total steps and the LR collapses immediately. Setting it to 3000 fixes this.

(PRs #48 and #73 flagged the same thing independently.)

## What stacked

| Change | Effect |
|--------|--------|
| WARMDOWN_ITERS=3000 | ~-0.010 bpb |
| MATRIX_LR=0.06 (up from 0.04) | -0.002 |
| LOGIT_SOFTCAP=15 (down from 30) | -0.001 |
| MUON_MOMENTUM=0.99 (up from 0.95) | -0.005 |
| TRAIN_BATCH_TOKENS=131072 (quarter-batch, 4x more steps) | -0.010 |
| EVAL_SEQ_LEN=2048 (train 1024, eval 2048) | -0.015 |

Total: 1.362 → 1.336. All env-var changes, zero code modifications needed for these.

## What didn't work

Tested a bunch of stuff that turned out to be dead ends (on this hardware/budget):

- **6 alternative shapes** (6x624, 8x544, 12x448, 6x768, 4x768, 6x648 kv2). None beat 9x512.
- **TRAIN_SEQ_LEN=512.** 2x more steps but shorter context hurts. Worse across the board.
- **Butterfly/Monarch MLP factorization.** Compresses beautifully (7MB artifact at 9x512) but 1.46 bpb. The factored matrices can't match dense quality in 600s of training. Tested at dim 512, 768, 1024, 1536.
- **Reservoir random MLPs** (random weights from seed, not stored, rank-4 LoRA adapters). 2.14 bpb. The random projections are just too weak.
- **Depth recurrence** (4 unique blocks x 3 passes = 12 effective). 1.50 bpb with butterfly. Ran into serialization bugs with dense recurrence (found and fixed mid-run, then ran out of compute before retesting).

## MLP3 + int6: promising, needs more time

First test of MLP_MULT=3 with int6 quantization (step_size=4) on all block matrices: **13.7MB artifact** with 2.3MB to spare. val_bpb 1.376, which is worse than baseline, but the model was undertrained (412ms/step = only ~1450 steps vs 2800 for MLP2). This direction probably works with more compute or a smarter batch schedule.

## Config

```bash
RUN_ID=stacked_best \
WARMDOWN_ITERS=3000 \
MATRIX_LR=0.06 \
LOGIT_SOFTCAP=15.0 \
MUON_MOMENTUM=0.99 \
TRAIN_BATCH_TOKENS=131072 \
EVAL_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Numbers

| Metric | Value |
|--------|-------|
| val_bpb (post-quant) | **1.3358** |
| Steps | ~2800 |
| Step avg | ~210ms |
| Wallclock | 600s |
| Peak GPU mem | 2.7 GB |
| Artifact (int8+zlib) | 15,794,837 bytes |

## Hardware

RTX 5090, RunPod community cloud ($0.69/hr). 20 train shards, not the full 80. This is a dev result, not a leaderboard submission.

## How

Autoresearch loop inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Each experiment: train 600s → parse val_bpb + size → keep if better, discard if worse. Ran 40+ experiments across 8 batches over ~12 hours. The loop scripts and experiment runners are in the repo.

## What's next (blocked on compute)

1. Sliding-window eval (stride-512). Implemented, tested, timed out at stride-64 (too many windows). stride-512 should work but the pod ran out of credits mid-experiment.
2. MLP3 + int6 + sliding eval. The combo that got PR #70 to 1.166.
3. 8xH100 verification run with 80 shards for a proper leaderboard submission.

## Files

- `README.md`
- `submission.json`
- `train_gpt.py` (modified: EVAL_SEQ_LEN decoupling, alias-aware quant, int6 support, sliding eval, depth recurrence)
- `train.log`
