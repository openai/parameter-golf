# Rakı Training — Parameter Golf Submission

**val_bpb: 1.3769** (1×H100, 10 min) | **Estimated 8×H100: ~1.20**
**Artifact size: 11.66 MB** (< 16 MB limit)

## Approach: OEE-Inspired Curriculum Learning

This submission introduces **Markov-weighted curriculum learning** — a training technique inspired by Overall Equipment Effectiveness (OEE) methodology from manufacturing engineering. The core insight: not all training tokens are equally valuable, and a cheap statistical prior can identify where the model should focus its limited training budget.

### How It Works

**1. Markov Teacher (Bigram Prior)**
Before training, we build a bigram probability table from one data shard (~2 seconds). This 1024×1024 table acts as a "teacher" that knows simple token co-occurrence patterns.

**2. Hybrid Entropy × Surprise Scoring**
For each training batch, we compute a curriculum weight:
- **Surprise**: How unexpected is this token sequence according to the bigram model?
- **Entropy**: How uncertain is the bigram model at this position?
- **Hybrid score** = surprise × entropy — high score means "the transformer can learn something here that the bigram model cannot predict"

This filters out both trivial patterns (low surprise) AND noise/typos (high surprise but low entropy).

**3. Gradient Weighting**
Surprising batches receive stronger gradients (up to 1.15×), directing the model's limited 10-minute training budget toward the most learnable content. This is equivalent to an adaptive curriculum without any data filtering or reordering.

**4. EMA Stabilization**
Exponential Moving Average of weights activates at 85% of training, smoothing the final model and reducing variance from the stochastic training process.

### Why This Is Different

Most leaderboard submissions optimize the **model** (architecture, quantization, attention variants). We optimize the **training process itself** — making each gradient step more informative. This is orthogonal to architectural improvements and can be combined with any of them.

The approach comes from production engineering: in a factory with limited machine time, you prioritize the jobs with highest value-add. Similarly, with a 10-minute training cap, we prioritize the token sequences with highest learning potential.

## Run

```bash
# 1×H100 (non-record, ~1.37 BPB):
python3 patch_raki.py
RUN_ID=raki NUM_LAYERS=10 WARMDOWN_ITERS=3500 MUON_WD=0.04 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8×H100 (record attempt, ~1.20 BPB):
python3 patch_raki.py
RUN_ID=raki NUM_LAYERS=10 WARMDOWN_ITERS=3500 MUON_WD=0.04 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results (1×H100)

| Metric | Value |
|---|---|
| val_bpb (quantized) | 1.3769 |
| val_loss | 2.3248 |
| Artifact size | 11.66 MB |
| Training steps | 1052 |
| Training time | 600s (wallclock cap) |
| EMA activated | Step 892 (85%) |
| Peak memory | 11,472 MiB |

## Files

- `patch_raki.py` — Patches baseline `train_gpt.py` to add Markov curriculum + EMA
- `train_gpt.py` — Baseline (modified in-place by patcher)

## Technical Details

- **Markov table**: 1024×1024 bigram log-probabilities, built from 1 shard in ~2s
- **Curriculum weight**: `1.0 + 0.15 × normalized_hybrid_score` per batch
- **EMA**: decay=0.995, starts at 85% of wallclock
- **Base config**: 10 layers, 512 dim, 8 heads, 4 KV heads, Muon optimizer

## Background

Author: Mert Yandımata — Production Data Analyst at DAB Pumps (Italy), Management Engineering background. The OEE-to-ML mapping comes directly from experience optimizing factory production lines: availability × performance × quality metrics applied to training step efficiency.
