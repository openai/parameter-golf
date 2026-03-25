# ArjunAutoResearch

Hey there! This was super fun. I took the approach of building an AutoResearch agent harness that could work towards solving this autonomously. ArjunAutoResearch ended up with a final val_bpb of **1.16323 +/- 0.00042** (mean across 3 seeds, p << 0.001).

Built an auto-research pipeline:

1. Gave an agent the GitHub CLI
2. Asked it to go through all of the open PRs, describe what the approach being taken was and bucket it by expected impact on BPB (High, Medium or Low)
3. The agent then composed the approaches in the "High" bucket
4. Iteratively trying to figure out which approaches built on top of each other well and led to the best score

## How I found the right config

The agent pulled every PR diff with `gh pr diff` and sorted them into three buckets:

- **High impact** (verified on 8xH100): sliding window eval (PR #50, +0.032), long-context training with seq_len=4096 (PR #52, +0.02), fp16 tied embedding export (PR #42, +0.007), int6 quantization + wider MLP (PR #70, +0.019), STE fake int6 QAT (PR #65, -0.003 quant penalty)
- **Medium** (promising but untested at scale): QAT, mixed-precision layers
- **Low**: warmdown-only fixes, depth recurrence

From there it stacked the high-impact changes and worked out which ones played nicely together.

With more compute I'd scale this by having the agent also try the medium/low bucket combos and research approaches from the internet that nobody's PR'd yet.

## What the agent ended up with

**1. Wider MLP (`MLP_MULT=3.0`, hidden=1536)** (from PR #70)
3x MLP expansion (1536 hidden vs baseline's 1024). Enabled by int6 quantization saving ~4MB of artifact space. 1536 is 64-aligned for optimal H100 matmul tile utilization.

**2. Longer training context (`TRAIN_SEQ_LEN=4096`)** (from PR #65)
4x more context per sequence than the baseline's 1024. Each token attends to much richer history, significantly improving convergence quality per step.

**3. Optimizer tuning** (from PR #52)
- `MUON_MOMENTUM=0.99` (vs 0.95): stronger gradient smoothing
- Learning rates halved across the board
- Batch tokens down to 393K (3/4 of default): more updates per wallclock second
- Warmdown stretched to 3000 steps, momentum warmup to 1500

**4. STE fake int6 quantization-aware training** (from PR #65)
During training, all `CastedLinear` weights get fake int6 quantization via Straight-Through Estimator: the forward pass uses quantized weights while gradients flow through the originals. This teaches the model weight distributions that survive int6 post-training quantization, dramatically reducing the quantization penalty. The token embedding (`nn.Embedding`) is unaffected, preserving the fp16 passthrough benefit.

**5. int6 per-row quantization on MLP+attention** (from PR #70)
Mixed precision quantization: int6 ([-32, 31] range) on all 2D MLP and attention weights, fp16 passthrough on the tied embedding, fp16/fp32 passthrough on small/control tensors. Values stored in int8 bytes; zstd-22 compresses the zero high bits efficiently.

**6. fp16 tied embedding passthrough** (from PR #42)
The tied embedding doubles as the output head, the most quantization-sensitive tensor. Keeping it in fp16 eliminates embedding quantization penalty entirely.

**7. Sliding window evaluation** (from PR #50, extended to seq_len=4096)
Each token scored with up to 4032 tokens of context instead of the baseline's average ~512. Windows advance by 64 tokens; only the rightmost 64 are scored per window so every token is evaluated exactly once. Compiled `forward_logits` for fast eval.

## Config

All hyperparameters are baked into the script as defaults. No env var overrides are needed.

```
# Model
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=3.0  TIE_EMBEDDINGS=1

# Training
TRAIN_SEQ_LEN=4096  TRAIN_BATCH_TOKENS=393216  MAX_WALLCLOCK_SECONDS=600

# Optimizer
MATRIX_LR=0.02  SCALAR_LR=0.02  TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99  MUON_MOMENTUM_WARMUP_STEPS=1500  MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000

# Eval (sliding window)
EVAL_STRIDE=64  EVAL_BATCH_SEQS=16
```

## How to reproduce

1. Clone the repo and download the SP-1024 dataset:

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
```

2. Run training + evaluation (all defaults are baked in, no env overrides required):

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_ArjunAutoResearch/train_gpt.py
```

Training stops at the 10-minute wallclock cap. Sliding window evaluation runs automatically afterward (~3 min). The final `final_int6_zstd_roundtrip_exact` line in the log is the submission score.

To run with a different seed: `SEED=1338 torchrun ...`

## Results

Final score: **val_bpb = 1.16356083** (mean across seeds: 1.16323, p << 0.001).
Artifact size: **15,265,243 bytes** (under 16,000,000).

### Key numbers (from `train.log`)

- Stopped at step 9211/20000 (wallclock cap)
- Pre-quant val_bpb: 1.1769
- Post-quant sliding window val_bpb: **1.16356083**
- Train time: 600s at 65.15ms/step
- Eval time: 157s (within the separate 10-min eval budget)
- Peak memory: 8521 MiB

### Statistical significance

Three seeds, all within the standard budget. Threshold to beat: **1.2194** (current SOTA - 0.005).

| Seed | val_bpb |
|------|---------|
| 1337 | 1.16356083 |
| 1338 | 1.16275343 |
| 1339 | 1.16337225 |

Mean: 1.16323, std: 0.00042. One-sample t-test against threshold: **t = 230.34** (df=2, critical value 6.965). p << 0.001.

## Files

- `train_gpt.py`: the script, with all settings above as defaults
- `train.log`: seed 1337 (canonical run)
- `train_seed1338.log`, `train_seed1339.log`: reproducibility reruns
- `submission.json`: leaderboard metadata
