# ArjunAutoResearch

Hey there! This was super fun. I took the approach of building an AutoResearch agent harness that could work towards solving this autonomously. ArjunAutoResearch ended up with a final val_bpb of **1.16520 +/- 0.00102** (mean across 3 seeds, p << 0.001).

Built an auto-research pipeline:

1. Gave an agent the GitHub CLI
2. Asked it to go through all of the open PRs, describe what the approach being taken was and bucket it by expected impact on BPB (High, Medium or Low)
3. The agent then composed the approaches in the "High" bucket
4. Iteratively trying to figure out which approaches built on top of each other well and led to the best score

## How I found the right config

The agent pulled every PR diff with `gh pr diff` and sorted them into three buckets:

- **High impact** (verified on 8xH100): sliding window eval (PR #50, +0.032), long-context training with seq_len=4096 (PR #52, +0.02), fp16 tied embedding export (PR #42, +0.007), int6 quantization + wider MLP (PR #70, +0.019)
- **Medium** (promising but untested at scale): QAT, mixed-precision layers
- **Low**: warmdown-only fixes, depth recurrence

From there it stacked the high-impact changes and worked out which ones played nicely together. A couple gotchas came up along the way: QAT turned out to be redundant once the quant gap was already near zero, and MLP_HIDDEN=992 caused H100 matmul tile misalignment costing ~8ms/step.

With more compute I'd scale this by having the agent also try the medium/low bucket combos and research approaches from the internet that nobody's PR'd yet.

## What the agent ended up with

**1. Wider MLP (`MLP_MULT=3.0`, hidden=1536)** (from PR #70)
3x MLP expansion (1536 hidden vs baseline's 1024). Enabled by int6 quantization saving ~4MB of artifact space. Provides ~0.019 BPB improvement from increased model capacity. 1536 is 64-aligned for optimal H100 matmul tile utilization.

**2. Longer training context (`TRAIN_SEQ_LEN=4096`)** (from PR #65)
4x more context per sequence than the baseline's 1024. Each token attends to much richer history, which significantly improves convergence quality per step. Steps are ~64ms (vs ~48ms at seq1024) but the quality gain more than compensates.

**3. Optimizer tuning** (from PR #52)
- `MUON_MOMENTUM=0.99` (vs 0.95): stronger gradient smoothing
- Learning rates halved across the board
- Batch tokens down to 393K (3/4 of default): more updates per wallclock second
- Warmdown stretched to 3000 steps, momentum warmup to 1500

**4. int6 per-row quantization on MLP+attention** (from PR #70)
Mixed precision quantization: int6 ([-32, 31] range) on all 2D MLP and attention weights, fp16 passthrough on the tied embedding, fp16/fp32 passthrough on small/control tensors. Values stored in int8 bytes (lazy packing); zstd-22 compresses the zero high bits efficiently.

**5. fp16 tied embedding passthrough** (from PR #42)
The tied embedding doubles as the output head, the most quantization-sensitive tensor. Keeping it in fp16 saves ~0.007 BPB over int8 quantization. With int6+zstd saving ~4MB, the ~523KB cost of fp16 tok_emb is easily affordable.

**6. Sliding window evaluation** (from PR #50, extended to seq_len=4096)
Each token scored with up to 4032 tokens of context instead of the baseline's average ~512. Windows advance by 64 tokens; only the rightmost 64 are scored per window so every token is evaluated exactly once. Compiled `forward_logits` for fast eval.

## Config

```
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=3.0  TIE_EMBEDDINGS=1  TRAIN_SEQ_LEN=4096  TRAIN_BATCH_TOKENS=393216
MATRIX_LR=0.02  SCALAR_LR=0.02  TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99  MUON_MOMENTUM_WARMUP_STEPS=1500  MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000  VAL_LOSS_EVERY=0
EVAL_STRIDE=64  EVAL_BATCH_SEQS=16
```

## Run Command

```bash
RUN_ID=arjun_autoresearch \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_ArjunAutoResearch/train_gpt.py
```

All other hyperparameters are baked into the script as defaults.

## Results

Final score: **val_bpb = 1.16614760** (mean across seeds: 1.16520, p << 0.001).
Artifact size: **15,619,929 bytes** (under 16,000,000).

Here's how it stacks up:

| Run | BPB | vs baseline |
|-----|-----|-------------|
| Naive Baseline | 1.2244 | - |
| 4-Hour Baseline (unlimited compute) | 1.2074 | -0.017 |
| PR #52 (optimizer + seq4096) | 1.2014 | -0.023 |
| PR #50 (sliding window, seq1024) | 1.1925 | -0.032 |
| PR #53 (SP-4096 + sliding window) | 1.1888 | -0.036 |
| PR #70 (MLP 3x + int6 + sliding) | 1.1659 | -0.059 |
| **This submission** | **1.1652** | **-0.059** |

### Key numbers (from `train.log`)

- Stopped at step 9370/20000 (wallclock cap)
- Pre-quant val_bpb: 1.1715
- Post-quant sliding window val_bpb: **1.16614760**
- Train time: 600s at 64.03ms/step
- Eval time: 170s (within the separate 10-min eval budget)
- Peak memory: 8519 MiB

### Statistical significance

Three seeds, all within the standard budget. Threshold to beat: **1.2194** (current SOTA - 0.005).

| Seed | val_bpb |
|------|---------|
| 1337 | 1.16614760 |
| 1338 | 1.16531758 |
| 1339 | 1.16412094 |

Mean: 1.16520, std: 0.00102. One-sample t-test against threshold: **t = 92.15** (df=2, critical value 6.965). p << 0.001.

## Files

- `train_gpt.py`: the script, with all settings above as defaults
- `train.log`: seed 1337 (canonical run)
- `train_seed1338.log`, `train_seed1339.log`: reproducibility reruns
- `submission.json`: leaderboard metadata
