# ArjunAutoResearch

Hey there! This was super fun. I took the approach of building an AutoResearch agent harness that could work towards solving this autonomously. ArjunAutoResearch ended up with a final val_bpb of **1.18418 +/- 0.00075** (mean across 3 seeds, p = 8e-5 << 0.01).

Built an auto-research pipeline:

1. Gave an agent the GitHub CLI
2. Asked it to go through all of the open PRs, describe what the approach being taken was and bucket it by expected impact on BPB (High, Medium or Low)
3. The agent then composed the approaches in the "High" bucket
4. Iteratively trying to figure out which approaches built on top of each other well and led to the best score

## How I found the right config

The agent pulled every PR diff with `gh pr diff` and sorted them into three buckets:

- **High impact** (verified on 8xH100): sliding window eval (PR #50, +0.032), long-context training with seq_len=4096 (PR #52, +0.02), fp16 tied embedding export (PR #42, +0.007)
- **Medium** (promising but untested at scale): QAT, mixed-precision layers
- **Low**: warmdown-only fixes, depth recurrence

From there it stacked the high-impact changes and worked out which ones played nicely together. A couple gotchas came up along the way: QAT turned out to be redundant once the quant gap was already near zero, and `eval_batch_seqs=1024` OOMs at seq_len=4096. Those got caught and fixed across two runs.

With more compute I'd scale this by having the agent also try the medium/low bucket combos and research approaches from the internet that nobody's PR'd yet.

## What the agent ended up with

**1. Longer training context (`TRAIN_SEQ_LEN=4096`)**
4x more context per sequence than the baseline's 1024. Each token attends to much richer history, which significantly improves convergence quality per step. Steps are slower (~60ms vs ~44ms) but the quality gain more than makes up for it.

**2. Optimizer tuning** (from PR #52)
- `MUON_MOMENTUM=0.99` (vs 0.95): stronger gradient smoothing
- Learning rates halved across the board, which alone cuts the post-quant BPB gap from ~0.007 to ~0.003
- Batch tokens down to 393K (3/4 of default): more updates per wallclock second
- Warmdown stretched to 3000 steps, momentum warmup to 1500

**3. fp16 tied embedding export** (from PR #42)
The tied embedding doubles as the output head, the most quantization-sensitive tensor. Keeping it in fp16 nearly eliminates the post-quant BPB gap. `MLP_HIDDEN=992` (vs 1024) compensates for the ~500KB size increase to stay under 16MB.

**4. Sliding window evaluation** (from PR #50, extended to seq_len=4096)
Each token scored with up to 4032 tokens of context instead of the baseline's average ~512. Windows advance by 64 tokens; only the rightmost 64 are scored per window so every token is evaluated exactly once. This is the single biggest free win, pure eval strategy, no training changes needed. The 4096-token context here provides substantially more signal than PR #50's 1024-token version.

## Config

```
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_HIDDEN=992  TIE_EMBEDDINGS=1  TRAIN_SEQ_LEN=4096  TRAIN_BATCH_TOKENS=393216
MATRIX_LR=0.02  SCALAR_LR=0.02  TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99  MUON_MOMENTUM_WARMUP_STEPS=1500  MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000  QAT=0  VAL_LOSS_EVERY=0
EVAL_STRIDE=64  EVAL_BATCH_SEQS=128
```

## Run Command

```bash
RUN_ID=arjun_autoresearch \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_HIDDEN=992 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_ArjunAutoResearch/train_gpt.py
```

All other hyperparameters are baked into the script as defaults.

## Results

Final score: **val_bpb = 1.18335372** (mean across seeds: 1.18418, p = 8e-5 << 0.01).
Artifact size: **15,932,938 bytes** (under 16MB).

Here's how it stacks up against the individual PRs it drew from:

| Run | BPB | vs baseline |
|-----|-----|-------------|
| Naive Baseline | 1.2244 | - |
| 4-Hour Baseline (unlimited compute) | 1.2074 | -0.017 |
| PR #52 (optimizer + seq4096) | 1.2014 | -0.023 |
| PR #50 (sliding window, seq1024) | 1.1925 | -0.032 |
| PR #53 (SP-4096 + sliding window) | 1.1888 | -0.036 |
| **This submission** | **1.1834** | **-0.041** |

### Key numbers (from `train.log`)

- Stopped at step 9919/20000 (wallclock cap)
- Pre-quant val_bpb: 1.1947
- Post-quant sliding window val_bpb: **1.18335372**
- Train time: 601s at 60.65ms/step
- Eval time: 278s (within the separate 10-min eval budget)
- Peak memory: 7712 MiB

### Statistical significance

Three seeds, all within the standard budget. Threshold to beat: **1.2194** (current SOTA − 0.005).

| Seed | val_bpb |
|------|---------|
| 1337 | 1.18335372 |
| 1338 | 1.18437368 |
| 1339 | 1.18481782 |

Mean: 1.18418, std: 0.00075. One-sample t-test against threshold: **t = 81.26** (df=2, critical value 6.965). p << 0.001.

## Files

- `train_gpt.py`: the script, with all settings above as defaults
- `train.log`: seed 1337 (canonical run)
- `train_seed1338.log`, `train_seed1339.log`: reproducibility reruns
- `submission.json`: leaderboard metadata
