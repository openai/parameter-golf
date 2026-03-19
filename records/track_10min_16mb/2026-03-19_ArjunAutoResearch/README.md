# ArjunAutoResearch

I used an LLM agent to read through all ~50 open PRs on this repo, sort the techniques by expected impact, combine the best ones, and iterate until the score stopped improving. The agent handled the full pipeline: research, coding, debugging, running experiments, and packaging the submission.

## How I found the right config

The agent pulled every PR diff with `gh pr diff` and sorted techniques into three buckets:

- **High impact** (verified on 8xH100): sliding window eval (PR #50, +0.032), long-context training with seq_len=4096 (PR #52, +0.02), fp16 tied embedding export (PR #42, +0.007)
- **Medium** (promising but untested at scale): QAT, mixed-precision layers
- **Low**: warmdown-only fixes, depth recurrence

It then stacked the high-impact changes, caught a few interactions that would have hurt (QAT is redundant once the quant gap is already near zero; `eval_batch_seqs=1024` OOMs at seq_len=4096), and fixed them across two runs.

## What changed from the baseline

**1. Train on longer sequences (`TRAIN_SEQ_LEN=4096`)**
Each sequence is 4x longer than the baseline, so the model sees much richer context during training. Steps are slower (~60ms vs ~44ms), but the quality gain more than makes up for it.

**2. Tuned the optimizer**
- Muon momentum up to 0.99 (from 0.95) — smoother gradient updates
- Learning rates halved across the board — this alone cuts the post-quantization BPB gap from ~0.007 to ~0.003
- Batch tokens reduced to 393K (3/4 of default) — more optimizer steps per minute
- Warmdown stretched to 3000 steps to match the shorter run

**3. Keep the embedding in fp16 during export**
The tied embedding is used as both the input embedding and the output head, so it's the most sensitive tensor to quantize. Keeping it fp16 drops the quant degradation to ~0.001 BPB. Trimmed MLP hidden from 1024 to 992 to stay under 16MB.

**4. Sliding window evaluation**
Instead of scoring each token with ~512 tokens of context on average (non-overlapping 4096-token chunks), overlapping windows with stride=64 give every token up to 4032 tokens of context. Each token is scored exactly once. This is the single biggest free win — pure eval strategy, no training changes needed.

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

| Run | BPB | vs baseline |
|-----|-----|-------------|
| Naive Baseline | 1.2244 | — |
| 4-Hour Baseline (unlimited compute) | 1.2074 | -0.017 |
| PR #52 (optimizer + seq4096) | 1.2014 | -0.023 |
| PR #50 (sliding window, seq1024) | 1.1925 | -0.032 |
| PR #53 (SP-4096 + sliding window) | 1.1888 | -0.036 |
| **This submission** | **1.1834** | **-0.041** |

## Key Numbers (from `train.log`)

- Stopped at step 9919/20000 (wallclock cap)
- Pre-quant val_bpb: 1.1947
- Post-quant sliding window val_bpb: **1.18335372**
- Train time: 601s at 60.65ms/step
- Eval time: 278s (within the separate 10-min eval budget)
- Peak memory: 7712 MiB
- Artifact: 15,879,361 bytes model + 53,577 bytes code = **15,932,938 bytes total**

## Statistical Significance

Three seeds, all within the standard budget. Threshold to beat: **1.2194** (current SOTA − 0.005).

| Seed | val_bpb |
|------|---------|
| 1337 | 1.18335372 |
| 1338 | 1.18437368 |
| 1339 | 1.18481782 |

Mean: 1.18418, std: 0.00075. One-sample t-test against threshold: **t = 81.26** (df=2, critical value 6.965). p << 0.001.

## Files

- `train_gpt.py` — the script, with all settings above as defaults
- `train.log` — seed 1337 (canonical run)
- `train_seed1338.log`, `train_seed1339.log` — reproducibility reruns
- `submission.json` — leaderboard metadata
