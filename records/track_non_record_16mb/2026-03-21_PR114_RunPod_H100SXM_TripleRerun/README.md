This package submits a clean same-provider RunPod verification of the PR114 long-context selective-precision recipe as a **non-record submission**.

Honesty note:
- The best run in this package is real, under the 16,000,000-byte cap, and reproduced on `8x H100 SXM` using the official RunPod Parameter Golf template.
- As of March 21, 2026, the live public leaderboard has already moved to `1.1428`, so this package is **not** a new SOTA record submission.
- I am submitting it under `track_non_record_16mb` because it is a strong, fully reproducible under-cap result with same-provider rerun evidence, but it no longer beats the live frontier.

## Recipe

- Source trainer: `experiments/external_prs/114/train_gpt.py`
- Model family: 9-layer, width-512, SP-1024, tied embeddings, GQA
- Main changes relative to the naive baseline:
  - `TRAIN_SEQ_LEN=2048`
  - `EVAL_SEQ_LEN=2048`
  - `EVAL_STRIDE=256`
  - `MLP_HIDDEN=1536`
  - mixed selective precision export with fp16 tied embedding and late-K passthrough
- Dataset/tokenizer: `fineweb10B_sp1024`
- Training budget: `MAX_WALLCLOCK_SECONDS=600`
- Hardware used for the three reruns in this package: RunPod `8x H100 SXM`

## Best Run

`train.log` is the strongest same-provider rerun from this batch:

- Run ID: `runpod-verify-pr114-20260320-seed1338`
- Exact post-eval metric:
  - `final_sliding_window_exact val_loss:1.95079135`
  - `final_sliding_window_exact val_bpb:1.15536852`
- Timed stop:
  - `step:7430/20000`
  - `train_time:600057ms`
- Size accounting:
  - compressed model: `15,909,623` bytes
  - code: `53,572` bytes
  - total: `15,963,195` bytes

## Same-Provider Reruns

I reran the exact same config three times on the same RunPod `8x H100 SXM` provider path:

| Run ID | Seed | Exact val_bpb | Total bytes |
|---|---:|---:|---:|
| `runpod-verify-pr114-20260320-seed1337` | 1337 | 1.15862019 | 15,956,272 |
| `runpod-verify-pr114-20260320-seed1338` | 1338 | 1.15536852 | 15,963,195 |
| `runpod-verify-pr114-20260320-seed1339` | 1339 | 1.15563461 | 15,959,710 |

Against the older `1.1698` threshold, these three same-provider reruns give:

- mean exact `val_bpb`: `1.15654111`
- sample stddev: `0.00180545`
- one-sided `p`: `0.0030619658979217116`

This robustness evidence is included for completeness even though this package is submitted as non-record.

## Why Non-Record

At the time of packaging, the live public README leaderboard shows `1.1428` as the current top score. This package therefore does not satisfy the current SOTA requirement of beating the live record by at least `0.005` nats, so it should be treated as a non-record submission only.

## Command

```bash
RUN_ID=runpod-verify-pr114-20260320-seed1338 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1338 \
ITERATIONS=20000 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
TRAIN_BATCH_TOKENS=786432 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=256 \
MLP_HIDDEN=1536 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
GRAD_CLIP_NORM=0.3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py`: self-contained PR114 trainer used for all three reruns
- `train.log`: best same-provider RunPod rerun (`seed=1338`)
- `rerun_runpod_seed1337.log`: additional same-provider rerun
- `rerun_runpod_seed1339.log`: additional same-provider rerun
- `significance_runs.tsv`: compact table of the three reruns
- `submission.json`: summary metadata for this package
