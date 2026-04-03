# 07c1 Base Reproduction of PR #1212 with Strict RunPod Proof

**val_bpb: 1.1101** (4-seed mean sliding s64) | **15.73 MB** max artifact | **8xH100 SXM**, **598.1s** max train time

Submitted artifact: **seed 2025** with `val_loss = 1.87233216`, `val_bpb = 1.10894203`, `bytes_total = 15,728,840`.

## Summary

This folder packages the strict RunPod H100 SXM base proof for the `07c1` line built on the faithful PR `#1212` reproduction in `records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py`.

The claimed result here is the **base path only**. Test-time training plumbing was repaired on this line, but `TTT_ENABLED=0` for all claimed runs in this submission.

## What Changed Relative to PR #1212

The architecture and hyperparameter stack remain intentionally close to the PR `#1212` reproduction. The main changes on `07c1` are evaluation and export hygiene:

1. The final sliding evaluation path correctly dispatches to the TTT evaluator when `TTT_ENABLED=1`.
2. The TTT evaluator uses the real evaluation sequence length instead of hardcoding `train_seq_len`.
3. `brotli` is a hard requirement for export/reload, removing the old silent `brotli`/`lzma` mismatch.
4. The claimed record candidate here keeps `TTT_ENABLED=0`, giving a clean base-only proof.

## Strict Multi-Seed Results

All four claimed runs were launched with `MAX_WALLCLOCK_SECONDS=598` on RunPod `8xH100 80GB HBM3 / SXM` and stayed under the nominal `600s` budget.

| Seed | Steps | train_time | Sliding s64 BPB | Roundtrip exact BPB | val_loss (nats) | bytes_total |
|------|------:|-----------:|----------------:|--------------------:|----------------:|------------:|
| 42 | 8343 | 598053ms | 1.11158737 | 1.12108524 | 1.87679853 | 15,723,725 |
| 1337 | 8338 | 598065ms | 1.10898094 | 1.11850508 | 1.87239786 | 15,722,146 |
| 2025 | 8342 | 598052ms | **1.10894203** | **1.11863096** | **1.87233216** | 15,728,840 |
| 7 | 8352 | 598027ms | 1.11070811 | 1.12027614 | 1.87531400 | 15,731,988 |
| **Mean** | **8343.75** | **598049ms** | **1.11005461** | **1.11962435** | **1.87421064** | **15,726,674.75** |
| **Std** | | | **0.00131238** | | **0.00221581** | |

## Record-Bar Comparison

Merged official `#1019` reports:

- `1.88217853` nats
- `1.11473509` BPB

This strict 4-seed proof gives:

- mean `1.87421064` nats / `1.11005461` BPB
- delta vs merged `#1019`: `-0.00796789` nats / `-0.00468048` BPB
- delta vs official record bar (`1.87717853` nats): `-0.00296789` nats
- Welch `t = -6.8667`
- one-sided `p = 0.001785`

Against the currently **merged** official leaderboard entry (`#1019`), this satisfies the written record conditions:

1. beats the merged SOTA by more than `0.005` nats
2. supports `p < 0.01`
3. trains under `10` minutes on `8xH100 SXM`
4. keeps every claimed artifact under `16,000,000` bytes

## Architecture and Training Stack

This keeps the reproduced PR `#1212` base stack:

- 12 transformer layers, 512 model dim, 8 attention heads, 4 KV heads
- GQA attention with 5 windowed attention layers (`2,4,6,8,10`)
- leaky ReLU squared MLP with slope `0.5`
- tied embeddings
- SentencePiece tokenizer (`1024` vocab)
- per-row int6 export with Brotli compression
- mixed per-GPU sequence packing for training throughput:
  - GPUs `0-4`: `36 x 2048`
  - GPUs `5-7`: `10 x 6144`

Fixed base hyperparameters used in all claimed strict runs:

- `TRAIN_BATCH_TOKENS=589824`
- `TRAIN_SEQ_LEN=2048`
- `EVAL_SEQ_LEN=6144`
- `EVAL_STRIDE=64`
- `WARMDOWN_ITERS=4000`
- `MATRIX_LR=0.024`, `MATRIX_LR_LATE=0.019`
- `SCALAR_LR=0.020`, `SCALAR_LR_LATE=0.038`
- `TIED_EMBED_LR=0.022`
- `MUON_MOMENTUM=0.985`
- `WINDOW_SIZE=512`
- `BIGRAM_VOCAB_SIZE=5120`
- `VE_DIM=128`

## Environment

The exact strict proof environment is preserved in [runpod_env.txt](runpod_env.txt). The relevant lines are:

- `NVIDIA H100 80GB HBM3` x 8
- driver `580.126.09`
- `CUDA Version: 13.0` in `nvidia-smi`
- `python=3.12.3`
- `torch=2.9.1+cu128`
- `torch_cuda=12.8`
- `sentencepiece=0.2.1`
- `brotli=1.2.0`

## How to Run

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

`flash_attn_interface` (FA3) must be available as a container-bundled package (present in the `torch 2.9.1+cu128` RunPod image at `/usr/local/lib/python3.12/dist-packages/flash_attn_interface.py`). It is **not** the PyPI `flash-attn` package and cannot be installed via pip. Triton is bundled with torch; the fused MLP kernel falls back gracefully if it is absent.

Then run the following strict base command **from the repo root** (so that `./data/` resolves correctly):

```bash
PYTHONUNBUFFERED=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MATRIX_LR=0.024 \
MATRIX_LR_LATE=0.019 \
SCALAR_LR=0.020 \
SCALAR_LR_LATE=0.038 \
TIED_EMBED_LR=0.022 \
MUON_MOMENTUM=0.985 \
WARMDOWN_ITERS=4000 \
GPTQ_RESERVE_MS=0 \
NUM_LAYERS=12 \
BIGRAM_VOCAB_SIZE=5120 \
VE_DIM=128 \
WINDOW_SIZE=512 \
WINDOW_ATTN_LAYERS=2,4,6,8,10 \
QK_GAIN_INIT=2.5 \
RUN_ID=submission \
SEED=2025 \
TRAIN_BATCH_TOKENS=589824 \
MAX_WALLCLOCK_SECONDS=598 \
SEQ_LENS_PER_GPU=2048,2048,2048,2048,2048,6144,6144,6144 \
SEQS_PER_GPU=36,36,36,36,36,10,10,10 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=6144 \
EVAL_STRIDE=64 \
TTT_ENABLED=0 \
MASTER_PORT=29500 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py`: exact script snapshot used for the strict RunPod proof
- `train_seed42.log`
- `train_seed1337.log`
- `train_seed2025.log`
- `train_seed7.log`
- `runpod_env.txt`

The claimed train logs are the clean `*.console.log` copies from the pod, renamed into the submission folder.
