# 11L XSA4 + EMA + BigramHash3072 + LZMA

**val_bpb: 1.14523122** | **15.616 MB** artifact | **8xH100 80GB SXM** | **599.087s train cap**

This folder packages a valid under-10-minute 8xH100 run built from an 11-layer XSA/EMA stack that was iterated in a fork and then rerun on RunPod 8xH100 SXM hardware. It is prepared in the format expected by the repository README for a `track_10min_16mb` submission folder.

This is intended as a **non-record submission** to the `track_10min_16mb` track: the run is valid and fully compliant, but it does not claim a new leaderboard best or statistical significance over the current SOTA.

The main practical goal here was to land a fully compliant `<16,000,000` byte run while keeping most of the stronger ideas from the 11-layer XSA/EMA branch. An earlier attempt reached a slightly better `1.14303161` BPB, but missed the size cap by `54,197` bytes. Switching the export path to `int6 + lzma`, increasing `BigramHash` to `3072 x 112`, and moving warmdown to `4000` produced the valid run included here.

## Results

| Run | Status | val_loss | val_bpb | bytes_total |
|-----|--------|---------:|--------:|------------:|
| `train.log` | PASS | 1.93366982 | 1.14523122 | 15,616,435 |

## Model Summary

- 11 transformer layers, `dim=512`, `heads=8`, `kv_heads=4`
- `MLP_MULT=3.0` with `lrelu2` activation (`slope=0.5`)
- XSA enabled on the last 4 layers
- `BigramHash(3072, dim=112)`
- Partial RoPE with `ROPE_DIMS=16`
- LayerNorm scale enabled
- VE enabled with `VE_DIM=128` on layers `9,10`
- EMA enabled from step 0 with `decay=0.997`
- SWA enabled every 50 steps
- Late QAT enabled at threshold `0.15`
- Sliding-window evaluation with `stride=64`
- Quantized export uses `int6 + lzma`

## Scope

- This folder contains one valid canonical run.
- It does not make any claim of new SOTA or statistical significance.
- The packaged result is the compliant `1.14523122` BPB run under the 16MB cap.

## Reproduction

Run from inside this folder:

```bash
cd records/track_10min_16mb/2026-04-05_11L_XSA4_EMA_LZMA_Bigram3072_1.1452
RUN_ID=train \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py | tee train.log
```

The raw rolling logs are also written under `./logs/`.

## Canonical Track Configuration

```bash
RUN_ID=train \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
ITERATIONS=9000 \
WARMUP_STEPS=20 \
WARMDOWN_ITERS=4000 \
MAX_WALLCLOCK_SECONDS=599 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=250 \
MLP_ACT=lrelu2 \
MLP_LEAKY_RELU_SLOPE=0.5 \
MLP_MULT=3.0 \
BIGRAM_VOCAB_SIZE=3072 \
BIGRAM_DIM=112 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
EMA_START_STEP=0 \
EMA_EVAL_AFTER_APPLY=1 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
EVAL_SLIDING=1 \
EVAL_STRIDE=64 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
HEAD_LR=0.008 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
LATE_QAT_THRESHOLD=0.15 \
RECOMPILE_ON_LATE_QAT=1 \
MODEL_COMPRESSOR=lzma \
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- `requirements.txt`

## Compliance

- [x] Trains on 8xH100 SXM hardware
- [x] Stops within the 600s training budget (`599.087s`)
- [x] Submission size is under `16,000,000` bytes (`15,616,435`)
- [x] Includes logs and runnable code inside the record folder
- [x] Uses the repository FineWeb SP-1024 dataset/tokenizer paths
- [x] No unsupported claim of record significance
