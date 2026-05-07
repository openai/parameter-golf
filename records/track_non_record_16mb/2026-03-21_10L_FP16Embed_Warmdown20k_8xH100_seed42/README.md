This is a non-record derivative submission built on the merged `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` baseline.

Main change:
- increase `WARMDOWN_ITERS` from `2500` to `20000`

The model stack is otherwise the same 10-layer FP16-embed sliding-window setup. The included `train_gpt.py` is the exact script snapshot used for the run; besides the warmdown change, it also includes the SDPA GQA compatibility fallback needed on the PyTorch 2.4 image used for this run.

This is not a SOTA claim. It is a verified `8x H100` non-record submission that slightly improves the merged seed-42 baseline while staying under the `16,000,000` byte cap.

Key metrics:
- `post-roundtrip val_bpb: 1.17389939`
- `pre-roundtrip val_bpb: 1.2055`
- `artifact bytes: 14122782`
- `total submission bytes: 14178772`
- `step_stop: 10919`
- `ms/step: 54.96`
- `eval_time: 57969 ms`

Comparison to the merged seed-42 baseline:
- merged seed-42: `1.17423973`
- this run: `1.17389939`
- delta: `-0.00034034`

Command (same params as the recorded run, from repo root):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=pivot_wd20k_8xh100_seed42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=42 \
NUM_LAYERS=10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.10 \
TIED_EMBED_INIT_STD=0.005 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
MUON_BACKEND_STEPS=5 \
TRAIN_BATCH_TOKENS=524288 \
VAL_BATCH_SIZE=524288 \
TRAIN_SEQ_LEN=1024 \
EVAL_STRIDE=64 \
ITERATIONS=20000 \
WARMDOWN_ITERS=20000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-03-21_10L_FP16Embed_Warmdown20k_8xH100_seed42/train_gpt.py
```
