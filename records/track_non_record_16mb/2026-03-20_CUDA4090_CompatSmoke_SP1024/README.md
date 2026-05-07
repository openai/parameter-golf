This record captures a finished non-record smoke submission built from the current root `train_gpt.py`, with a small CUDA compatibility patch in the local copy used for the run.

This run is not intended for the 10-minute leaderboard. It is a short, fully completed non-record baseline on a single RTX 4090 using the fixed full FineWeb validation split and a single training shard. The main purpose is to document a clean, reproducible CUDA submission path with final metrics, artifact bytes, and logs.

## Why this script differs slightly from root

The Vast.ai image used for this run shipped with a PyTorch build that does not accept the `enable_gqa=` argument on `scaled_dot_product_attention`. To keep the run reproducible on that image, the copied `train_gpt.py` expands KV heads manually when `num_kv_heads != num_heads` and then calls `scaled_dot_product_attention` without `enable_gqa`.

The model, tokenizer, data, and training setup otherwise follow the baseline configuration.

## Configuration

- Track: `non-record`, unlimited compute, still under the `16,000,000` byte artifact cap
- Hardware: `1x RTX 4090` on Vast.ai
- Tokenizer / dataset: `sp1024`, full fixed `fineweb_val_*`, `1` training shard
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=1024`
- Validation cadence: final-only validation on the full fixed validation split
- Training length: `ITERATIONS=50`

## Command

```bash
RUN_ID=stukenov_4090_smoke50 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
TRAIN_BATCH_TOKENS=8192 \
TRAIN_SEQ_LEN=1024 \
VAL_BATCH_SIZE=65536 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=25 \
ITERATIONS=50 \
MAX_WALLCLOCK_SECONDS=0 \
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
python train_gpt.py
```

## Key Metrics

- Training stopped at `50/50` steps.
- Pre-quant eval at stop: `val_loss:5.3102`, `val_bpb:3.1450`
- Post-quant int8+zlib roundtrip: `val_loss:5.7139`, `val_bpb:3.3841`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_loss:5.71391837 val_bpb:3.38410431`
- Train time: `12070ms` (`step_avg:241.40ms`)
- Eval time: `28444ms`
- Peak memory: `565 MiB allocated`, `750 MiB reserved`
- Serialized model int8+zlib: `5121054 bytes`
- Code size: `47999 bytes`
- Total submission size int8+zlib: `5169053 bytes`

## Included Files

- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact training log)
- `submission.json` (metadata for this non-record run)
