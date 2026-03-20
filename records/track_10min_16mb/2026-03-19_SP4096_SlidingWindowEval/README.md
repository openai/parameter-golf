This record captures `SP-4096 + Sliding Window Eval`.

Two independent improvements stacked multiplicatively via the BPB formula `(loss / ln2) * (tokens/bytes)`:

1. **SP-4096 tokenizer**: Encodes ~3.32 bytes/token vs ~2.44 for SP-1024, reducing the tokens/bytes multiplier from 0.41 to 0.30 (26.6% compression advantage).
2. **Sliding window evaluation**: Stride-64 overlapping windows give each token ~960 context tokens instead of ~512 average with non-overlapping chunks, lowering per-token cross-entropy.

Trainer changes in this snapshot:
- Switched to `fineweb10B_sp4096` dataset and `fineweb_4096_bpe.model` tokenizer
- Reduced to 8 layers (from 9) to fit within 16MB after int8+zlib with the larger embedding table
- Added `forward_logits()` method for efficient sliding window inference
- Added `eval_val_sliding()` function with distributed multi-GPU support
- Changed quantization to use `amax` (from `quantile`) for both fake-quant and real quant
- Periodic validation disabled during training (`VAL_LOSS_EVERY=0`) to maximize training steps

Configuration:
- Layout: `VOCAB_SIZE=4096 NUM_LAYERS=8 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Eval: `EVAL_STRIDE=64 EVAL_BATCH_SEQS=1024` (sliding window)

Command (track-relevant params):
```bash
NCCL_IB_DISABLE=1 \
DATA_PATH=/data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=/data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
NUM_LAYERS=8 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=1024 \
QAT_START_FRAC=1.1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `14006/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.7763`, `val_bpb:1.2067`
- Post-quant sliding window eval: `val_loss:2.7351`, `val_bpb:1.1888`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_loss:2.73510678 val_bpb:1.18883084`
- Train time: `600062ms` (`step_avg:42.84ms`)
- Sliding window eval time: `52977ms`
- Peak memory: `9367 MiB allocated`, `10052 MiB reserved`
- Serialized model int8+zlib: `15629714 bytes`
- Code size: `54244 bytes`
- Total submission size int8+zlib: `15683958 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `7341080576`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
