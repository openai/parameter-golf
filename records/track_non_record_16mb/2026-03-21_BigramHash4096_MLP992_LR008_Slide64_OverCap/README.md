This non-record submission captures a strong `8xH100` run using a CUDA variant derived from the local research branch.

It is placed in `track_non_record_16mb` because the final `int8+zlib` artifact exceeds the `16,000,000` byte cap by `179,102` bytes, even though the score is strong.

## Key Idea

This run combines a few ideas that proved useful locally and were worth validating on the target hardware:

- narrower FFN than the naive baseline via `MLP_HIDDEN=992`
- higher matrix LR via `MATRIX_LR=0.08`
- a small hashed bigram embedding side channel via `BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=64`
- sliding-window evaluation via `EVAL_STRIDE=64`
- fp16 tied embedding passthrough during export

Compared with the naive baseline, this run improves score dramatically but misses the strict artifact-size cap.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- FFN: `MLP_HIDDEN=992`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Learning rates: `TIED_EMBED_LR=0.05 MATRIX_LR=0.08 SCALAR_LR=0.04`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Bigram hash side channel: `BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=64`
- Evaluation: `EVAL_STRIDE=64 EVAL_BATCH_SEQS=1024`
- Export precision policy: `QUANT_POLICY='tok_emb.weight=fp16'`

## Command

```bash
RUN_ID=challenge_8gpu_v1 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_HIDDEN=992 \
MATRIX_LR=0.08 \
BIGRAM_HASH_BUCKETS=4096 \
BIGRAM_HASH_DIM=64 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=1024 \
QUANT_POLICY='tok_emb.weight=fp16' \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- Timed training stopped at `13333/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0114`, `val_bpb:1.1913`
- Post-quant roundtrip eval: `val_loss:2.01410605`, `val_bpb:1.19286858`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.19286858`
- Train time: `599854ms` (`step_avg:44.99ms`)
- Peak memory: `20405 MiB allocated`, `46818 MiB reserved`
- Eval time: `74988ms`
- Serialized model int8+zlib: `16120324 bytes`
- Code size: `58778 bytes`
- Total submission size int8+zlib: `16179102 bytes`

## Why Non-Record

The score is submission-worthy in spirit, but the artifact is too large for the strict `track_10min_16mb` rules:

- cap: `16000000 bytes`
- this run: `16179102 bytes`

So the remaining work is primarily compression / byte allocation, not training quality.

## Included Files

- `train_gpt.py` — standalone CUDA script snapshot used for the run
- `train.log` — exact `8xH100` log
- `submission.json`
