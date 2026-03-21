This submission captures a non-record 10-minute track run using a shared-loop recurrent transformer built from the current root `train_gpt.py`.

This run is intended to satisfy the 10-minute constraint while maintaining strong convergence using a compact recurrent architecture. It uses a 512-dim model with looped layers and evaluates against the standard FineWeb SP1024 validation split.

Configuration:
- Track: `non-record-10min-16mb`
- Layout: `VOCAB_SIZE=1024 MODEL_DIM=512 NUM_LOOP_ITERS=3 MIN_LOOP_ITERS=1`
- Tokenizer: SentencePiece (`fineweb_1024_bpe.model`)
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Validation cadence: `VAL_LOSS_EVERY=20`
- Iterations: `6000`
- Hardware: `8x H100`

Command (track-relevant params):
```bash
RUN_ID=recurrent_512_l3_6k \
MODEL_DIM=512 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=20 \
ITERATIONS=6000 \
NUM_LOOP_ITERS=3 \
MIN_LOOP_ITERS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train.log