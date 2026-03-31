This is a non-record 16MB submission built from the current root `train_gpt.py` with a small but consistent architecture simplification from the published baseline. I am not claiming a new leaderboard record or verified reproduction under the official 8xH100 environment. This submission is intended as a clean, self-contained non-record result showing that a shallower model can improve the capacity-speed tradeoff under a strict wallclock cap while remaining well under the 16,000,000-byte artifact limit.

Configuration:
- Track: `non-record`, 16MB artifact-limited
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=7 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied input/output embeddings: `TIE_EMBEDDINGS=1`
- Learning rates:
  - `TIED_EMBED_LR=0.05`
  - `MATRIX_LR=0.04`
  - `SCALAR_LR=0.04`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Seed for the reported run: `2026`

Motivation:
The default baseline uses 9 layers at width 512. On my 1xA100 development setup, broader or deeper variants did not help under the 600-second wallclock cap because they reduced the number of optimization steps completed before stopping. I tested deeper, shallower, and width-modified variants, and found that 7 layers gave the best capacity-speed tradeoff among the tested models.

Reported command:
```bash
RUN_ID=layers7_seed2026 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=2026 \
NUM_LAYERS=7 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
python3 train_gpt.py
