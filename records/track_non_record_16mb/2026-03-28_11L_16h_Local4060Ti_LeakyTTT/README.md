This record captures a local unlimited-compute non-record submission built from the modified root `train_gpt_stack.py` snapshot included in this folder.

This run is not intended to satisfy the 10-minute cutoff for the main leaderboard. It was trained locally on a single RTX 4060 Ti for a 16-hour wallclock cap, while still fitting under the `16,000,000` byte artifact cap.

Configuration:
- Track: `non-record`, unlimited compute, still under the `16,000,000` byte artifact cap
- Hardware: `1x RTX 4060 Ti 16GB`, Windows, local run
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- Stack: `LeakyReLU(0.5)^2`, `Parallel Muon`, `XSA last 4`, `Partial RoPE 16/64`, `Layerwise LN scale`, `Warmdown 3500`, `BigramHash 1536`, `VE128 layers 9,10`, `EMA(0.997)`, `SWA every 50`, `Legal Score-First TTT`
- Batching: `TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=2048`
- Evaluation: `EVAL_STRIDE=64`, legal score-first TTT with `chunk=32768`, `lr=0.002`, `epochs=3`

Command (track-relevant params):
```bash
RUN_ID=stack16h_base_s1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ENABLE_COMPILE=0 \
SDP_BACKEND=math \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=57600 \
TRAIN_BATCH_TOKENS=262144 \
VAL_BATCH_SIZE=262144 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
VAL_LOSS_EVERY=4000 \
TRAIN_LOG_EVERY=50 \
SEED=1337 \
python train_gpt_stack.py
```

Key metrics (from `stack16h_base_s1337.txt`):
- Timed training stopped at `7998/9000` steps due to the wallclock cap.
- In-training periodic eval at stop: `val_loss:1.9840`, `val_bpb:1.1751`
- Post-EMA diagnostic eval: `val_loss:1.9826`, `val_bpb:1.1742`
- Post-int6 roundtrip eval: `val_loss:1.99107258`, `val_bpb:1.17922533`
- Post-int6 sliding-window eval: `val_loss:1.95050821`, `val_bpb:1.15520389`
- Final legal score-first TTT eval: `val_loss:1.94810047`, `val_bpb:1.15377788`
- Legal TTT eval time: `21,509.768s`
- Total submission size int6+lzma: `15,807,729` bytes
- Peak memory: `11,699 MiB allocated`, `12,102 MiB reserved`

Legal score-first TTT:
- Chunks: `1,893`
- Chunk size: `32,768` tokens
- Running score bottomed around the mid-run `1.1518-1.1520` region, then finished at `1.15377788`
- Terminal lines:
  - `ttt_sliding:done val_loss=1.948100 val_bpb=1.153778 elapsed=21509.2s`
  - `legal_ttt_exact val_loss:1.94810047 val_bpb:1.15377788`

Training volume:
- Global batch: `262,144` tokens/step
- Total train tokens seen: `2,096,627,712`
- Fraction of default published 8B-token training export seen: `0.2621 epochs`

Included files:
- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_gpt_stack.py`
- full train log from `stack16h_base_s1337.txt`
