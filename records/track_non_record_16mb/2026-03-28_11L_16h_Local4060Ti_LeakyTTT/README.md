## Record: 11L 16h Local 4060 Ti LeakyTTT (val_bpb: 1.1538)

**val_bpb = 1.15377788** (legal score-first TTT) | **15.81 MB** total artifact | **1x RTX 4060 Ti 16GB**, Windows, 16h wallclock

This record captures a local unlimited-compute non-record submission using a local RTX4060 Ti 16GB GPU only. It is not intended to satisfy the 10-minute `8xH100` constraint for the main leaderboard. The goal of this run was to test how far the 11-layer LeakyReLU^2 + Parallel Muon + XSA4 + Partial RoPE + LN scale + EMA + legal TTT stack can be pushed on a single low end consumer GPU while still fitting under the `16,000,000` byte artifact cap.

### Summary

- Timed training stopped at `7998/9000` steps due to the 16-hour wallclock cap
- Post-EMA diagnostic score: `1.1742 val_bpb`
- Post-int6 roundtrip score: `1.17922533 val_bpb`
- Post-int6 sliding-window score: `1.15520389 val_bpb`
- Final legal score-first TTT score: `1.15377788 val_bpb`
- Total submission size: `15,807,729` bytes

### Stack

- 11 layers, `512d`, `8` heads, `4` KV heads
- LeakyReLU(0.5)^2 MLP
- Parallel Muon
- XSA on the last 4 layers
- Partial RoPE (`16/64`)
- Layerwise LN scale
- BigramHash `1536`
- VE128 on layers `9,10`
- EMA(`0.997`)
- SWA every `50`
- Legal score-first TTT
- int6 + lzma export
- Sliding-window evaluation with stride `64`

### Results

| Metric | Value |
|--------|-------|
| Stop step | `7998/9000` |
| In-training periodic eval | `1.1751 val_bpb` |
| Post-EMA diagnostic | `1.1742 val_bpb` |
| Int6 roundtrip | `1.17922533 val_bpb` |
| Int6 sliding-window | `1.15520389 val_bpb` |
| **Legal TTT final** | **`1.15377788 val_bpb`** |
| Artifact bytes | `15,710,900` |
| Code bytes | `96,829` |
| **Total bytes** | **`15,807,729`** |

### Legal TTT

Backward-looking, score-first TTT was applied after the int6 sliding-window evaluation:

- `1,893` chunks
- chunk size `32,768` tokens
- SGD with `lr=0.002`, `momentum=0.9`
- `3` epochs per chunk
- all blocks unfrozen
- gradient clip `1.0`

Terminal lines:

```text
ttt_sliding:done val_loss=1.948100 val_bpb=1.153778 elapsed=21509.2s
legal_ttt_exact val_loss:1.94810047 val_bpb:1.15377788
```

### Training Volume

- Global batch: `262,144` tokens/step
- Total train tokens seen: `2,096,627,712`
- Fraction of default published `8B`-token training export seen: `0.2621 epochs`

### Command

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
python train_gpt.py
```

### Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_gpt_stack.py`
- `stack16h_base_s1337.txt`
