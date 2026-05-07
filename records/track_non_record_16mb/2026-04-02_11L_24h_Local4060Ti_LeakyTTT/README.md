## Record: 11L 24h Local 4060 Ti LeakyTTT (val_bpb: 1.1443)

**val_bpb = 1.14430187** (legal score-first TTT) | **15.70 MB** total artifact | **1x RTX 4060 Ti 16GB**, Windows, 24h wallclock

This record captures a local unlimited-compute non-record submission. It is not intended to satisfy the 10-minute `8xH100` constraint for the main leaderboard. This run re-used the same 11-layer LeakyReLU^2 + Parallel Muon + XSA4 + Partial RoPE + LN scale + EMA + legal TTT stack as the earlier 16-hour local run, but extended training to a 24-hour wallclock cap at fixed `2048` sequence length.

### Summary

- Timed training stopped at `12037/14000` steps due to the 24-hour wallclock cap
- Final stop-time validation: `1.1661 val_bpb`
- Post-EMA diagnostic score: `1.16489097 val_bpb`
- Post-int6 roundtrip score: `1.16959971 val_bpb`
- Post-int6 sliding-window score: `1.14561473 val_bpb`
- Final legal score-first TTT score: `1.14430187 val_bpb`
- Total submission size: `15,702,576` bytes

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
| Stop step | `12037/14000` |
| Stop-time validation | `1.1661 val_bpb` |
| Post-EMA diagnostic | `1.16489097 val_bpb` |
| Int6 roundtrip | `1.16959971 val_bpb` |
| Int6 sliding-window | `1.14561473 val_bpb` |
| **Legal TTT final** | **`1.14430187 val_bpb`** |
| Artifact bytes | `15,605,300` |
| Code bytes | `97,276` |
| **Total bytes** | **`15,702,576`** |

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
ttt_sliding:done val_loss=1.932101 val_bpb=1.144302 elapsed=21077.5s
legal_ttt_exact val_loss:1.93210066 val_bpb:1.14430187
```

### Training Volume

- Global batch: `262,144` tokens/step
- Total train tokens seen: `3,155,427,328`
- Fraction of default published `8B`-token training export seen: `0.3944 epochs`

### Command

```bash
RUN_ID=stack24h_base_s1337 \
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
ITERATIONS=14000 \
MAX_WALLCLOCK_SECONDS=86400 \
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
- `stack24h_base_s1337.txt`
