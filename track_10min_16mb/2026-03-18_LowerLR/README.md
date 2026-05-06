This record captures the `Lower LR` submission.

## Summary

Same baseline architecture (9x512, SP-1024, 4 KV heads, tied embeddings, relu^2 MLP) with lower learning rates. A systematic LR sweep over 8 experiments showed the default Muon/Adam learning rates (MATRIX_LR=0.04, SCALAR_LR=0.04, TIED_EMBED_LR=0.05) were too high. Optimal is approximately half the default.

## Changes from baseline
- `MATRIX_LR=0.02` (default: 0.04)
- `SCALAR_LR=0.02` (default: 0.04)
- `TIED_EMBED_LR=0.03` (default: 0.05)

No architecture, schedule, or other hyperparameter changes.

## LR sweep results (8-GPU H200, 600s)

| MATRIX_LR | val_bpb (post-quant) | Delta vs baseline |
|---|---|---|
| 0.06 | 1.2445 | +0.0159 (much worse) |
| 0.04 (default) | 1.2286 | — |
| 0.03 | 1.2279 | -0.0007 |
| 0.025 | 1.2250 | -0.0036 |
| **0.02** | **1.2230** | **-0.0056** |
| 0.015 | 1.2234 | -0.0052 |

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Learning rates: `MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command:
```bash
RUN_ID=exp25_lr_0.02 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `14421/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0571`, `val_bpb:1.2183`
- Post-quant roundtrip eval: `val_loss:2.0649`, `val_bpb:1.2230`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22296644`
- Train time: `599847ms` (`step_avg:41.60ms`)
- Peak memory: `10246 MiB allocated`, `10310 MiB reserved`
- Serialized model int8+zlib: `15803327 bytes`
- Code size: `50919 bytes`
- Total submission size int8+zlib: `15854246 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `7560609792`

Note: Run performed on 8xH200 (141GB HBM3e). Step time (41.60ms) is comparable to 8xH100 baseline (43.54ms), and memory usage (10.2 GiB) is well within H100's 80GB limit. The ~5% faster step time on H200 yields ~400 extra steps, which may account for a small portion of the improvement.

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact training log)
- `submission.json` (leaderboard metadata)
