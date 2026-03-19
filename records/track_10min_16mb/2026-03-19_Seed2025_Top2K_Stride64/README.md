This folder captures a non-record 10-minute submission for the `top2K + sliding-window` lane.

The submitted run is the best under-cap seed:
- `seed2025`
- `final_sliding_window_eval_exact stride:64 val_loss:1.95946000 val_bpb:1.16050360`
- `Total submission size quant+zstd: 15844924`

Technique summary:
- `9x512`, `KV4`, `MLP_MULT=3`, tied embeddings
- `int6` on `.mlp.`, `.attn.c_q.`, `.attn.c_v.`, `.attn.proj.`
- `blocks.7.attn.c_k.weight` and `blocks.8.attn.c_k.weight` kept in fp16
- remaining `c_k` matrices stored as grouped int8 with `group_size=64`
- `tok_emb.weight` kept in fp16
- `zstd` artifact compression
- sliding-window evaluation with `EVAL_STRIDE=64`

This is not packaged as a new SOTA claim. The exact same lane was rerun on one confirming under-cap seed:

| seed | val_loss | val_bpb | total bytes | note |
|-----:|---------:|--------:|------------:|------|
| 2025 | 1.95946000 | 1.16050360 | 15,844,924 | submitted run |
| 42 | 1.96035715 | 1.16103494 | 15,802,877 | under cap |

The lane is strong and reproducible on two under-cap seeds, but this folder is still presented as a non-record submission rather than a SOTA claim.

Command for the submitted run:

```bash
RUN_ID=seed2025_top2k_stride64_v1 \
SEED=2025 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
QK_GAIN_INIT=1.7 \
LOWBIT_BITS=6 \
LOWBIT_STE=0 \
LOWBIT_NAME_PATTERNS=.mlp.,.attn.c_q.,.attn.c_v.,.attn.proj. \
INT8_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight,blocks.7.attn.c_k.weight,blocks.8.attn.c_k.weight \
INT8_GROUP_OVERRIDES=.attn.c_k.:64 \
SWA_ENABLED=0 \
SERIAL_COMPRESSOR=zstd \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Included files:
- `train_gpt.py`: exact minified trainer snapshot used for the run
- `train.log`: submitted `seed2025` log
- `train_seed42.log`: confirming run under the cap
- `submission.json`: metadata for the submitted `seed2025` run
