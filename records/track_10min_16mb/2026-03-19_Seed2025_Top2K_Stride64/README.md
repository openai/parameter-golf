# Int6 MLP3x + Late-K Passthrough + SlidingWindow

## Summary

This folder captures a 10-minute submission for leaderboard placement, not a record claim.

The submitted run is the best under-cap seed on this lane:

- `seed2025`
- `final_sliding_window_eval_exact stride:64 val_loss:1.95946000 val_bpb:1.16050360`
- `Total submission size quant+zstd: 15844924`

The lane stacks four practical improvements on the strong 9-layer, 512-dim GPT recipe:

1. **Int6 mixed quantization + zstd**: `.mlp.`, `.attn.c_q.`, `.attn.c_v.`, and `.attn.proj.` are stored in int6, then compressed with `zstd`.
2. **3x MLP expansion**: `MLP_MULT=3` keeps the wider hidden layer that materially improves score within the byte budget.
3. **Selective K preservation**: `blocks.7.attn.c_k.weight` and `blocks.8.attn.c_k.weight` stay in fp16, while the remaining `c_k` matrices use grouped int8 with `group_size=64`.
4. **Sliding-window evaluation**: `EVAL_STRIDE=64` gives near-full context at evaluation time and is the main improvement over the stride-256 variant.

## Configuration

```text
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=3 TIE_EMBEDDINGS=1
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000 QK_GAIN_INIT=1.7
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024
LOWBIT_BITS=6 LOWBIT_STE=0
LOWBIT_NAME_PATTERNS=.mlp.,.attn.c_q.,.attn.c_v.,.attn.proj.
INT8_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight,blocks.7.attn.c_k.weight,blocks.8.attn.c_k.weight
INT8_GROUP_OVERRIDES=.attn.c_k.:64
SERIAL_COMPRESSOR=zstd
EVAL_STRIDE=64
MAX_WALLCLOCK_SECONDS=600
```

## Command

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

## Key Metrics

- Training stopped at step **12791/20000** due to the 600s wallclock cap
- Average training step time: **46.91ms**
- Model params: **21,778,504**
- Pre-quant eval: `val_loss:2.0047 val_bpb:1.1873`
- Quantized roundtrip: `val_loss:2.01675174 val_bpb:1.19443398`
- Sliding window (stride=64): `val_loss:1.95946000 val_bpb:1.16050360`
- Sliding-window eval time: **70834ms**
- Code size: **37,988 bytes**
- Total submission size: **15,844,924 bytes**

## Quantization Strategy

The main serializer choice is to spend the remaining bytes on the attention `K` path rather than on broader fp16 promotion:

- `tok_emb.weight`: fp16 passthrough
- `blocks.7.attn.c_k.weight` and `blocks.8.attn.c_k.weight`: fp16 passthrough
- remaining `.attn.c_k.` matrices: grouped int8, `group_size=64`
- `.mlp.`, `.attn.c_q.`, `.attn.c_v.`, `.attn.proj.`: int6
- compressor: `zstd`

This keeps the artifact under `16,000,000` bytes while preserving the highest-value late-layer key projections.

## Additional Seeds

The same lane was rerun on two more under-cap seeds:

| seed | val_loss | val_bpb | total bytes | note |
|-----:|---------:|--------:|------------:|------|
| 2025 | 1.95946000 | 1.16050360 | 15,844,924 | submitted run |
| 42 | 1.96035715 | 1.16103494 | 15,802,877 | under cap |
| 4242 | 1.96595032 | 1.16434753 | 15,822,568 | under cap |

This folder is still presented as a submission for placement rather than a new SOTA claim.

## Included Files

- `train_gpt.py`: exact minified trainer snapshot used for the submitted run
- `train.log`: submitted `seed2025` log
- `train_seed42.log`: confirming run under the cap
- `train_seed4242.log`: second confirming run under the cap
- `submission.json`: metadata for the submitted `seed2025` run
