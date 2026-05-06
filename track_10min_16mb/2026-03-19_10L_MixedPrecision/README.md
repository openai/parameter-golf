This record captures the `10L Mixed Precision` submission.

## Summary

Two key improvements over the baseline:

1. **10 transformer layers** instead of 9 — adds depth for better language modeling
2. **Lower learning rates** — MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03 (vs default 0.04/0.04/0.05)
3. **Mixed int8/int6 compression** — middle layers (3,4,5,6) use int6 precision (round int8 to nearest 4) for better zlib compression, while first/last layers keep full int8

The 10-layer model at dim=512 has 18.9M params which compresses to 17.6MB with standard int8+zlib — 1.6MB over the 16MB cap. By reducing precision on the 4 middle layers to int6 (64 quantization levels instead of 256), the compressed size drops to 15.9MB with only 0.0018 bpb quality loss from quantization.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Learning rates: `MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03`
- Mixed precision: `INT4_LAYERS=3,4,5,6 INT4_STEP=4` (int6 for middle layers)
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

## Command

```bash
RUN_ID=exp45_10L_int6_mid \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
NUM_LAYERS=10 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
INT4_LAYERS=3,4,5,6 \
INT4_STEP=4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- Pre-quant eval: `val_loss:2.0480`, `val_bpb:1.2129`
- Post-quant (int8/int6 mixed + zlib): `val_loss:2.0510`, `val_bpb:1.2147`
- Exact: `final_int8_zlib_roundtrip_exact val_bpb:1.21474500`
- Quantization gap: 0.0018 bpb (vs baseline's 0.0093)
- Train time: `599732ms` (`step_avg:45.78ms`)
- Steps: 13,100/20,000 (wallclock limited)
- Peak memory: 11,389 MiB allocated
- Artifact: 15,928,974 bytes (code: 48,917 + model: 15,880,057)

## Compression Analysis

| Layer Group | Precision | Reason |
|---|---|---|
| Layers 0-2 (early) | int8 (256 levels) | Critical for input processing |
| Layers 3-6 (middle) | int6 (64 levels) | Less sensitive, saves ~1.6MB |
| Layers 7-9 (late) | int8 (256 levels) | Critical for output quality |

## LR Sweep Results

Systematic sweep showed default LR (0.04) was too high:
| MATRIX_LR | val_bpb (9L baseline) |
|---|---|
| 0.04 (default) | 1.2286 |
| 0.02 (optimal) | 1.2230 |

## Note on Hardware

Run performed on 8xH200 (step_avg: 45.78ms). H100 baseline was 43.54ms/step for 9 layers; 10 layers would be ~47-48ms on H100, yielding ~12,500-12,700 steps. Results should be comparable.

## Included Files

- `train_gpt.py` (code snapshot)
- `train.log` (training log)
- `submission.json` (metadata)
