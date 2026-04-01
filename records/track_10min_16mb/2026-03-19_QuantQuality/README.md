# Quant Quality

## Summary

Improved quantization fidelity on the seq4096 training trunk via two changes:

1. **Tighter int8 clipping** — `INT8_CLIP_PERCENTILE=99.99995` (vs default 99.99984), retaining more of the weight distribution tail
2. **Higher-precision per-row scales** — `INT8_PER_ROW_SCALE_DTYPE=float32` (vs default float16), reducing scale quantization error

Combined with the strong TrainingOptSeq4096 optimizer tuning (Muon momentum 0.99, extended warmup, warmdown 3000).

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Training geometry: `TRAIN_SEQ_LEN=4096 TRAIN_BATCH_TOKENS=393216`
- Learning rates: `TIED_EMBED_LR=0.03 MATRIX_LR=0.02 SCALAR_LR=0.02`
- Muon: `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500`
- Schedule: `WARMDOWN_ITERS=3000`
- Quantization: `INT8_CLIP_PERCENTILE=99.99995 INT8_PER_ROW_SCALE_DTYPE=float32`

## Command

```bash
TRAIN_SEQ_LEN=4096 \
TRAIN_BATCH_TOKENS=393216 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
INT8_CLIP_PERCENTILE=99.99995 \
INT8_PER_ROW_SCALE_DTYPE=float32 \
MAX_WALLCLOCK_SECONDS=600 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- Pre-quant eval: `val_bpb:1.1895` (step 11300)
- Post-quant (int8 + zlib): `val_loss:2.01343926 val_bpb:1.19247214`
- Quantization gap: ~0.003 bpb
- Train time: 600069ms (step_avg: 52.67ms)
- Steps: 11,389/20,000 (wallclock limited)
- Artifact: 15,934,552 bytes (code: 58,672 + model: 15,875,880)

## Hardware

Run on 8xH100 (Hyperbolic).

## Included Files

- `train_gpt.py` (code snapshot)
- `train.log` (training log)
- `submission.json` (metadata)
