# Experiment 0 — Baseline

**Date:** 2026-03-19T14:00:00+00:00
**Result:** BASELINE
**val_bpb:** 1.6419
**Artifact size:** 8,899,669 bytes
**Train time:** ~180s (3 min wallclock on 1×H100)

## Configuration
- 9 layers, 512 dim, 8 heads, 4 KV heads
- MLP expansion: 2x (ReLU²)
- Tied embeddings, vocab 1024, seq len 1024
- 17,059,912 parameters
- ~334 steps at ~540ms/step
- Muon optimizer (matrix_lr=0.04), Adam for embeddings/scalars

## Notes
This is the unmodified `train_gpt.py` from the OpenAI repo, run on 1×H100 with
`MAX_WALLCLOCK_SECONDS=180`. The official 8×H100/10min baseline achieves 1.2244 BPB.

## Command
```bash
RUN_ID=baseline_1gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=180 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Key output
```
model_params:17059912
step:334/20000 val_loss:2.7228 val_bpb:1.6126 train_time:180224ms
final_int8_zlib_roundtrip val_loss:2.7723 val_bpb:1.6419
Total submission size int8+zlib: 8899669 bytes
peak memory allocated: 10239 MiB
```
