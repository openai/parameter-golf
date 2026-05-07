# R08 Higher-LR Matrix/Scalar GPT 17M A40

- Date: 2026-04-02
- Track: non_record_16mb
- Author: Siddhardha Nanda (SID-6921)
- Reported val_bpb: 2.1827

## Summary

Increased matrix and scalar parameter learning rates relative to the stock baseline. Setting `MATRIX_LR=0.05` and `SCALAR_LR=0.04` (with matching `TIED_EMBED_LR=0.05`) while also halving the batch size to 1M tokens and using a 400-step warmdown schedule achieved a significant improvement from the stock baseline.

## What Changed

- `MATRIX_LR=0.05` (increased from stock default ~0.04)
- `SCALAR_LR=0.04` (adjusted)
- `TIED_EMBED_LR=0.05` (matched to MATRIX_LR)
- `TRAIN_BATCH_TOKENS=1048576` (halved from 2M default)
- `WARMDOWN_ITERS=400` (reduced from 1200 default)
- `ITERATIONS=60` (doubled from baseline 30)
- Architecture unchanged: 17M params, GQA (8 heads, 4 KV heads), ReLU², sp_bpe_1024 tokenizer

## Repro Command

```bash
export DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  TRAIN_BATCH_TOKENS=1048576 \
  WARMDOWN_ITERS=400 \
  MATRIX_LR=0.05 \
  SCALAR_LR=0.04 \
  TIED_EMBED_LR=0.05 \
  ITERATIONS=60 \
  MAX_WALLCLOCK_SECONDS=900
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Results

- val_bpb: 2.18271188 (int8+zlib roundtrip)
- val_loss: 3.68541758 (int8+zlib roundtrip)
- pre_quant_val_bpb: 2.1795
- pre_quant_val_loss: 3.6800
- compressed_bytes: 9,897,284 bytes (~9.4 MB, well under 16 MB cap)
- wallclock_seconds: ~169s
- GPU: 1× NVIDIA A40

## Notes

This was the best run (#1 of 10) from an automated campaign (`04_non_record_a40_campaign.sh`) testing batch sizes, warmdown schedules, QK gain values, and learning rate combinations. Higher LR for matrix parameters combined with a smaller batch size appears to be the key driver of improvement over the stock baseline.
