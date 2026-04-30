# 10L Hybrid SWA 2048 FP16 Embed

**val_bpb: 1.20128722** | **val_loss: 2.02832315** | **15,551,524 bytes** | 8×H100

## Summary

This record improves the simple baseline with a modest 10-layer transformer and longer-context hybrid sliding-window attention.

Main changes:

- Increased depth from 9 to 10 transformer blocks.
- Increased training sequence length from 1024 to 2048 tokens.
- Used hybrid attention with 1024-token sliding-window layers and full-attention layers every 2 blocks.
- Kept tied token embeddings in fp16 during int8 export to reduce quantization loss.
- Used AdamW for token/scalar parameters and decoupled weight decay for Muon matrix parameters.
- Tuned optimizer defaults: `TIED_EMBED_LR=0.07`, `MATRIX_LR=0.06`, `SCALAR_LR=0.05`.

## Configuration

```bash
RUN_ID=swafourth10layers_2048window \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key defaults in `train_gpt.py`:

```text
NUM_LAYERS=10
TRAIN_SEQ_LEN=2048
SLIDING_WINDOW=1024
GLOBAL_LAYER_STRIDE=2
TRAIN_BATCH_TOKENS=524288
WARMDOWN_ITERS=2500
MAX_WALLCLOCK_SECONDS=600
```

## Results

From `train.log`:

- Hardware: `8xH100`
- Steps: `11996/20000`
- Train time: `600042ms`
- Step average: `50.02ms`
- Peak memory: `9848 MiB allocated`, `10118 MiB reserved`
- Pre-quant eval: `val_loss:2.0234`, `val_bpb:1.1984`
- Post-quant roundtrip: `val_loss:2.02832315`, `val_bpb:1.20128722`
- Serialized model int8+zlib: `15501734 bytes`
- Code size: `49790 bytes`
- Total submission size int8+zlib: `15551524 bytes`

## Baseline Comparison

Simple baseline:

- Post-quant `val_loss:2.0727`
- Post-quant `val_bpb:1.2244`

This run:

- Post-quant `val_loss:2.0283`
- Post-quant `val_bpb:1.2013`

Improvement:

- `val_loss`: about `-0.0444`
- `val_bpb`: about `-0.0231`

## Quantization Gap

The fp16 tied embedding passthrough kept the export gap small:

- Pre-quant: `val_loss:2.0234`, `val_bpb:1.1984`
- Post-quant: `val_loss:2.02832315`, `val_bpb:1.20128722`
- Gap: about `+0.0049 loss`, `+0.0029 bpb`

## Record-Track Note

The simple baseline record reports `val_bpb:1.22436570`. This run reaches `val_bpb:1.20128722`, improving by about `0.0231 bpb` while staying under the 16MB submission size limit. 

## Standalone Reproducibility

This folder contains the exact `train_gpt.py` snapshot used for the run, with the main defaults baked in. The only expected runtime inputs are the cached `fineweb10B_sp1024` dataset and SentencePiece tokenizer described in the main repository README.

## Included Files

- `train_gpt.py`: training script used for the run
- `train.log`: full training log
- `submission.json`: record metadata