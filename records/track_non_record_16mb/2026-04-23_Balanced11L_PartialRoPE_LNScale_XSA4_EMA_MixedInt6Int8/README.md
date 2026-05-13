# Balanced 11L + Partial RoPE + LN Scale + XSA4 + EMA + mixed int6/int8

This submission is a standalone variant of `openai/parameter-golf`'s `train_gpt.py` baseline for the 16 MB / 10 minute challenge. It keeps the baseline training/evaluation structure and byte-accurate SentencePiece BPB calculation, then layers a set of targeted changes that improved the attached run.

## Summary

Relative to the public naive baseline, this script pushes the model toward better final quality without exceeding the artifact cap. The main changes are:

1. Deeper model with U-Net skip connections: `NUM_LAYERS=11` (5 encoder + 6 decoder) instead of `9`.
2. Longer context during training and evaluation: `TRAIN_SEQ_LEN=2048` instead of `1024`.
3. Partial RoPE: rotary applied to only the first `ROPE_DIMS=16` of `64` head dimensions; the remaining 48 dims attend without positional bias.
4. Layer-wise LayerNorm scaling: `LN_SCALE=1` damps each block's pre-attention and pre-MLP norms by `1/sqrt(layer_idx + 1)`.
5. Exclusive Self-Attention on the last 4 layers: `XSA_LAST_N=4` subtracts the projection of `y` onto `v_self` after SDPA on the deepest blocks, with zero new parameters.
6. Longer warmdown and longer Muon momentum warmup: `WARMDOWN_ITERS=3500`, Muon momentum ramps from `0.92` to `0.99` over the first `1500` steps.
7. Weight decay enabled on both optimizers: `MUON_WD=0.04`, `ADAM_WD=0.04`.
8. EMA for final export: `EMA_ENABLED=1`, `EMA_DECAY=0.997`, `EMA_WARMUP_STEPS=500`. The script applies EMA weights before the final eval/export path.
9. Per-row mixed-precision GPTQ-lite quantization: `MATRIX_QUANT_BITS=6` for 2D matrices, `EMBED_QUANT_BITS=8` for the token embedding, with a per-row clipping search across 11 percentiles.
10. Multi-codec compression: `COMPRESSION_PREFERENCE=auto` evaluates `zstd-22`, `lzma-9`, and `zlib-9` and keeps the smallest. In the attached run `lzma-9` won.
11. Sliding-window evaluation: `USE_SLIDING_EVAL=1` with `EVAL_STRIDE=64`, while still scoring each target token once.

## Architecture and training setup

Model configuration in the logged run:

* Vocabulary: `1024` (SentencePiece)
* Layers: `11` (5 encoder + 6 decoder with U-Net skip weights)
* Model dim: `512`
* Attention: `8` heads, `4` KV heads (GQA)
* MLP expansion: `3x`
* Activation: ReLU squared
* Tied embeddings: enabled
* Skip connections: encoder/decoder U-Net-style per-dim skip weights
* Partial RoPE: `16` of `64` head dims
* XSA: last `4` layers
* LayerNorm scaling: `1/sqrt(layer_idx + 1)` on each block's attn and mlp inputs
* Logit softcap: enabled (`30.0`)
* Parameter count: `26,501,720`

Run configuration in the attached log:

* Hardware: `8x H100 80GB HBM3` (SXM)
* Global batch: `524,288` tokens/step
* Target iterations: `9,000`
* Actual timed stop: `7,887` steps due to the wallclock guard
* Total train tokens seen: `4,135,059,456`
* Sequence length: `2048`
* Seed: `1337`

## Results from the attached run

Key metrics from the attached log:

* Last pre-EMA eval at stop: `val_loss=1.9564`, `val_bpb=1.1587`
* EMA eval before export: `val_loss=1.9413`, `val_bpb=1.1497`
* Final post-quant roundtrip eval: `val_loss=1.9502`, `val_bpb=1.1550`
* Train time: `570136 ms`
* Average step time at stop: `72.29 ms`
* Artifact payload: `15,192,076` bytes compressed
* Code size: `61,011` bytes
* Total submission size: `15,253,087` bytes
* Under budget: `True`

Compared with the public naive baseline (`val_bpb=1.2243657`), this run improves the final post-quant score by about `0.0694` BPB.

## Quantization and packaging

The script is configured with `MATRIX_QUANT_BITS=6` for 2D matrix weights and `EMBED_QUANT_BITS=8` for the token embedding, keeps small/control tensors in float, and applies a per-row clipping search (`_best_clip_per_row_2d`) that picks the percentile minimizing row-wise dequantization MSE. The dequantized state dict is loaded back into the model and re-evaluated before export, so the reported score reflects the exact artifact a verifier would reproduce.

For compression, the script tries `zstd-22`, `lzma-9`, and `zlib-9` and keeps the smallest. In the attached run, the final export path produced:

* `final_mixedq_lzma-9_roundtrip`

Even though the script defaults to `USE_ZSTD=1`, the auto-selector chose `lzma-9` because it produced the smaller payload at this model size. Reproducers with `zstandard` installed should land on the same codec; if the `zstandard` package is missing, the script silently skips it and selects between `lzma-9` and `zlib-9`.

## Reproduction command

A concise reproduction command for the logged configuration is:

```
NCCL_IB_DISABLE=1 \
SEED=1337 \
RUN_ID=balanced_11l_xsa4_ema_mixedq \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 \
ARCH_PRESET=balanced_11l \
USE_ZSTD=1 \
ZSTD_LEVEL=22 \
COMPRESSION_PREFERENCE=auto \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=9000 \
WARMDOWN_ITERS=3500 \
MATRIX_QUANT_BITS=6 \
EMBED_QUANT_BITS=8 \
FP16_EMBED=0 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
EMA_WARMUP_STEPS=500 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
EVAL_STRIDE=64 \
USE_SLIDING_EVAL=1 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
MLP_MULT=3 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
TIE_EMBEDDINGS=1 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=0 \
 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included files

Submission folder contents:

* `train_gpt.py`
* `README.md`
* `submission.json`
* `train.log`
