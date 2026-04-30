# Seq2048 + EMA + per-row GPTQ-lite int6

This submission is a standalone variant of `openai/parameter-golf`'s `train_gpt.py` baseline for the 16 MB / 10 minute challenge. It keeps the baseline training/evaluation structure and byte-accurate SentencePiece BPB calculation, then layers a small set of targeted changes that improved the attached run.

## Summary

Relative to the public naive baseline, this script pushes the model toward better final quality without exceeding the artifact cap. The main changes are:

1. **Longer context during training and evaluation**: `TRAIN_SEQ_LEN=2048` instead of 1024.
2. **Longer warmdown**: `WARMDOWN_ITERS=3000` to spend more of the run on a gentler decay.
3. **Muon momentum warmup**: momentum ramps from `0.9` to `0.985` over the first `500` steps instead of staying flat.
4. **EMA for final export**: the script applies EMA weights before the final eval/export path.
5. **Per-row GPTQ-lite clipping search**: large 2D tensors try several clipping percentiles and keep the one with the lowest row-wise dequantization error.
6. **Sliding-window evaluation**: evaluation uses `USE_SLIDING_EVAL=1` with `EVAL_STRIDE=64`, while still scoring each target token once.

## Architecture and training setup

Model configuration in the logged run:

- Vocabulary: `1024` (SentencePiece)
- Layers: `9`
- Model dim: `512`
- Attention: `8` heads, `4` KV heads (GQA)
- MLP expansion: `3x`
- Tied embeddings: enabled
- Skip connections: encoder/decoder U-Net-style skip weights
- Activation: ReLU
- Logit softcap: enabled
- Parameter count: `21,778,504`

Run configuration in the attached log:

- Hardware: `8x H100 80GB HBM3`
- Global batch: `524,288` tokens/step
- Target iterations: `12,000`
- Actual timed stop: `9,627` steps due to the wallclock guard
- Total train tokens seen: `5,047,320,576`
- Sequence length: `2048`
- Seed: `1337`

## Results from the attached run

Key metrics from the attached log:

- Last pre-EMA eval at stop: `val_loss=1.9642`, `val_bpb=1.1633`
- EMA eval before export: `val_loss=1.9582`, `val_bpb=1.1598`
- Final post-quant roundtrip eval: `val_loss=1.9689`, `val_bpb=1.1661`
- Train time: `570062 ms`
- Average step time at stop: `59.21 ms`
- Artifact payload: `15,579,818` bytes compressed
- Code size: `58,789` bytes
- Total submission size: `15,638,607` bytes
- Under budget: `True`

Compared with the public naive baseline (`val_bpb=1.2243657`), this run improves the final post-quant score by about `0.0583` BPB.

## Quantization and packaging

The script is configured with `QUANT_BITS=6`, keeps the token embedding in fp16, and keeps small/control tensors in float. In the attached run, the final export path produced:

- `final_int6_zlib-9_roundtrip`

Even though the script defaults to `USE_ZSTD=1`, the canonical logged run fell back to `zlib-9` at export time. If you want the submission folder to match this log exactly on a machine that has `zstandard` installed, set `USE_ZSTD=0` when reproducing.

## Reproduction command

A concise reproduction command for the logged configuration is:

```bash
NCCL_IB_DISABLE=1 \
SEED=1337 \
RUN_ID=seq2048_ema_int6 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 \
USE_ZSTD=0 \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=12000 \
WARMDOWN_ITERS=3000 \
QUANT_BITS=6 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
MUON_MOMENTUM=0.985 \
MUON_MOMENTUM_WARMUP_START=0.9 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
EVAL_STRIDE=64 \
USE_SLIDING_EVAL=1 \
MLP_MULT=3 \
NUM_LAYERS=9 \
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

- `train_gpt.py`
- `README.md`
- `submission.json`
- `train.log`
