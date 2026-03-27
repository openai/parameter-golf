# Non-Record Submission: CompressedUT CE + EMA Export + Export-Aligned Late QAT

This non-record submission captures a full-validation 10-minute run of the byte-level `compressed_ut` CE-only model on `8xH100`. The main focus of this variant is not JEPA training, but reducing the deployed artifact gap: the run exports EMA weights, applies late QAT aligned to the actual export quantizer, and uses a stronger clip-search pass during int6 packing.

This line does not beat the current main leaderboard, but it is a concrete non-record result with a reproducible full-val shipped-artifact score under the `16,000,000` byte cap.

## Summary

- Track: `non-record-16mb`
- Hardware: `8xH100 SXM`
- Model family: `compressed_ut`
- Training stage: `ce` only
- Validation scope: full FineWeb validation split
- Final shipped score: `final_quant_roundtrip_exact val_bpb: 1.44568091`
- Total submission size: `14,707,311` bytes

## Main Changes

- Larger byte-level compressed-UT layout:
  - `BYTE_DIM=384`
  - `MODEL_DIM=1024`
  - `BIGRAM_DIM=96`
  - `DECODER_DIM=192`
- CE-only path with no live JEPA objective in this run
- EMA export enabled for the final artifact
- Late QAT aligned to the same quantizer used at export
- Expanded export clip-search grid for int6 packing
- H100 launch wrapper disabling cuDNN SDPA due backend instability on this path

## Reproduction Command

Run from this records folder. `USE_ZSTD=0` is set explicitly because the logged run fell back to `zlib-9`, and this keeps the reproduced artifact path aligned with the saved log.

```bash
mkdir -p logs

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MODEL_FAMILY=compressed_ut
export TRAIN_STAGE=ce
export DATA_PATH=../../../data/datasets/fineweb10B_bytes
export SEED=1337

export ITERATIONS=1000000
export WARMUP_STEPS=20
export WARMDOWN_ITERS=50
export TRAIN_LOG_EVERY=10
export VAL_LOSS_EVERY=200
export MAX_VAL_TOKENS=32768
export FINAL_FULL_VAL=1
unset FINAL_MAX_VAL_TOKENS
export USE_COMPILE=0

export TRAIN_SEQ_LEN=1536
export TRAIN_BATCH_TOKENS=528384
export EVAL_STRIDE=64

export VOCAB_SIZE=256
export PAD_ID=0
export BYTE_OFFSET=0
export BYTE_COUNT=256
export BYTE_DIM=384

export MODEL_DIM=1024
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4.0
export PARTIAL_ROPE_DIM=32
export BIGRAM_DIM=96

export DECODER_DIM=192
export DECODER_LAYERS=1
export DECODER_MLP_MULT=4.0

export EXPORT_EMA_ENABLED=1
export EXPORT_EMA_DECAY=0.997
export LATE_QAT_THRESHOLD=0.05
export TTT_ENABLED=0
export USE_ZSTD=0

export MAX_WALLCLOCK_SECONDS=600
export RUN_ID=h100_ce_qat_ema_full

torchrun \
  --standalone \
  --nproc_per_node=8 \
  ./launch_train_no_cudnn_sdp.py \
  2>&1 | tee logs/${RUN_ID}.txt
```

## Key Metrics

From `train.log`:

- Final pre-quant export-view eval: `val_loss:0.9972`, `val_bpb:1.4386`
- Final quantized roundtrip eval: `val_loss:1.00206964`, `val_bpb:1.44568091`
- Quantization penalty: about `+0.0071 bpb`
- Timed stop: `8694/1000000` steps at `600191ms`
- Step average at stop: `69.04ms`
- Peak memory: `8409 MiB`
- Quantized model bytes: `14588307`
- Code bytes: `119004`
- Total submission bytes: `14707311`

## Notes

- This run is CE-only `compressed_ut`; the JEPA codepaths still exist in the script, but they were not active for this result.
- The launcher wrapper is included because this configuration previously hit a cuDNN SDPA backend failure on H100 when run directly.
- The log shows `zstd=1` in the config printout, but the actual artifact line is `Quantized model (zlib-9)`, so reproduction uses `USE_ZSTD=0` explicitly.

## Included Files

- `README.md` - submission summary and reproduction notes
- `submission.json` - metadata for the run
- `train.log` - exact captured run log
- `train_gpt.py` - code snapshot used for the run
- `launch_train_no_cudnn_sdp.py` - H100 launch wrapper used for the run
- `requirements.txt` - dependency snapshot
