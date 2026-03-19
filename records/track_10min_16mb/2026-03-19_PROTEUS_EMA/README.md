# PROTEUS EMA — Parameter Golf Submission

**Built with [PROTEUS](https://lightspeedup.com) by LightSpeedUp**

## Approach

Minimal modification to the baseline: **EMA weight averaging** (Exponential Moving Average) applied during training and exported at serialization. EMA smooths weight distributions, reducing INT8 quantization degradation from 0.0072 BPB (baseline) to 0.0048 BPB.

### What changed (26 lines added to baseline)

1. **EMA state initialization** — fp32 clone of all parameters after model creation
2. **EMA update** — `lerp_(params, 1 - decay)` every 10 steps during training
3. **EMA export** — copy EMA weights to model before INT8 quantization

No architectural changes. Same model shape, same optimizer, same training loop.

### Why EMA helps

Post-training INT8 quantization introduces rounding error. EMA-averaged weights have smoother distributions with fewer outliers, making them more quantization-friendly. The improvement is small but free at inference time — the exported model IS the EMA model.

## Configuration

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Same as baseline with additional env vars:
- `EMA_ENABLED=1` (default)
- `EMA_DECAY=0.999` (default)
- `EMA_EVERY=10` (default)

All other parameters match the baseline exactly:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

## Key Metrics

- Timed training stopped at `12881/20000` steps due to wallclock cap
- Pre-quant eval at stop: `val_loss:2.0607`, `val_bpb:1.2205`
- Post-quant roundtrip eval: `val_loss:2.0689`, `val_bpb:1.2253`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.22534607`
- Train time: `599970ms` (`step_avg:46.58ms`)
- Peak memory: `10249 MiB allocated`
- Serialized model int8+zlib: `15843708 bytes`
- Code size: `49825 bytes`
- Total submission size int8+zlib: `15863608 bytes`

### Training volume

- Global batch: `524288` tokens/step
- Total train tokens seen: `~6.75B`

## Platform

Run on Modal 8×H100 (SXM). Note: baseline was run on RunPod 8×H100 SXM at 43.54ms/step; our Modal runs averaged 46.58ms/step due to platform differences (volume I/O, container overhead). On matched hardware, we expect the score to improve.

## Included Files

- `train_gpt.py` — training script (baseline + 26 lines of EMA)
- `train.log` — full training log from the submission run
- `submission.json` — leaderboard metadata
- `README.md` — this file
