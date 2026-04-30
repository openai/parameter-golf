# Non-Record Run: RandomLinearMaps (Random Subspace Optimization)

This folder contains a non-record experiment snapshot. The README only documents information that can be directly verified from files in this directory.

## Hardware and Precision

- Device: `4x Quadro RTX 8000`
- GPU architecture: `SM 7.5` (Turing)
- Training precision: `float32`
- Reason: this setup uses FP32 training for compatibility and stability on this GPU architecture.

## Files in This Directory

- `train_gpt.py`: training script with RSOAdamW
- `train_log.txt`: full run output (script echo, config, training progress, and final metrics)
- `train.sh`: a short launch command example

## Training Configuration (from `train_log.txt`)

- `world_size=4`, `grad_accum_steps=2`
- `train_batch_tokens=524288`, `train_seq_len=1024`
- `iterations=2000`, `warmup_steps=0`, `max_wallclock_seconds=0.0` (no wallclock early stop)
- Model parameters: `20,893,768`
- Tokenizer: `fineweb_1024_bpe.model`

## Key Results (this run)

- Final validation at `step 2000/2000`: `val_loss=2.5683`, `val_bpb=1.5211`
- Quantized round-trip eval: `final_int8_zlib_roundtrip_exact val_bpb=:1.53014584`
- Peak memory allocated: `38824 MiB`
- Submission size:
  - `Total submission size = 68323975 bytes`
  - `Total submission size int8+zlib = 12358817 bytes`

