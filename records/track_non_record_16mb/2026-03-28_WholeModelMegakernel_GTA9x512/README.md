This folder packages a standalone non-record submission attempt for a packed-GTA whole-model CUDA path.

Status

- Track: `non-record-unlimited-compute-16mb`
- Training intent: unlimited-compute training
- Rule intent: keep all other competition rules unchanged
- Current implementation: single-file trainer with inline CUDA extension, packed fp32 runtime buffers, custom CUDA forward/eval kernels, cuBLAS-backed projections, and a manual reverse pass over the packed buffers

Completed Blackwell run

- Hardware: `1x RTX PRO 6000 Blackwell Workstation Edition`
- Safe settings used: `TRAIN_BATCH_TOKENS=262144`, `VAL_BATCH_SIZE=262144`, `SDP_BACKEND=math`
- Wallclock stop: `723` steps in about `7206.646s`
- Throughput: about `9.97s/step`, about `26.3k tokens/s`
- Pre-quant final validation: `val_loss=2.5300`, `val_bpb=1.4984`
- Post-quant exact roundtrip validation: `val_loss=2.59101960`, `val_bpb=1.53454775`
- Peak memory: `15715 MiB allocated`, `17114 MiB reserved`
- Compressed artifact size: `8389415` bytes
- Total code + compressed artifact size with the current branch code: `8519424` bytes

What is in `train_gpt.py`

- GTA attention with `Wq [512,512]`, `Wkv_tied [256,512]`, `Wk_rope [32,512]`
- Partial RoPE on the last `32` query/key dimensions
- Flat fp32 runtime parameters:
  - `tok_emb_weight`
  - `block_matrix_buf`
  - `block_scalar_buf`
  - `skip_weights`
- Inline extension entrypoints for:
  - whole-model forward loss
  - whole-model eval loss
  - whole-model backward
- Deterministic named tensor views for `state_dict`, checkpointing, and int8 export
- Local safety guard for single-GPU cards under about `36 GiB` VRAM

Local validation completed

- Python compile/import checks passed
- Packed reference path forward/backward passed on the local RTX 4090
- Exact `state_dict` roundtrip passed
- Inline extension built successfully on this workstation
- Extension parity gate passed
- Extension-backed forward/backward succeeded with `MEGAKERNEL_REQUIRED=1`
- Float32 gradient spot check was close for the packed CUDA path

Local 4090 smoke benchmark

- Hardware: `RTX 4090 24 GB`
- Safe settings: `TRAIN_BATCH_TOKENS=65536`, `VAL_BATCH_SIZE=65536`, `SDP_BACKEND=math`
- Measured window: `300` training steps in about `1185.4s`
- Throughput: about `3.95s/step`, about `16.6k tokens/s`, about `911 steps/hour`
- Loss trend: `6.9347 -> 4.2258`
- Checkpoints written cleanly at steps `100`, `200`, and `300`
- Peak observed board memory during the measured run stayed around `9.8 / 23.0 GiB`

Artifacts in this folder

- `train_gpt.py`: standalone trainer and CUDA path
- `train.log`: sanitized Blackwell finished-run log excerpt
- `submission.json`: metadata for the completed local Blackwell run

Limits of this draft

- This does not claim a Hopper-specific WGMMA/TMA implementation
- This does not claim a single persistent whole-model kernel launch
- This is still a local Blackwell result, not an H100 validation run
- The PR branch includes small post-run hygiene fixes for review feedback, but the recorded Blackwell metrics above are from the finished local training run
