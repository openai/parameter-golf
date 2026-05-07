This non-record submission reproduces the public PR60-style stack on local `1xA100` hardware under a strict 10-minute train cap.

Goal:
- Test transfer of the PR60 recipe to constrained local throughput.
- Provide a reproducible negative result useful for compute-scaling analysis.

Recipe included in this script:
- Sliding-window final eval (`EVAL_STRIDE=64`)
- FP16 tied embedding export (`tok_emb` kept fp16)
- 10 transformer layers
- Decoupled Muon weight decay
- Overtone spectral embedding init
- Phase-transition residual mixing

Run setup:
- Hardware: `1x NVIDIA A100-SXM4-40GB`
- Dataset: `fineweb10B_sp1024`
- `TRAIN_SHARDS=80`
- `MAX_WALLCLOCK_SECONDS=600`
- `EVAL_STRIDE=64`

Command:
```bash
RUN_ID=exp_a100_20260320_pr60stack_v1 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Observed outcome (`train.log`):
- Stop: `step:716`, `train_time:600393ms` (`step_avg:838.54ms`)
- Final: `final_int8_zlib_roundtrip_exact val_bpb:1.41057617`
- Artifact: `Total submission size int8+zlib: 11124153 bytes`
- Final eval time: `568060ms`

Interpretation:
- The stack remains under the 16MB limit, but local single-GPU throughput reaches fewer optimization steps within the same wallclock budget, yielding weaker BPB than faster multi-GPU settings.

Included files:
- `train_gpt.py` (exact PR60-style script snapshot used)
- `train.log` (full run log)
- `submission.json` (metadata)
