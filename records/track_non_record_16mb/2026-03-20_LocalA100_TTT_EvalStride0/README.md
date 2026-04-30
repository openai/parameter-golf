This non-record submission captures a local 1xA100 run optimized for the TTT metric with standard final eval (`EVAL_STRIDE=0`).

Purpose:
- Provide a reproducible local baseline for `final_int8_ttt_lora` under the 16MB artifact limit.
- Keep training capped at 10 minutes (`MAX_WALLCLOCK_SECONDS=600`).

Run setup:
- Hardware: `1x NVIDIA A100-SXM4-40GB`
- Dataset: `fineweb10B_sp1024`
- Tokenizer: `fineweb_1024_bpe.model`
- `TRAIN_SHARDS=80`
- `EVAL_STRIDE=0` (standard final roundtrip eval)
- `DISABLE_TTT=0` (TTT enabled)

Command:
```bash
RUN_ID=exp_a100_20260320_ttt_e0_v1 \
TRAIN_SHARDS=80 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=0 \
DISABLE_TTT=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Key metrics (`train.log`):
- Stop: `step:856`, `train_time:600328ms`
- `final_int8_ttt_lora val_bpb:1.3510` (eval_time `515653ms`)
- `final_int8_zlib_roundtrip_exact val_bpb:1.37827010`
- `Total submission size int8+zlib: 11876675 bytes`

Included files:
- `train_gpt.py` (exact script snapshot used)
- `train.log` (full run log)
- `submission.json` (metadata)
