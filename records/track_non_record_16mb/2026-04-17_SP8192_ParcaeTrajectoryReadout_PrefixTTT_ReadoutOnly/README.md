# SP8192 Parcae Trajectory Readout + Phased Prefix TTT + Readout-Only Adaptation

Experimental non-record submission folder for a distinct eval-time overlay on top of our Parcae trajectory-readout stack.

## Current Result

Completed seed on 8xH100:

- `RUN_ID=sp8192_parcae_readout_prefix20_readoutonly_s42_r1`
- `seed=42`
- `steps=3475`
- `pre-quantization post-ema val_bpb = 1.10376925`
- `quantized val_bpb = 1.11420379`
- `quantized_sliding_window val_bpb = 1.09762780`
- `quantized_ttt val_bpb = 1.09760452`
- `Serialized model quantized+brotli = 15,968,435 bytes`
- `Total submission size quantized+brotli = 16,071,408 bytes`

This overlay clearly underperformed the full-TTT Parcae readout run. It is archived as a negative experiment.

## Idea

Keep the train-side model identical to `ParcaeTrajectoryReadout_Late2600`, but change the legal TTT schedule:

1. adapt only the tiny `readout_delta` parameters at eval time
2. spend extra adaptation budget on a scored prefix of each chunk
3. use the improved readout state for the remainder of the chunk

This is a cheap way to test whether trajectory-state readout becomes more useful when TTT is focused on the readout path itself instead of updating the full model.

## Reproduced Run

```bash
RUN_ID=sp8192_parcae_readout_prefix20_readoutonly_s42_r1 \
SEED=42 \
VOCAB_SIZE=8192 \
NUM_LOOPS=2 \
LOOP_START=3 \
LOOP_END=5 \
ENABLE_LOOPING_AT_STEP=2600 \
ENABLE_PARALLEL_RESIDUAL_AT_STEP=0 \
PARALLEL_RESIDUAL_START=7 \
RECUR_ATTN_GATE=0 \
TOKEN_ROUTE_ENABLED=0 \
SHARED_ADAPTER_DIM=0 \
AUX_EXIT_LAYER=-1 \
MUON_MOMENTUM=0.98 \
GPTQ_RESERVE_SECONDS=4 \
GPTQ_CALIBRATION_BATCHES=16 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
TTT_PARAM_MODE=readout_only \
TTT_PREFIX_CHUNK_RATIO=0.20 \
TTT_PREFIX_EPOCHS=4 \
TTT_PREFIX_LR_SCALE=1.15 \
TTT_PREFIX_HARD_WINDOW_FRACTION=0.25 \
LOOP_INJECT_ENABLED=1 \
LOOP_INJECT_SCALE=1.0 \
LOOP_INJECT_START_PASS=1 \
LOOP_INJECT_INIT=0.10 \
USE_PASS_READOUT=1 \
READOUT_GROUPS=16 \
READOUT_SCALE=0.35 \
FIND_UNUSED_PARAMETERS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

- phased prefix TTT plus `readout_only` adaptation reduced step count substantially
- train-side quality and final TTT quality were both worse than the full-TTT Parcae readout run
- this branch is not a promising leaderboard path in its current form

## Artifacts

- [train_seed42.log](./train_seed42.log)
- [train_gpt.py](./train_gpt.py)
