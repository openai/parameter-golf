# SP8192 Parcae Trajectory Readout + Late Loop Onset

Experimental non-record submission folder for a distinct train-side recurrence idea.

## Current Result

Completed seed on 8xH100:

- `RUN_ID=sp8192_parcae_readout_l2600_s42_r4`
- `seed=42`
- `steps=4317`
- `pre-quantization post-ema val_bpb = 1.09058710`
- `quantized val_bpb = 1.10176743`
- `quantized_sliding_window val_bpb = 1.08508030`
- `quantized_ttt val_bpb = 1.08322102`
- `Serialized model quantized+brotli = 15,959,009 bytes`
- `Total submission size quantized+brotli = 16,061,982 bytes`

This branch was competitive enough to keep, but the current run is invalid for the 16MB track because it is `61,982` bytes over the limit.

## Idea

This variant combines two tiny train-side mechanisms around the recurrent band:

1. `Parcae`-style bounded loop reinjection at each loop boundary:

   `x <- A_bar * x + B_bar * x0`

2. grouped trajectory-delta readout on the final looped block:

   `x <- x_final + scale * Σ_g,p alpha[g,p] * (x_pass[p] - x_final)`

The intent is different from our prior pass-gate work:
- reinjection stabilizes loop dynamics and keeps the recurrent state anchored
- trajectory readout lets the model recover useful information from earlier loop passes
- the correction is tiny and identity-initialized, so parameter/size overhead stays negligible

## Distinctives vs prior local runs

- no pass-gated recurrence
- no token routing
- no shared adapter
- no auxiliary exit
- no loop-q-only TTT
- step-based late loop onset retained because it was a robust systems/training choice in our own sweeps

## Reproduced Run

```bash
RUN_ID=sp8192_parcae_readout_l2600_s42 \
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
TTT_PARAM_MODE=full \
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

## Notes From This Run

- throughput was materially lower than our best loop2600 baseline
- `find_unused_parameters=True` was likely only needed as a safety guard and showed DDP warnings about extra graph traversal
- the train-side idea looks real, but the size/quality tradeoff is not yet leaderboard-competitive

## Planned follow-up

If the first run is healthy, the next ablations are:
- `READOUT_SCALE` in `{0.25, 0.35, 0.5}`
- `READOUT_GROUPS` in `{8, 16, 32}`
- `TTT_PARAM_MODE=readout_only`

## Artifacts

- [train_seed42.log](./train_seed42.log)
- [train_gpt.py](./train_gpt.py)
