# Non-Record Submission: SP8192 Trajectory Readout + Late Loop Onset

**Author:** Artem Buldin ([@Buld1n](https://github.com/Buld1n))  
**Track:** `non_record_16mb`  
**Status:** completed, negative result

## What This Is

This variant keeps the recurrent SP8192 backbone but adds a tiny grouped trajectory-delta readout on the final looped block.

For the final recurrent pass, it stores the hidden state from each visit to the loop-end block and applies a grouped correction:

`x <- x_final + scale * Σ_g,p alpha[g,p] * (x_pass[p] - x_final)`

Where:
- `x_pass[p]` is the hidden state from an earlier recurrent pass
- `x_final` is the final pass hidden state at the same loop boundary
- `alpha[g,p]` is a learned grouped coefficient

The goal is to let later layers recover useful information that may be present in earlier trajectory states but only partially preserved in the final recurrent state.

## Observed Result

Server run: `sp8192_trajreadout_l2600_s42_r1`

- `3507` steps in the effective training window
- `3507/20000 val_bpb = 1.1016`
- `pre-quantization post-ema val_bpb = 1.10244985`
- `quantized val_bpb = 1.11301355`
- `Serialized model quantized+brotli = 15,964,675 bytes`
- `Total submission size quantized+brotli = 16,064,921 bytes`

This direction did not become competitive: train-side quality regressed materially, and the packed artifact still missed the 16 MB limit.

## Distinctives vs prior local runs

- no pass-gated recurrence
- no token routing
- no shared adapter
- no auxiliary exit
- late loop onset at step `2600`, chosen from our own onset sweep
- optional `readout_only` / `loop_readout` legal TTT modes for follow-up ablations

## Reproduced Run

```bash
RUN_ID=sp8192_trajreadout_l2600_s42 \
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
USE_PASS_READOUT=1 \
READOUT_GROUPS=16 \
READOUT_SCALE=0.35 \
FIND_UNUSED_PARAMETERS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Follow-up That Was Motivated By This Run

The immediate follow-up ablations after this run were:
- `READOUT_SCALE` in `{0.25, 0.35, 0.5}`
- `READOUT_GROUPS` in `{8, 16, 32}`
- `TTT_PARAM_MODE=readout_only`

## Included Files

- `README.md`
- `requirements.txt`
- `train_gpt.py`
- `train_seed42.log`
- `submission.json`
