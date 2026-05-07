# HybridSSM SparseAttention MixedQuant

This record captures the `HybridSSM SparseAttention MixedQuant` submission.

Trainer changes in this snapshot:

* hybrid Transformer–SSM architecture with final-layer SSM replacement
* selective attention removal in selected intermediate transformer blocks
* heterogeneous feedforward allocation between transformer and SSM blocks
* selective mixed int8/int6 post-training quantization export
* full compressed roundtrip validation using the final quantized artifact
* 10-minute wallclock cap on 8xH100

## Configuration

* Layout: `VOCAB_SIZE=2048 NUM_LAYERS=8 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
* Transformer MLP: `MLP_MULT=3.25`
* SSM layers: `SSM_LAYERS=7`
* Sparse attention disabled on: `NO_ATTN_LAYERS=3,6`
* SSM state: `SSM_STATE_DIM=128`
* SSM MLP: `SSM_MLP_MULT=2`
* SSM groups: `SSM_NUM_GROUPS=8`
* Quantization: `USE_INT6=1`
* Batching: `TRAIN_BATCH_TOKENS=196608 TRAIN_SEQ_LEN=3072`

## Command (track-relevant params)

```bash
NCCL_IB_DISABLE=1 \
SEED=1337 \
NUM_LAYERS=8 \
MLP_MULT=3.25 \
NO_ATTN_LAYERS="3,6" \
SSM_STATE_DIM=128 \
SSM_LAYERS="7" \
SSM_MLP_MULT=2 \
SSM_NUM_GROUPS=8 \
USE_INT6=1 \
TRAIN_SEQ_LEN=3072 \
TRAIN_BATCH_TOKENS=196608 \
WARMDOWN_FRAC=0.30 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key metrics (from `train.log`)

* Timed training stopped at `17942/20000` steps due to the wallclock cap.
* Final pre-export eval at stop: `val_loss:2.3870`, `val_bpb:1.1933`
* Final mixed-quant roundtrip eval: `val_loss:2.4098`, `val_bpb:1.2047`
* Exact printed metric: `final_mixed_quant_zstd_roundtrip_exact val_bpb:1.20473575`
* Train time: `600016ms` (`step_avg:33.44ms`)
* Peak memory: `3469 MiB allocated`, `4032 MiB reserved`
* Serialized model mixed-quant+zstd: `15510808 bytes`
* Code size: `60510 bytes`
* Total submission size mixed-quant+zstd: `15571318 bytes`

## Training volume

* Global batch: `196608` tokens/step
* Total train tokens seen: `3527540736`

## Architectural notes

This submission focuses on architectural changes and compression-aware export rather than only tuning standard hyperparameters:

* Intermediate transformer ablations showed layers 3 and 6 contributed lower marginal utility under the fixed 10-minute training budget, enabling selective attention removal.
* Final-layer SSM replacement substitutes the final transformer attention block with a linear-time state-space module for sequence refinement.
* Selective attention removal removes selected intermediate attention blocks to reduce compute and increase training throughput under fixed wallclock.
* Heterogeneous MLP allocation assigns different expansion ratios to transformer and SSM blocks.
* Mixed int8/int6 quantization reduces artifact size while preserving model quality under the submission size cap.

## Included files

* `train_gpt.py` (code snapshot used for the run)
* `train.log` (exact training log)
* `submission.json` (submission metadata)
