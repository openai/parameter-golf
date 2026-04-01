This non-record submission captures the strongest 8-GPU plain baseline result we obtained so far in the official-style FineWeb path while staying under the 16MB artifact cap.

## Summary

This run uses the official `parameter-golf` `train_gpt.py` workflow on cached FineWeb and reports the repository's tokenizer-aware `val_bpb` metrics.

This is not a 10-minute leaderboard claim. It is an 8xH100 non-record candidate that reflects the strongest compliant baseline point reached in the current 8-GPU recipe search.

## Setup

Hardware / execution regime:

- 8x H100 80GB
- Runpod remote machine
- FineWeb cached subset with `--train-shards 1`

Required multi-GPU NCCL compatibility flags on this machine:

- `NCCL_P2P_DISABLE=1`
- `NCCL_IB_DISABLE=1`
- `NCCL_NVLS_ENABLE=0`
- `NCCL_CUMEM_ENABLE=0`

Main baseline configuration:

- `NUM_LAYERS=5`
- `MODEL_DIM=672`
- `NUM_HEADS=12`
- `NUM_KV_HEADS=4`
- `MLP_MULT=3`
- `TRAIN_BATCH_TOKENS=4194304`
- `TRAIN_SEQ_LEN=1024`
- `TRAIN_ON_VAL=1`
- `QAT_ENABLE=1`
- `QAT_START_FRAC=0.1`
- `QK_GAIN_INIT=3.5`
- `BETA2=0.99`
- `MATRIX_LR=0.06`
- `SCALAR_LR=0.06`
- `MUON_MOMENTUM=0.95`
- `MUON_MOMENTUM_WARMUP_START=0.85`
- `MUON_MOMENTUM_WARMUP_STEPS=100`
- `GRAD_CLIP_NORM=0.5`
- `ITERATIONS=505`
- `RUN_TTT_EVAL=1`
- `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS=tok_emb,attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights`

## Main Result

Primary reported result:

- `model_params: 20,271,612`
- `final_int8_ttt_lora val_loss: 2.3343`
- `final_int8_ttt_lora val_bpb: 1.3825`
- `Total submission size int8+zlib: 15,818,566 bytes`

Reference exact post-quant metric from the same run:

- `final_int8_zlib_roundtrip_exact val_loss: 2.40189046`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.42253482`

This remains below the 16,000,000-byte artifact cap.

## Interpretation

This should be interpreted conservatively.

- It is a non-record run.
- It uses only 1 training shard of cached FineWeb.
- It is 8-GPU candidate work, not the official final 10-minute leaderboard claim.
- The strongest reported metric in this run comes from the repository's `final_int8_ttt_lora` evaluation path.

## Included Files

- `train_gpt.py`: official-path training script used for this run
- `train.log`: exact training log for this baseline result
- `submission.json`: metadata for this non-record result
