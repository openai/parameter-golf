# 11L EMA + BigramHash(12288) + Mixed Int5 + FA3

This submission package documents a main-track 10-minute / 16MB run family.

The model is an 11-layer, 512-dim GQA transformer with:
- 8 heads / 4 KV heads
- MLP multiplier `3.0`
- tied embeddings
- EMA (`0.997`)
- BigramHash with `12288` buckets and dim `128`
- mixed low-bit quantization with `ATTN_QUANT_BITS=5` and `BIGRAM_QUANT_BITS=5`
- FP16 keep-pattern for `blocks.8.attn.c_k`
- stride-64 sliding evaluation

## FA3 Disclosure

This run family uses a FlashAttention-3 path through `kernels-community/flash-attn3`.

Important notes:
- this does **not** download model weights, training data, prompts, or any external user code
- it **does** fetch the FA3 kernel package at runtime on the pod
- the actual model code used for the run is fully included in `train_gpt.py`

This is disclosed explicitly here. If OpenAI prefers a strictly local-only dependency path, this family can be rerun with a preinstalled/local FA3 path.

## Run Command

```bash
# Example (seed 42)
SIMON_ENV_FILE=.env.8xh100_submission_r20_noprune \
bash pod_train.sh /dev/shm/parameter-golf-local/simon-submit-r20np-s42-rerun
```

## 3-Seed Results

| Seed | val_bpb | pre-quant val_bpb | total bytes | valid |
|------|---------|-------------------|-------------|-------|
| 42 | 1.13593695 | 1.1424 | 15,967,704 | yes |
| 471 | 1.13389376 | 1.1412 | 15,663,365 | yes |
| 777 | 1.13626774 | 1.1422 | 15,660,237 | yes |
| **Mean** | **1.13536615** | | | |
| **Std** | **0.00128581** | | | |

## Main Hyperparameters

- `TRAIN_BATCH_TOKENS=393216`
- `TRAIN_SEQ_LEN=2048`
- `MAX_WALLCLOCK_SECONDS=600`
- `WARMUP_STEPS=20`
- `WARMDOWN_FRAC=0.48`
- `MATRIX_LR=0.025`
- `SCALAR_LR=0.025`
- `MUON_MOMENTUM=0.99`
- `WEIGHT_DECAY=0.04`
- `BIGRAM_VOCAB_SIZE=12288`
- `EMA_ENABLED=1`
- `PRUNE_FRAC=0.00`
- `EVAL_STRIDE=64`

## Included Files

- `train_gpt.py`
- `env_utils.py`
- `pod_train.sh`
- `requirements.txt`
- `submission.json`
- `train_seed42.log`
- `train_seed471.log`
- `train_seed777.log`
