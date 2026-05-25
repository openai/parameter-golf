# UT7 RLMA320 K6 SP8192 75k WD15k — 3-Seed Non-Record

This is a non-record / unlimited-compute update for PR #2114. It reports one fixed UT7 configuration trained from scratch for the required seeds `42`, `314`, and `999` on 8xH100, with the result summarized by the 3-seed mean rather than a single best run.

The run is not a 10-minute leaderboard claim. Training used `MAX_WALLCLOCK_SECONDS=0`, `ITERATIONS=75000`, and `WARMDOWN_ITERS=15000`, so warmdown starts at step `60000`. Validation was disabled during training with `VAL_LOSS_EVERY=0`; validation appears only after each seed reaches `step:75000/75000`.

## Results

3-seed quant mean: `val_loss=3.19216775`, `val_bpb=1.23577911`, sample std `0.00134704` bpb.

3-seed FP mean: `val_loss=3.15644855`, `val_bpb=1.22195119`, sample std `0.00172696` bpb.

| seed | FP bpb | quant bpb | total bytes | headroom | train time |
|---:|---:|---:|---:|---:|---:|
| 42 | `1.22000161` | `1.23437923` | `15,731,914` | `268,086` | `242.5m` |
| 314 | `1.22256300` | `1.23589187` | `15,782,047` | `217,953` | `239.3m` |
| 999 | `1.22328895` | `1.23706622` | `15,736,978` | `263,022` | `239.3m` |

## Selected Artifact

The included artifact is the cap-valid seed-42 `GPTQ_CLIP_K=13` artifact:

| file | compressed bytes | code bytes | total bytes | headroom | sha256 |
|---|---:|---:|---:|---:|---|
| `final_model.rlma_int6_int8.zst` | `15,669,084` | `62,830` | `15,731,914` | `268,086` | `bc7353433da4988f5dbb551b83f98f575e8f083e7cfee9a8319d97050877fafb` |

Seed 42 first produced an over-cap `GPTQ_CLIP_K=12.5` artifact in `train_seed42.log` (`16,143,995` total bytes). The reported seed-42 quant score and included artifact come from the single fixed `GPTQ_CLIP_K=13` requantization of that seed's saved FP state, recorded in `train_seed42_quant_clip13.log`. No final quant sweep was run.

## Configuration

The three seeds used the same configuration:

```text
VOCAB_SIZE=8192
DATA_PATH=../../../data/datasets/fineweb10B_sp8192
TOKENIZER_PATH=../../../data/tokenizers/fineweb_8192_bpe.model
MODEL_DIM=1024 D_FF=3072 NUM_HEADS=8 NUM_KV_HEADS=4 HEAD_DIM=128
USE_RLMA=1 USE_TTT=0 ADAPTER_RANK=320 K_ITERS=6 TTT_CHUNK_SIZE=32
UT_RESIDUAL_DELTA=1 BRANCH_SCALE_INIT=0.6
TRAIN_SEQ_LEN=8192 TRAIN_BATCH_TOKENS=262144 GRAD_ACCUM_STEPS=1
MATRIX_LR=0.026 GRAD_CLIP_NORM=0.2
ITERATIONS=75000 WARMUP_STEPS=500 WARMDOWN_ITERS=15000
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=0
EVAL_CTX=8192 EVAL_STRIDE=8192 SW_ATTN_WINDOW_EVAL=512 VAL_TOKEN_LIMIT=0
GPTQ_BITS=8 GPTQ_CLIP_K=13 EMBED_QUANT_BITS=8 ZSTD_LEVEL=22
TARGET_ARTIFACT_BYTES=16000000 FINAL_FP_EVAL=1 QUANT_SWEEP_SPECS=""
SAVE_FP_CHECKPOINT_EVERY=0 GPTQ_CALIB_TOKENS=0
```

Model family: UT7 delta residual with RLMA rank 320, six recurrence iterations, custom SP8192 SentencePiece tokenizer, and no TTT path enabled.

Quantization is GPTQ-style int8 metadata with mixed int6/int8 packed tensors and zstd level 22 compression. The scored load, decompression, dequantization, and roundtrip evaluation path lives inside `train_gpt.py`; `requirements.txt` only adds `zstandard>=0.22.0`.

## Validation and BPB

The custom tokenizer is SP8192. The metric is still byte-normalized: `train_gpt.py` evaluates the official validation token stream and reports bits per byte using the trainer's SentencePiece byte accounting. The logged full-validation pass uses:

```text
val_tokens:40534017 eval_ctx:8192 eval_stride:8192 val_token_limit:0
```

No `val_tokens:` line appears before the final `step:75000/75000` line in the three training logs. The seed-42 requant log has `iterations=0` and loads the saved FP state only to rerun final FP eval and the cap-valid quant roundtrip.

The final quant roundtrip eval wall-clock stayed well under the 600 second evaluation limit in all logs: seed 42 `71301ms`, seed 314 `71149ms`, seed 999 `71145ms`.

## Included Files

- `train_gpt.py`: exact trainer used for the run, SHA256 `1098ab9ebb45d61e9f883220adc19df3208b826f4620f05659759b4d8bf275d1`, `1485` lines, `62,830` bytes.
- `requirements.txt`: local package dependency for zstd compression.
- `submission.json`: 3-seed mean metadata plus selected artifact metadata.
- `train_seed42.log`, `train_seed314.log`, `train_seed999.log`: from-scratch seed logs.
- `train_seed42_quant_clip13.log`: cap-valid seed-42 fixed requantization log.
- `final_model.rlma_int6_int8.zst`: selected seed-42 compressed artifact.

This folder intentionally excludes FP checkpoints, controller scripts, smoke outputs, manifests, local pull archives, quant sweep CSVs, and non-selected seed artifacts.
