# Non-record Submission: Faithful KV-cache Quantization Backends on 1x RTX 3090

This submission keeps the training path close to the stock `parameter-golf` baseline and moves the novelty into a new **teacher-forced autoregressive evaluation path with explicit KV cache**. The script implements three paper-inspired backends under a shared interface:

- `qjl`: JL-style random rotation + 1-bit sign sketch for keys + low-bit grouped value quantization
- `polar`: random rotation + recursive polar transform + angle quantization
- `turbo`: random rotation + scalar codebooks + QJL-style residual sketch

The implementation is intentionally inference-focused: it does **not** reinterpret these papers as weight quantizers, does **not** add Triton/custom CUDA kernels, and leaves the model artifact export on the existing `int8 + zlib` path.

## What Is In The Script

- Standard training forward path
- `forward_train`, `forward_logits`, `forward_prefill`, and `forward_decode`
- Explicit per-layer KV cache for autoregressive scoring
- `KV_QUANT_BACKEND={none,qjl,polar,turbo}`
- `KV_BITS_MODE`, `KV_ROTATION_SEED`, `KV_GROUP_SIZE`
- `KV_EVAL_CONTEXT_LEN`, `KV_EVAL_MAX_TOKENS`
- `KV_CACHE_BASELINE={float,int8_backend}`
- `KV_EVAL_COMPARE_BACKENDS` for evaluating multiple backends on the same checkpoint

## Local Environment

- Python: `py -3.11`
- PyTorch: `2.10.0+cu128`
- GPU: `1x RTX 3090 24GB`
- `torch.compile`: disabled by default in this local submission because Triton is not installed on this machine

## Data

- Tokenizer: `sp1024`
- Smoke run: `1` train shard
- Main local run: `80` train shards
- Validation: full frozen challenge val split is available locally, but the new autoregressive KV evaluation was **capped to 2048 tokens** for the local study below

## Main Local 10-minute Run

Training config used for the local run:

```bash
RUN_ID=run10m_kv3090 \
DATA_PATH=/abs/path/to/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/abs/path/to/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=100000 \
WARMUP_STEPS=0 \
TRAIN_BATCH_TOKENS=65536 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=600 \
KV_EVAL_CONTEXT_LEN=1024 \
KV_EVAL_MAX_TOKENS=2048 \
KV_EVAL_COMPARE_BACKENDS=none,qjl,polar,turbo \
python train_gpt.py
```

Key training results:

- Steps completed before wallclock stop: `570`
- Final teacher-forced val before quantized export: `val_bpb=1.6507`
- Compressed artifact size: `10,458,900` bytes
- Peak allocated GPU memory: `5980 MiB`

## Capped Autoregressive KV Evaluation

The table below uses the same trained checkpoint and the new autoregressive evaluator on a fixed `2048`-token validation prefix with `KV_EVAL_CONTEXT_LEN=1024`.

| Backend | val_bpb | eval_time_ms | tok/s | logical cache bytes | tensor cache bytes |
|---------|--------:|-------------:|------:|--------------------:|-------------------:|
| `none`  | `1.98159643` | 22095 | 92.69 | 9437184 | 9437184 |
| `qjl`   | `2.09637861` | 29919 | 68.45 | 1548288 | 5087232 |
| `polar` | `1.98346882` | 84451 | 24.25 | 3059712 | 4792320 |
| `turbo` | `1.99298916` | 59360 | 34.50 | 1990656 | 7299072 |

Observations from this local run:

- `polar` was the best quantized backend on this capped evaluation and stayed very close to the float-cache baseline.
- `turbo` reduced logical cache size substantially versus float while keeping the loss gap modest.
- `qjl` achieved the smallest logical cache among the three quantized backends here, but with the largest quality gap on this short local run.
- The new evaluator is substantially slower than the baseline sliding/full-window evaluation, especially for `polar`, because v1 intentionally uses pure PyTorch dequantization and no custom kernels.

## Smoke Validation

A short smoke run with `1` train shard and `2048` capped eval tokens completed successfully and exercised:

- training
- artifact serialization / reload
- `none`, `qjl`, `polar`, and `turbo` backends
- CUDA backend self-tests

Smoke log: `logs/smoke_kv3090.txt`  
Main run log: `logs/run10m_kv3090.txt`

## Limitations

- The KV evaluation in this folder is a **local capped study**, not a full official challenge evaluation.
- No Triton/custom CUDA kernels are used, so runtime is not competitive with the papers' optimized implementations.
- The default code path keeps `KV_QUANT_BACKEND=qjl`, but the best local capped result in this folder came from `polar`.
