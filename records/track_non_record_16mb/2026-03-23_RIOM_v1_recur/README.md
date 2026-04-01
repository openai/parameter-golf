This package is the first RIOM-flavored non-record variant. It replaces the baseline's 9 distinct blocks with 3 shared transformer blocks repeated 3 times, with a lightweight learned recurrence gate after each shared block application, while keeping the tokenizer, dataset, optimizer family, and export path unchanged.

Why this is RIOM-style:
- Effective depth is increased through parameter sharing instead of distinct weights.
- The 3x3 recurrence keeps the nominal 9-block depth budget while dropping parameter count and compressed bytes sharply.
- The change is isolated to the record-local script. Training flow, quantized export, and tokenizer-aware bpb accounting remain the same as the baseline package.

Architecture:
- shared transformer depth: `3`
- recurrence loops: `3`
- effective depth: `9`
- recurrence gate: per-loop, per-shared-block learned vector mixed through `sigmoid`
- `VOCAB_SIZE=1024`, `MODEL_DIM=512`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`, `MLP_MULT=2`
- tied embeddings preserved

Tokenizer and dataset:
- Tokenizer unchanged: official `fineweb_1024_bpe.model`
- Dataset unchanged: official `fineweb10B_sp1024`
- Training shards present locally: `1/195`
- Validation accounting for this smoke run: first `1,048,576` official validation tokens via `VAL_MAX_TOKENS`

Exact command used:
```bash
source ../../../.venv/bin/activate
RUN_ID=riom_v1_dev_mlx_20260323 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
OUT_DIR=. \
ITERATIONS=50 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
VAL_MAX_TOKENS=1048576 \
TRAIN_LOG_EVERY=10 \
python3 -u train_gpt.py
```

Hardware:
- Apple Silicon arm64 MacBook Air
- macOS 26.3.1
- Python 3.13.12
- MLX 0.31.1

Measured results from `train.log`:
- model params: `6,040,088`
- pre-quant: `val_loss=5.4143`, `val_bpb=3.2439`
- post-quant roundtrip: `val_loss=5.42207763`, `val_bpb=3.24862008`
- training time: `234.488s`
- final eval time: `105.973s`

Artifact size accounting:
- code bytes: `51,404`
- compressed model bytes: `2,273,437`
- total counted bytes: `2,324,841`
- raw MLX snapshot bytes: `23,119,132`

Comparison to v0 on the same dev prefix:
- `val_bpb`: `3.2749 -> 3.2486`
- total bytes: `6,258,232 -> 2,324,841`
- params: `17,059,912 -> 6,040,088`

What is unfinished:
- This is still a local development smoke, not an upstream-ready leaderboard submission.
- Validation was capped with `VAL_MAX_TOKENS`; rerun with `VAL_MAX_TOKENS=0` before any upstream submission.
- This MLX-only path demonstrates the idea locally, but a CUDA/PyTorch port is still required for a serious record attempt.

Next planned ablations:
- port this shared-depth recurrence into the official CUDA `train_gpt.py`
- tune recurrence gate initialization and loop-specific learning dynamics
- add sliding-window evaluation on top of this recurrent package
