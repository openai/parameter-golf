This draft folder documents an in-progress local Apple Silicon reproduction using the current root `train_gpt_mlx.py`.

This is not a finished non-record submission yet. It exists to show concrete reproduction work, local logs, and the exact script snapshot before moving to a completed cloud-backed run.

What is included:
- `train_gpt_mlx.py`: exact MLX script snapshot used for the local runs.
- `train_partial.log`: 200-step local smoke run on an Apple M1 with 1 FineWeb train shard and the full fixed validation split. Training completed through step 200 and then entered full validation.
- `eval_probe.log`: follow-up probe using a larger validation batch size to test the local 8GB memory / eval-time tradeoff.

Local machine:
- Apple M1 MacBook Air
- 8GB unified memory
- Python 3.14
- MLX 0.31.1

Smoke configuration:
- Tokenizer / dataset: `sp1024`, `fineweb10B_sp1024`
- Train shards: `1`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Training: `ITERATIONS=200 TRAIN_BATCH_TOKENS=8192`
- Validation: full fixed `fineweb_val_*` split

Observed outcome so far:
- Local training is stable and reproduces the baseline setup on Apple Silicon.
- Full final validation on an 8GB M1 is much slower than training, so this draft does not yet include a finished `submission.json`.
- The next step is to rerun the same baseline path on cloud GPUs and convert this folder into a completed non-record submission with final `val_bpb`, artifact bytes, and `submission.json`.

Command used for the main smoke run:
```bash
source .venv/bin/activate
RUN_ID=stukenov_mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
TRAIN_LOG_EVERY=50 \
python train_gpt_mlx.py
```

Command used for the eval probe:
```bash
source .venv/bin/activate
RUN_ID=stukenov_mlx_probe \
ITERATIONS=1 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=65536 \
TRAIN_LOG_EVERY=1 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt_mlx.py
```
