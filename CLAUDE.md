# Parameter Golf Project

## What this is
OpenAI competition to train the best LM that fits in 16MB.
Baseline: train_gpt_mlx.py (stock), my version: train_gpt_mlx_kl.py

## Commands
- Smoke test (baseline): `RUN_ID=test ITERATIONS=100 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 WARMUP_STEPS=3 python3 train_gpt_mlx_kl.py`
- Smoke test (moonshot): `RUN_ID=test ITERATIONS=100 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 WARMUP_STEPS=3 ENGRAM_LITE_ENABLED=1 COMPLEMENT_ALPHA=0.5 NGRAM_MIXER_ENABLED=1 python3 train_gpt_mlx_kl.py`
- Must activate venv first: `source ~/pg_env/bin/activate`  (venv is at ~/pg_env, NOT .venv)
- Data is in ./data/datasets/fineweb10B_sp1024/
- Full moonshot run (8×H100): `ENGRAM_LITE_ENABLED=1 COMPLEMENT_ALPHA=0.5 NGRAM_MIXER_ENABLED=1 NGRAM_ALPHA=0.25 NGRAM_MAX_ORDER=4 python3 train_gpt_mlx_kl.py`

## Key metrics
- train_loss: lower is better, compare at same step count
- val_bpb: the competition metric, but eval takes 36 min on M1
- Artifact size must be < 16,000,000 bytes

## Hardware
- Mac Mini M1 16GB — use for smoke tests only
- Real runs happen on RunPod 8×H100
