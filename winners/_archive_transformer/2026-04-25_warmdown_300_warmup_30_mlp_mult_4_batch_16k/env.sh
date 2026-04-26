# Source this from inside the experiment folder before running.
export RUN_ID="0020_warmdown_300_warmup_30"
export DATA_PATH="../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export ITERATIONS=200
export WARMUP_STEPS=0
# WARMDOWN_ITERS >= ITERATIONS triggers the step-based warmdown from step 0,
# yielding an effective LR ramp of (1200 - step) / 1200 across the whole run
# (0.167 at step 0, ~0 by step 200). This is the regime the Apr-18 reference
# baseline ran in (ITERATIONS=200, WARMDOWN_ITERS unset → defaulted to 1200),
# and it's what our smoke needs: full canonical LR (warmdown_iters << iterations)
# is too aggressive for MPS bf16 numerics and NaNs around step 165.
# To run an experiment at full canonical LR, override per-experiment with a
# small WARMDOWN_ITERS plus an explicit LR_WARMUP_STEPS (10–20).
export WARMDOWN_ITERS=300
export LR_WARMUP_STEPS=30
export MLP_MULT=4
# Wallclock cap disabled so lr_mul uses the step-based warmdown branch (the
# wallclock branch's formula doesn't fire for short smokes — see git log).
export MAX_WALLCLOCK_SECONDS=0
export TRAIN_BATCH_TOKENS=16384
export TRAIN_SEQ_LEN=1024
export VAL_BATCH_SIZE=8192
export VAL_LOSS_EVERY=0
# 16384-token val cap keeps eval ~1 s. Do NOT set to 0 — full val on MPS
# takes 60-120 min per experiment (eval is called twice: pre-quant + post-
# int8-quant). Use SEED=42 re-run for marginal-result confirmation instead.
export VAL_TOKENS=16384
# Dense step logs to catch divergence early; default 200 prints only steps
# 1–10 and step 200, leaving the bulk of training invisible.
export TRAIN_LOG_EVERY=5
# Experiment-specific overrides go below:
