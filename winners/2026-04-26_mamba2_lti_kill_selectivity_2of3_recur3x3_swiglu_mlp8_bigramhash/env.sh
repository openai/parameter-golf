# Source this from inside the experiment folder before running.
export RUN_ID="0038_mamba2_kill_selectivity"
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
export WARMDOWN_ITERS=1200
# Wallclock cap disabled so lr_mul uses the step-based warmdown branch (the
# wallclock branch's formula doesn't fire for short smokes — see git log).
export MAX_WALLCLOCK_SECONDS=0
export TRAIN_BATCH_TOKENS=8192
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
# Inherited from 0012:
export WARMDOWN_ITERS=300
export LR_WARMUP_STEPS=30
export TIED_EMBED_INIT_STD=0.05
export MUON_BACKEND_STEPS=15
export TRAIN_BATCH_TOKENS=24576
export MATRIX_LR=0.045
export CONTROL_TENSOR_NAME_PATTERNS="attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,A_log,A_im,B_proj,C_proj,dt_log,D_skip,dt_bias,delta_bias,conv1d"
export NUM_UNIQUE_LAYERS=3
export NUM_LOOPS=3
export MLP_MULT=8
export MLP_TYPE=swiglu
# 0035: 2 of 3 unique blocks are Mamba-2/SSD; 1 attention at end (position 2).
# Pattern per K=3 group: MAMBA2-MAMBA2-ATTN -> looped 3x = 6 mamba2 + 3 attn.
# Tests whether the Mamba-2 win compounds vs the 1-of-3 sandwich (0032/0034).
export ATTN_LAYER_POSITIONS=2
export MAMBA2_LAYER_POSITIONS=0,1
# 0018: BigramHash recall augmentation
export BIGRAM_VOCAB_SIZE=4096
export BIGRAM_DIM=64
# 0038: KILL Mamba-2's selectivity. Replace input-dependent (dt, B, C) with
# learned per-head/per-state constants. Same in_proj/conv1d/out_proj/A_log/
# dt_bias/D_skip as full Mamba-2 — only the dynamics become LTI. Decisive
# ablation: if val ≈ 2.04 win is parameters/structure, if val ≈ 2.16 the
# selectivity IS the mechanism. _B_const/_C_const are 1D, auto-fp32.
export MAMBA2_KILL_SELECTIVITY=1
