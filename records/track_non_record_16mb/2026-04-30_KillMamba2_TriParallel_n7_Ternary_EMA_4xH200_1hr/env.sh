# Canonical environment for this submission.
#
# Run from the REPO ROOT (where data/ lives):
#
#   source records/track_non_record_16mb/2026-04-30_KillMamba2_TriParallel_n7_Ternary_EMA_4xH200_1hr/env.sh
#   torchrun --standalone --nproc_per_node=4 \
#     records/track_non_record_16mb/2026-04-30_KillMamba2_TriParallel_n7_Ternary_EMA_4xH200_1hr/train_gpt.py
#
# train_gpt.py defaults DATA_PATH and TOKENIZER_PATH to ./data/datasets/fineweb10B_sp1024
# and ./data/tokenizers/fineweb_1024_bpe.model — both resolve from repo root, matching
# the convention of the other records-folder submissions.

# --- Identity / data ---
export RUN_ID="kill_mamba2_n7_ternary_ema_h200_1hr"
export SEED=1337
export VOCAB_SIZE=1024
export TRAIN_SEQ_LEN=1024

# --- Topology: 7 unique blocks × 3 weight-shared loops; parallel(attn || kill-Mamba-2) at every position ---
export NUM_UNIQUE_LAYERS=7
export NUM_LOOPS=3
export ATTN_LAYER_POSITIONS=
export MAMBA2_LAYER_POSITIONS=
export PARALLEL_LAYER_POSITIONS=0,1,2,3,4,5,6
export PARALLEL_SSM_TYPE=mamba2_kill
export MAMBA2_KILL_SELECTIVITY=1
export MLP_TYPE=swiglu
export MLP_MULT=8
export BIGRAM_VOCAB_SIZE=0   # off — interacts negatively with conv1d already in Mamba-2

# --- Quantization: BitNet-b1.58 ternary body, exported 2-bit packed ---
export TERNARY_BODY=1
# Tensors matching these patterns stay fp32 throughout (1D / small, ≤65,536 elem).
# Load-bearing for SSM dynamics buffers (A_log, B_proj, C_proj, dt_bias, D_skip, conv1d, etc.).
export CONTROL_TENSOR_NAME_PATTERNS="attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,A_log,A_im,B_proj,C_proj,dt_log,D_skip,dt_bias,delta_bias,conv1d"

# --- EMA-of-weights, β=0.999 (effective averaging window ~1000 steps; shadow swapped before final eval) ---
export EMA_BETA=0.999

# --- Schedule / optimizer ---
export TRAIN_BATCH_TOKENS=524288
export ITERATIONS=20000              # upper bound; the wallclock cap below is the binding limit
export MAX_WALLCLOCK_SECONDS=3600    # 1 hour
export WARMDOWN_ITERS=1800
export LR_WARMUP_STEPS=30
export WARMUP_STEPS=0                # batch-size warmup (separate from LR warmup); off
export MATRIX_LR=0.045
export TIED_EMBED_INIT_STD=0.05
export MUON_BACKEND_STEPS=15

# --- Eval ---
export VAL_TOKENS=0                  # 0 = full validation set, writeup-quality
export VAL_LOSS_EVERY=0              # no mid-training val (eval runs once at the end after EMA swap)
