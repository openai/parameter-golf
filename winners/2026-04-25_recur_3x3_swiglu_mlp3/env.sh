# Source this from inside the experiment folder before running.
# K=3 L=3 depth recurrence (from 0056) + SwiGLU(mlp=3) MLP_TYPE swiglu.
# Each block has SwiGLU(mlp_mult=3): w_gate(d, 3d) + w_up(d, 3d) + w_down(3d, d) ≈ 2.36M.
# 3 unique blocks looped 3 times = effective depth 9.
# Expected param count ~10M, artifact ~6 MB int8.
export RUN_ID="0057_swiglu_recur_3x3"
export DATA_PATH="../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export ITERATIONS=200
export WARMUP_STEPS=0
export WARMDOWN_ITERS=300
export LR_WARMUP_STEPS=30
export MLP_MULT=3
export MLP_TYPE=swiglu
export MATRIX_LR=0.045
export TIED_EMBED_INIT_STD=0.05
export MUON_BACKEND_STEPS=15
export MAX_WALLCLOCK_SECONDS=0
export TRAIN_BATCH_TOKENS=24576
export TRAIN_SEQ_LEN=1024
export VAL_BATCH_SIZE=8192
export VAL_LOSS_EVERY=0
export VAL_TOKENS=16384
export TRAIN_LOG_EVERY=5
# Depth recurrence config (inherited from 0056):
export NUM_UNIQUE_LAYERS=3
export NUM_LOOPS=3
