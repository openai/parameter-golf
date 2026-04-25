# K=3 L=3 + SwiGLU(mlp=8) — aggressive width push, uses ~12.7 MB int8 of cap.
# Per block: 0.79M attn + 6.30M swiglu(8) ≈ 7.09M; K=3 → 21.3M params.
# Estimated artifact ~12-13 MB int8 (vs cap 16 MB).
export RUN_ID="0062_swiglu_recur_3x3_mlp8"
export DATA_PATH="../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export ITERATIONS=200
export WARMUP_STEPS=0
export WARMDOWN_ITERS=300
export LR_WARMUP_STEPS=30
export MLP_MULT=8
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
export NUM_UNIQUE_LAYERS=3
export NUM_LOOPS=3
