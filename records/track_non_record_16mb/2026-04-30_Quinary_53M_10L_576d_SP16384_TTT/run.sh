# Quinary (5-level) Parameter Golf submission.
#
# Defaults below match the canonical 53M-param model + per-stream v2
# layout-aware compression that produced the artifact in this folder:
#   - sp16384 vocab + tokenizer
#   - EMBED_DIM=380, MODEL_DIM=576, NUM_LAYERS=10, NUM_HEADS=6, NUM_KV_HEADS=3
#   - QK_GAIN_INIT=5.0, MATRIX_LR=0.035
#   - TTT_STEPS=3, TTT_LR=0.005, TTT_TOKENS=32768
#   - per-stream v2 archive (header byte 0x03):
#       * splits each bulk tensor into its own compressed payload
#       * for each quinary tensor, screens 4 layouts {base5, base5_T,
#         bitmask, bitmask_T} by LZMA9 size, then runs LZMA9 vs lrzip-zpaq
#         only on the winning layout (bounded heuristic, not exhaustive 4×2)
#       * for c_qkv.weight, splits rows into Q/K/V sub-payloads independently
#       * robust to the seed-dependent lrzip cliff (full-blob lrzip can OVER
#         on ~33% of seeds; per-stream v2 consistently FITS at ~15.64 MB)
#   - SCALE_QUANT_BITS=5 (per-group scale log-delta quant, saves ~141 KB
#     at +2.1 mBPB TTT cost; net Pareto-positive)
#
# To run with a different seed (e.g., for the 3-seed mean):
#   SEED=1337 bash run.sh

RUN_ID=${RUN_ID:-quinary_seed42} \
DATA_PATH=${DATA_PATH:-./data/canonical/datasets/fineweb10B_sp16384} \
TOKENIZER_PATH=${TOKENIZER_PATH:-./data/canonical/tokenizers/fineweb_16384_bpe.model} \
VOCAB_SIZE=${VOCAB_SIZE:-16384} \
BITNET_GROUP_SIZE=${BITNET_GROUP_SIZE:-192} \
EMBED_DIM=${EMBED_DIM:-380} \
NUM_LAYERS=${NUM_LAYERS:-10} \
MODEL_DIM=${MODEL_DIM:-576} \
NUM_KV_HEADS=${NUM_KV_HEADS:-3} \
NUM_HEADS=${NUM_HEADS:-6} \
MLP_MULT=${MLP_MULT:-4} \
MATRIX_OPTIMIZER=${MATRIX_OPTIMIZER:-muon} \
ADAM_LR=${ADAM_LR:-0.05} \
ADAM_WD=${ADAM_WD:-0.05} \
MUON_BACKEND_STEPS=${MUON_BACKEND_STEPS:-3} \
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95} \
MUON_MOMENTUM_WARMUP_START=${MUON_MOMENTUM_WARMUP_START:-0.85} \
MUON_MOMENTUM_WARMUP_STEPS=${MUON_MOMENTUM_WARMUP_STEPS:-500} \
MUON_WD=${MUON_WD:-0.0} \
MATRIX_LR=${MATRIX_LR:-0.035} \
SCALAR_LR=${SCALAR_LR:-0.02} \
TIED_EMBED_LR=${TIED_EMBED_LR:-0.02} \
WARMDOWN_FRACTION=${WARMDOWN_FRACTION:-0.2} \
LOGIT_SOFTCAP=${LOGIT_SOFTCAP:-10} \
QK_GAIN_INIT=${QK_GAIN_INIT:-5.0} \
ROPE_TYPE=${ROPE_TYPE:-yarn} \
YARN_MAX_LEN=${YARN_MAX_LEN:-2048} \
ROPE_BASE=${ROPE_BASE:-5000} \
BATCH_TOKENS_START=${BATCH_TOKENS_START:-0} \
BATCH_SCHEDULE_FRACTION=${BATCH_SCHEDULE_FRACTION:-0.33} \
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-524288} \
SEQ_LEN_START=${SEQ_LEN_START:-0} \
SEQ_SCHEDULE_FRACTION=${SEQ_SCHEDULE_FRACTION:-0.0} \
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-1024} \
ITERATIONS=${ITERATIONS:-10000} \
WARMUP_STEPS=${WARMUP_STEPS:-5} \
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-599} \
VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0} \
TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-1000} \
CHURN_LOG_EVERY=${CHURN_LOG_EVERY:-0} \
VAL_MAX_TOKENS=${VAL_MAX_TOKENS:-0} \
TIE_EMBEDDINGS=${TIE_EMBEDDINGS:-1} \
HEAD_LR=${HEAD_LR:-0.02} \
ACTIVATION=${ACTIVATION:-relu2} \
SOFTCAP_TYPE=${SOFTCAP_TYPE:-poly} \
TTT_STEPS=${TTT_STEPS:-3} \
TTT_LR=${TTT_LR:-0.005} \
TTT_TOKENS=${TTT_TOKENS:-32768} \
SCALE_QUANT_BITS=${SCALE_QUANT_BITS:-5} \
SEED=${SEED:-42} \
COMPILE_MODE=${COMPILE_MODE:-default} \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} torchrun --standalone --nproc_per_node=8 train_gpt.py
