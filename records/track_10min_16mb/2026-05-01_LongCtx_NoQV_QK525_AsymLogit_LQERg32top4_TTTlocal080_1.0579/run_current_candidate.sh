#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-longctx_noqv_qk525_asym_lqer_g32_top4_tttlocal080_seed${SEED}}"
CASEOPS_ROOT="${CASEOPS_ROOT:-/workspace/caseops_data}"

export SEED RUN_ID
export DATA_DIR="${DATA_DIR:-${CASEOPS_ROOT}/datasets}"
export DATA_PATH="${DATA_PATH:-${CASEOPS_ROOT}/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"

export CASEOPS_ENABLED=1
export VOCAB_SIZE=8192
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600
export GPTQ_RESERVE_SECONDS=4.0
export VAL_LOSS_EVERY=0
export COMPRESSOR=pergroup

export WARMDOWN_FRAC=0.85
export BETA2=0.99
export GRAD_CLIP_NORM=0.3
export MIN_LR=0.1
# delta vs parent (#2007): MATRIX_LR 0.026 -> 0.028
export MATRIX_LR=0.028
export QK_GAIN_INIT=5.25

export SPARSE_ATTN_GATE_ENABLED=1
export SPARSE_ATTN_GATE_SCALE=0.5
export SMEAR_GATE_ENABLED=1
export GATE_WINDOW=12
export GATED_ATTN_QUANT_GATE=1
export FUSED_CE_ENABLED=1
export ASYM_LOGIT_RESCALE=1

export EMBED_BITS=7
export MLP_CLIP_SIGMAS=11.5
export ATTN_CLIP_SIGMAS=13.0
export EMBED_CLIP_SIGMAS=14.0
export LQER_ENABLED=1
export LQER_ASYM_ENABLED=1
# delta vs parent (#2007): LQER_RANK 4 -> 2
export LQER_RANK=2
export LQER_FACTOR_BITS=4
# delta vs parent (#2007): LQER_ASYM_GROUP 64 -> 32
export LQER_ASYM_GROUP=32
# delta vs parent (#2007): LQER_TOP_K 3 -> 4
export LQER_TOP_K=4
export AWQ_LITE_ENABLED=1
export AWQ_LITE_BITS=8
export AWQ_LITE_GROUP_TOP_K=1
export AWQ_LITE_GROUP_SIZE=64

export EVAL_SEQ_LEN=2560
export TTT_EVAL_SEQ_LEN=2560
export PHASED_TTT_ENABLED=1
export PHASED_TTT_NUM_PHASES=3
export PHASED_TTT_PREFIX_DOCS=3000
export TTT_MASK=no_qv
# delta vs parent (#2007): TTT_LOCAL_LR_MULT 0.75 -> 0.80
export TTT_LOCAL_LR_MULT=0.80
export TTT_CHUNK_SIZE=48
export TTT_BETA2=0.99
export TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80
export TTT_LORA_ALPHA=144
export MUON_BACKEND_STEPS=5
export NCCL_NET=Socket

torchrun --standalone --nproc_per_node=8 train_gpt.py
