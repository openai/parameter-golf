#!/usr/bin/env bash
# === Parameter Golf: MoS + SOTA Techniques on 1x/8x H100 (RunPod) ===
# Tests Mixture of Softmax (K=2) with full SOTA technique stack.
#
# Usage on RunPod:
#   git clone https://github.com/User123331/runpod-testing.git
#   cd runpod-testing
#   bash run_mos_sota.sh
#
# Modes:
#   MODE=baseline bash run_mos_sota.sh   # SOTA stack without MoS (control)
#   MODE=mos      bash run_mos_sota.sh   # SOTA stack + MoS K=2 (experiment)
#   MODE=smoke    bash run_mos_sota.sh   # Quick 300s smoke test with MoS

set -euo pipefail

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Keep-alive heartbeat: prevents RunPod from killing pod during long builds
(while true; do sleep 60; nvidia-smi > /dev/null 2>&1; done) &
KEEPALIVE_PID=$!
trap "kill ${KEEPALIVE_PID} 2>/dev/null" EXIT

MODE="${MODE:-mos}"
SEED="${SEED:-1337}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-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}}"

case "${MODE}" in
    baseline)
        USE_MOS=0
        MOS_K=2
        BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
        RUN_TAG="sota_baseline"
        ;;
    mos)
        USE_MOS=1
        MOS_K="${MOS_K:-2}"
        BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1024}"  # reduced to fit MoS in 16MB
        RUN_TAG="sota_mos_k${MOS_K}"
        ;;
    smoke)
        USE_MOS=1
        MOS_K="${MOS_K:-2}"
        BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1024}"
        MAX_WALLCLOCK_SECONDS=300
        RUN_TAG="sota_mos_smoke"
        ;;
    *)
        echo "Unknown MODE=${MODE}. Use: baseline, mos, smoke" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt_mos_sota.py"
RUN_ID="${RUN_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[ -f "${TRAIN_SCRIPT}" ] || { echo "ERROR: ${TRAIN_SCRIPT} not found"; exit 1; }

# Ensure deps
python3 -c "import huggingface_hub, zstandard, sentencepiece, numpy" 2>/dev/null || \
    pip install --quiet huggingface_hub zstandard sentencepiece numpy --break-system-packages

# Build FA3 (selective, ~5 min) if not already installed
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    log "FA3 not found. Building selectively (~5 min)..."
    FA3_DIR="${HOME}/flash-attention"
    if [ ! -d "${FA3_DIR}" ]; then
        git clone https://github.com/Dao-AILab/flash-attention.git "${FA3_DIR}"
    fi
    cd "${FA3_DIR}/hopper"
    rm -rf build/  # clear any stale full-build artifacts
    mkdir -p flash_attn_3  # pip copies .so here; dir must exist
    # Only build bf16 hdim64 SM90 causal — skip everything else
    export FLASH_ATTENTION_DISABLE_FP16=TRUE
    export FLASH_ATTENTION_DISABLE_FP8=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
    export FLASH_ATTENTION_DISABLE_SM80=TRUE
    export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
    export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
    export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
    export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
    export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
    export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
    export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
    export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
    pip install --no-build-isolation --break-system-packages -e .
    cd "${SCRIPT_DIR}"
    log "FA3 build complete."
else
    log "FA3 already installed."
fi

# Download dataset if needed
DATA_DIR="data/datasets/fineweb10B_sp1024"
TOK_PATH="data/tokenizers/fineweb_1024_bpe.model"
if [ ! -f "${DATA_DIR}/fineweb_train_000000.bin" ] || [ ! -f "${TOK_PATH}" ]; then
    log "Downloading FineWeb dataset..."
    if [ -n "${HF_TOKEN}" ]; then export HF_TOKEN; fi
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
log "Detected ${GPU_COUNT} GPU(s). Mode: ${MODE}"
log "MoS: USE_MOS=${USE_MOS} MOS_K=${MOS_K} BIGRAM_VOCAB_SIZE=${BIGRAM_VOCAB_SIZE}"
log "Run ID: ${RUN_ID}"

export PYTHONUNBUFFERED=1
export RUN_ID
export DATA_PATH="./${DATA_DIR}"
export TOKENIZER_PATH="./${TOK_PATH}"
export VOCAB_SIZE=1024
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3.0
export TIE_EMBEDDINGS=1
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export BIGRAM_VOCAB_SIZE
export BIGRAM_DIM=128
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3000
export ITERATIONS=9000
export MAX_WALLCLOCK_SECONDS
export EVAL_STRIDE=64
export SWA_ENABLED=1
export SWA_EVERY=50
export MUON_WD=0.04
export ADAM_WD=0.04
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export LATE_QAT_THRESHOLD=0.1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export USE_MOS
export MOS_K
export SEED
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"  # Disable torch.compile by default (fixes inductor issues)

log "Starting training..."
log "Log file: ${LOG_PATH}"
torchrun --standalone --nproc_per_node="${GPU_COUNT}" "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_PATH}"
TRAIN_EXIT=${PIPESTATUS[0]}

log "Training finished (exit code: ${TRAIN_EXIT}). Key metrics:"
grep -E 'val_bpb|model_params|mos_params|final_int|submission|Serialized|artifact|swa:' "${LOG_PATH}" | tail -20 || true

log "Done. Log: ${LOG_PATH}"
