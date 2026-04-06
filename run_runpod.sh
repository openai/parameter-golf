#!/usr/bin/env bash
# ============================================================
# run_runpod.sh — Full H100 training run for Parameter Golf
#
# Usage (RunPod terminal):
#   bash run_runpod.sh
#   RUN_ID=my_run bash run_runpod.sh
#   MDL_LAMBDA=0.1 bash run_runpod.sh
#
# What it does:
#   1. Installs zstandard if missing
#   2. Checks data is present (errors clearly if not)
#   3. Auto-detects GPU count
#   4. Runs train_gpt_stack.py for 20000 steps (wallclock-uncapped)
#   5. Saves logs to logs/<RUN_ID>.log
#   6. Prints final BPB and compressed size at end
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

# ── Environment ───────────────────────────────────────────────────────────────

# Activate venv if present (created by challenge setup)
if [[ -d .venv ]]; then
    source .venv/bin/activate
fi

# Install zstandard if missing (needed for zstd-22 compression)
python3 -c "import zstandard" 2>/dev/null || {
    echo "[setup] Installing zstandard..."
    pip install zstandard --quiet
}

# ── Configuration ─────────────────────────────────────────────────────────────

: "${RUN_ID:=runpod_$(date +%Y%m%d_%H%M%S)}"
: "${SCRIPT:=train_gpt_stack_v2.py}"
: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"
: "${VOCAB_SIZE:=1024}"
: "${ITERATIONS:=20000}"
: "${TRAIN_SEQ_LEN:=1024}"
: "${VAL_LOSS_EVERY:=1000}"
: "${WARMDOWN_ITERS:=3500}"
: "${MDL_LAMBDA:=5.0}"
: "${MDL_QUANT_BITS:=6}"
: "${EMA_DECAY:=0.997}"
: "${LATE_QAT_THRESHOLD:=0.15}"

# Auto-detect GPU count
N_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
echo "[setup] Detected ${N_GPUS} GPU(s)"

# Tokens per step: 524288 is standard for 8xH100.
# Scales with GPU count to keep per-GPU work constant.
TOKENS_PER_STEP=$((65536 * N_GPUS))

mkdir -p logs

# ── Pre-flight checks ─────────────────────────────────────────────────────────

echo "[check] Verifying data paths..."
if [[ ! -d "$DATA_PATH" ]]; then
    echo "ERROR: Data directory not found: $DATA_PATH"
    echo "Run: python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
    exit 1
fi
N_TRAIN_SHARDS=$(ls "$DATA_PATH"/fineweb_train_*.bin 2>/dev/null | wc -l || echo 0)
N_VAL_SHARDS=$(ls "$DATA_PATH"/fineweb_val_*.bin 2>/dev/null | wc -l || echo 0)
if [[ "$N_TRAIN_SHARDS" -eq 0 || "$N_VAL_SHARDS" -eq 0 ]]; then
    echo "ERROR: No train ($N_TRAIN_SHARDS) or val ($N_VAL_SHARDS) shards found in $DATA_PATH"
    echo "Run: python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
    exit 1
fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER_PATH"
    exit 1
fi
echo "[check] Found ${N_TRAIN_SHARDS} train shards, ${N_VAL_SHARDS} val shards"

# ── Run ───────────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  Parameter Golf — MDL-T Stack"
echo "========================================================"
echo "  RUN_ID           : $RUN_ID"
echo "  SCRIPT           : $SCRIPT"
echo "  GPUs             : $N_GPUS"
echo "  ITERATIONS       : $ITERATIONS"
echo "  TOKENS_PER_STEP  : $TOKENS_PER_STEP"
echo "  WARMDOWN_ITERS   : $WARMDOWN_ITERS"
echo "  MDL_LAMBDA       : $MDL_LAMBDA"
echo "  EMA_DECAY        : $EMA_DECAY"
echo "  LATE_QAT_THRESH  : $LATE_QAT_THRESHOLD"
echo "  Log              : logs/${RUN_ID}.log"
echo "========================================================"
echo ""

RUN_ID="$RUN_ID" \
ITERATIONS="$ITERATIONS" \
TRAIN_BATCH_TOKENS="$TOKENS_PER_STEP" \
TRAIN_SEQ_LEN="$TRAIN_SEQ_LEN" \
VAL_BATCH_SIZE=524288 \
VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
WARMDOWN_ITERS="$WARMDOWN_ITERS" \
MAX_WALLCLOCK_SECONDS=0 \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE="$VOCAB_SIZE" \
MDL_LAMBDA="$MDL_LAMBDA" \
MDL_QUANT_BITS="$MDL_QUANT_BITS" \
EMA_DECAY="$EMA_DECAY" \
LATE_QAT_THRESHOLD="$LATE_QAT_THRESHOLD" \
torchrun \
    --standalone \
    --nproc_per_node="$N_GPUS" \
    "$SCRIPT" 2>&1 | tee "logs/${RUN_ID}.log"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  RESULTS: $RUN_ID"
echo "========================================================"
echo "  Val BPB history:"
grep "val_bpb:" "logs/${RUN_ID}.log" | tail -6 | sed 's/^/    /'
echo ""
echo "  Final compressed size:"
grep -E "int6\+|int8\+" "logs/${RUN_ID}.log" | grep "bytes" | tail -2 | sed 's/^/    /'
echo "========================================================"

# ── Copy result to submission record ─────────────────────────────────────────

RECORD_DIR="records/track_10min_16mb/2026-03-26_MDL-T_Stack_EMA_GPTQ-lite_LateQAT"
if [[ -d "$RECORD_DIR" ]]; then
    cp "logs/${RUN_ID}.log" "$RECORD_DIR/train.log"
    echo ""
    echo "[record] Log copied to $RECORD_DIR/train.log"
    echo "[record] Update submission.json with final BPB before PR"
fi
