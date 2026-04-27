#!/usr/bin/env bash
# Launch N independent train_gpt_0427.py runs (one per GPU, one per seed) in
# detached tmux sessions. Each run is single-process, so world_size=1 and
# grad_accum_steps=8 — effective batch matches an 8xH100 run.

set -u

# === Configuration ====================================================
REPO=${REPO:-/project/flame/xingyuad/parameter-golf}
VENV=${VENV:-/project/flame/xingyuad/Efficient-Distillation/hugging-face-trl/openr1_2/bin/activate}
SCRIPT=${SCRIPT:-train_gpt_0427.py}
SEEDS=(${SEEDS_OVERRIDE:-1337 1338 1339})
GPUS=(${GPUS_OVERRIDE:-0 1 2})
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-4800}
TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-100}
VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-500}
VOCAB_SIZE=${VOCAB_SIZE:-8192}
EXTRA_ENV=${EXTRA_ENV:-}              # e.g. EXTRA_ENV="ITERATIONS=3000"

# === Pre-flight =======================================================
[ -f "$VENV"   ] || { echo "ERROR: venv activate not found: $VENV";   exit 1; }
[ -d "$REPO"   ] || { echo "ERROR: repo not at: $REPO";               exit 1; }
[ -f "$REPO/$SCRIPT" ] || { echo "ERROR: training script not at: $REPO/$SCRIPT"; exit 1; }
[ "${#SEEDS[@]}" -eq "${#GPUS[@]}" ] || { echo "ERROR: SEEDS and GPUS must be same length"; exit 1; }

cd "$REPO"
mkdir -p logs

DATA_DIR="$REPO/data/datasets/fineweb10B_sp${VOCAB_SIZE}"
TOK="$REPO/data/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model"
TRAIN_N=$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_N=$(ls "$DATA_DIR"/fineweb_val_*.bin     2>/dev/null | wc -l)
[ "$TRAIN_N" -ge 1 ] || { echo "ERROR: no train shards at $DATA_DIR
       run: python3 data/cached_challenge_fineweb.py --variant sp${VOCAB_SIZE} --train-shards 10"; exit 1; }
[ "$VAL_N"   -ge 1 ] || { echo "ERROR: no val shards at $DATA_DIR";   exit 1; }
[ -f "$TOK" ]       || { echo "ERROR: tokenizer not at $TOK";        exit 1; }

# === Launch ===========================================================
TS=$(date +%Y%m%d_%H%M%S)
echo "Run timestamp: $TS"
echo "Wall clock per run: ${MAX_WALLCLOCK_SECONDS}s"
echo

# Kill leftover sessions with the same names
for SEED in "${SEEDS[@]}"; do
  tmux has-session -t "pg_s${SEED}" 2>/dev/null && {
    echo "Killing existing session pg_s${SEED}"
    tmux kill-session -t "pg_s${SEED}"
  }
done

for i in "${!SEEDS[@]}"; do
  SEED="${SEEDS[$i]}"
  GPU="${GPUS[$i]}"
  RUN_ID="0427_seed${SEED}_gpu${GPU}_${TS}"
  SESSION="pg_s${SEED}"
  LOG="${REPO}/logs/${RUN_ID}.log"

  CMD="source ${VENV} \
    && cd ${REPO} \
    && PYTHONUNBUFFERED=1 \
       CUDA_VISIBLE_DEVICES=${GPU} \
       SEED=${SEED} \
       RUN_ID=${RUN_ID} \
       MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} \
       TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY} \
       VAL_LOSS_EVERY=${VAL_LOSS_EVERY} \
       VOCAB_SIZE=${VOCAB_SIZE} \
       ${EXTRA_ENV} \
       python -u ${SCRIPT} 2>&1 | tee ${LOG}"

  tmux new-session -d -s "${SESSION}" "${CMD}"
  echo "  spawned ${SESSION}  GPU=${GPU}  SEED=${SEED}  →  ${LOG}"
done

echo
echo "tmux sessions:"
tmux ls
echo
echo "Monitor:"
echo "  tmux attach -t pg_s${SEEDS[0]}                     # detach: Ctrl-b d"
echo "  tail -f ${REPO}/logs/0427_seed*_${TS}.log         # tee'd stdout"
echo "  tail -f ${REPO}/logs/0427_seed*_${TS}.txt         # script's structured log"
echo "  watch -n 5 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'"
echo
KILL_LIST=$(printf "pg_s%s " "${SEEDS[@]}")
echo "Kill all:"
echo "  for s in ${KILL_LIST}; do tmux send-keys -t \$s C-c 2>/dev/null; done; sleep 2"
echo "  for s in ${KILL_LIST}; do tmux kill-session -t \$s 2>/dev/null; done"
echo
echo "Aggregate results when done:"
echo "  grep -h 'final_int' ${REPO}/logs/0427_seed*_${TS}.log"
echo "  grep -h 'val_bpb'   ${REPO}/logs/0427_seed*_${TS}.txt | tail -20"
