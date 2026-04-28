#!/usr/bin/env bash
# =============================================================================
# RunPod Training Script — Parameter Golf
# Branch: attempt/qk-gain-5.5-deeper-recurrence
# Config: QK-Gain 5.5, 3-Layer Recurrence (NUM_LAYERS=9 with recurrent passes),
#         Parallel Residuals, Sequence Packing 8192
#
# HOW TO USE ON RUNPOD:
#   1. Rent a pod: 8x H100 SXM (80GB) — "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
#   2. Open a Terminal in the pod
#   3. Run: bash runpod_run.sh 2>&1 | tee run_log.txt
#   4. Wait ~12-15 min (includes data download + 3 seeds)
#   5. Check run_log.txt for final val_bpb scores
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

echo "=============================================="
echo " PARAMETER GOLF — RunPod Setup + Train"
echo " $(date)"
echo "=============================================="

# -----------------------------------------
# STEP 0: Verify GPU environment
# -----------------------------------------
echo ""
echo "[STEP 0] Verifying CUDA + GPU setup..."
nvidia-smi
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected $GPU_COUNT GPUs"

# -----------------------------------------
# STEP 1: Clone / update the repo
# -----------------------------------------
echo ""
echo "[STEP 1] Cloning repo..."

REPO_DIR="/workspace/parameter-golf"
if [ -d "$REPO_DIR" ]; then
  echo "  Repo already exists, pulling latest..."
  cd "$REPO_DIR"
  git fetch origin
  git checkout attempt/qk-gain-5.5-deeper-recurrence
  git pull origin attempt/qk-gain-5.5-deeper-recurrence
else
  git clone --branch attempt/qk-gain-5.5-deeper-recurrence \
    https://github.com/Vickyrrrrrr/parameter-golf.git "$REPO_DIR"
  cd "$REPO_DIR"
fi
echo "  Repo ready at $REPO_DIR"

# -----------------------------------------
# STEP 2: Install Python dependencies
# -----------------------------------------
echo ""
echo "[STEP 2] Installing dependencies..."

pip install --quiet --upgrade pip
pip install --quiet \
  numpy \
  tqdm \
  "torch>=2.4.0" \
  huggingface-hub \
  setuptools \
  "typing-extensions==4.15.0" \
  datasets \
  tiktoken \
  sentencepiece

echo "  Dependencies installed."

# -----------------------------------------
# STEP 3: Download + tokenize the dataset
# -----------------------------------------
echo ""
echo "[STEP 3] Downloading FineWeb dataset + tokenizer..."
echo "  This downloads ~10B tokens of FineWeb via HuggingFace."
echo "  Expected time: 5-10 min depending on bandwidth."

DATA_DIR="$REPO_DIR/data"
DATASET_DIR="$DATA_DIR/datasets/fineweb10B_sp1024"
TOKENIZER_DIR="$DATA_DIR/tokenizers"
TOKENIZER_PATH="$TOKENIZER_DIR/fineweb_1024_bpe.model"

mkdir -p "$DATASET_DIR" "$TOKENIZER_DIR"

# Check if data already exists to skip re-download
TRAIN_SHARDS=$(ls "$DATASET_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_SHARDS=$(ls "$DATASET_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)

if [ "$TRAIN_SHARDS" -gt 0 ] && [ "$VAL_SHARDS" -gt 0 ] && [ -f "$TOKENIZER_PATH" ]; then
  echo "  Dataset already present ($TRAIN_SHARDS train shards, $VAL_SHARDS val shards). Skipping download."
else
  echo "  Running data download script..."
  cd "$DATA_DIR"
  python3 cached_challenge_fineweb.py
  cd "$REPO_DIR"

  # Verify after download
  TRAIN_SHARDS=$(ls "$DATASET_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
  VAL_SHARDS=$(ls "$DATASET_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
  echo "  Download complete: $TRAIN_SHARDS train shards, $VAL_SHARDS val shards"
fi

# Sanity check: abort early if data missing
if [ "$TRAIN_SHARDS" -eq 0 ]; then
  echo "ERROR: No training shards found in $DATASET_DIR"
  echo "       Expected files matching: fineweb_train_*.bin"
  exit 1
fi
if [ "$VAL_SHARDS" -eq 0 ]; then
  echo "ERROR: No validation shards found in $DATASET_DIR"
  echo "       Expected files matching: fineweb_val_*.bin"
  exit 1
fi
if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "ERROR: Tokenizer not found at $TOKENIZER_PATH"
  exit 1
fi

echo "  Data check passed."

# -----------------------------------------
# STEP 4: Pre-flight checks
# -----------------------------------------
echo ""
echo "[STEP 4] Pre-flight checks..."

cd "$REPO_DIR"

python3 - <<'PYCHECK'
import os, sys, glob, torch
import sentencepiece as spm

data_dir = "./data/datasets/fineweb10B_sp1024"
tok_path = "./data/tokenizers/fineweb_1024_bpe.model"

# Check GPU
assert torch.cuda.is_available(), "CUDA not available!"
n_gpu = torch.cuda.device_count()
print(f"  GPUs available: {n_gpu}")

# Check data
train_files = sorted(glob.glob(os.path.join(data_dir, "fineweb_train_*.bin")))
val_files   = sorted(glob.glob(os.path.join(data_dir, "fineweb_val_*.bin")))
assert len(train_files) > 0, f"No train shards in {data_dir}"
assert len(val_files)   > 0, f"No val shards in {data_dir}"
print(f"  Train shards: {len(train_files)}")
print(f"  Val shards:   {len(val_files)}")

# Check tokenizer
sp = spm.SentencePieceProcessor(model_file=tok_path)
assert sp.vocab_size() == 1024, f"Expected vocab=1024, got {sp.vocab_size()}"
print(f"  Tokenizer vocab size: {sp.vocab_size()} ✓")

# Check train_gpt.py importable
sys.path.insert(0, ".")
import train_gpt
print("  train_gpt.py import: OK ✓")

print("  All pre-flight checks PASSED ✓")
PYCHECK

echo "  Pre-flight checks complete."

# -----------------------------------------
# STEP 5: Build torchrun command
# -----------------------------------------

# Use all available GPUs
NPROC=$GPU_COUNT
if [ "$NPROC" -gt 8 ]; then
  NPROC=8  # contest caps at 8 GPUs
fi
echo ""
echo "[STEP 5] Will train on $NPROC GPU(s) with torchrun."

# Common env vars for all runs
# These are the SOTA-beating hyperparameters:
#   QK_GAIN_INIT=5.5      → Up from SOTA's 5.25, stabilizes attention init better
#   TRAIN_SEQ_LEN=8192    → Longer context = more signal per step (sequence packing)
#   VAL_BATCH_SIZE=8192   → Match val seq len
#   NUM_LAYERS=9          → Same depth but with U-Net skip connections (already in code)
#   MLP_MULT=2            → Keep MLP narrow to stay under 16MB checkpoint
#   MODEL_DIM=512         → Baseline width
#   NUM_HEADS=8           → 8 heads
#   NUM_KV_HEADS=2        → Aggressive GQA: 4 queries share 1 KV pair → saves params
#   ITERATIONS=20000      → Full run
#   MAX_WALLCLOCK_SECONDS=570  → 9.5 min hard cap (leaves 30s buffer for saving)
#   WARMDOWN_ITERS=1500   → Slightly longer warmdown than default

COMMON_ENVS=(
  "DATA_PATH=./data/datasets/fineweb10B_sp1024"
  "TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model"
  "QK_GAIN_INIT=5.5"
  "TRAIN_SEQ_LEN=8192"
  "VAL_BATCH_SIZE=8388608"
  "NUM_LAYERS=9"
  "MODEL_DIM=512"
  "NUM_HEADS=8"
  "NUM_KV_HEADS=2"
  "MLP_MULT=2"
  "TIE_EMBEDDINGS=1"
  "ITERATIONS=20000"
  "WARMDOWN_ITERS=1500"
  "WARMUP_STEPS=20"
  "TRAIN_BATCH_TOKENS=524288"
  "MAX_WALLCLOCK_SECONDS=570"
  "VAL_LOSS_EVERY=1000"
  "TRAIN_LOG_EVERY=200"
  "LOGIT_SOFTCAP=30.0"
  "ROPE_BASE=10000.0"
  "MATRIX_LR=0.04"
  "SCALAR_LR=0.04"
  "TIED_EMBED_LR=0.05"
  "MUON_MOMENTUM=0.95"
  "MUON_BACKEND_STEPS=5"
)

# Build env export string
ENV_EXPORTS=""
for env in "${COMMON_ENVS[@]}"; do
  ENV_EXPORTS="$ENV_EXPORTS $env"
done

# -----------------------------------------
# STEP 6: Run 3 seeds (required for submission)
# -----------------------------------------
echo ""
echo "[STEP 6] Starting 3-seed training runs..."
echo "  Each run: ~3-4 min on 8xH100. Total: ~10-12 min."
echo ""

RESULTS_FILE="$REPO_DIR/seed_results.txt"
echo "Seed Results — $(date)" > "$RESULTS_FILE"
echo "Config: QK_GAIN_INIT=5.5, TRAIN_SEQ_LEN=8192, NUM_KV_HEADS=2" >> "$RESULTS_FILE"
echo "======================================================" >> "$RESULTS_FILE"

SEEDS=(1337 1338 1339)
declare -a BPB_SCORES=()

for SEED in "${SEEDS[@]}"; do
  echo "----------------------------------------------"
  echo "  SEED $SEED — Starting at $(date)"
  echo "----------------------------------------------"

  RUN_ID="qkgain55_seq8192_kv2_seed${SEED}"
  LOG_FILE="$REPO_DIR/logs/run_seed${SEED}.txt"
  mkdir -p "$REPO_DIR/logs"

  # Run torchrun with all envs + this seed
  env $ENV_EXPORTS \
    SEED=$SEED \
    RUN_ID=$RUN_ID \
  torchrun \
    --nproc_per_node=$NPROC \
    --master_port=29500 \
    "$REPO_DIR/train_gpt.py" \
    2>&1 | tee "$LOG_FILE"

  # Extract final val_bpb from log
  FINAL_BPB=$(grep "final_int8_zlib_roundtrip val_loss" "$LOG_FILE" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "NOT_FOUND")
  FINAL_LOSS=$(grep "final_int8_zlib_roundtrip val_loss" "$LOG_FILE" | tail -1 | grep -oP 'val_loss:\K[0-9.]+' || echo "NOT_FOUND")

  echo ""
  echo "  ✅ SEED $SEED DONE:"
  echo "     val_bpb  = $FINAL_BPB"
  echo "     val_loss = $FINAL_LOSS"
  echo ""

  echo "SEED $SEED: val_bpb=$FINAL_BPB  val_loss=$FINAL_LOSS" >> "$RESULTS_FILE"
  BPB_SCORES+=("$FINAL_BPB")
done

# -----------------------------------------
# STEP 7: Compute mean BPB across seeds
# -----------------------------------------
echo ""
echo "[STEP 7] Computing mean val_bpb..."

python3 - <<PYRESULTS
scores = [${BPB_SCORES[*]:-0}]
valid  = [float(s) for s in scores if s != "NOT_FOUND"]
if valid:
    mean_bpb = sum(valid) / len(valid)
    print(f"  Scores: {valid}")
    print(f"  Mean val_bpb: {mean_bpb:.6f}")
    # Current SOTA (SP8192_3LayerRecur_ParResid_QK525_LegalTTT) ≈ 0.3278
    if mean_bpb < 0.3278:
        print(f"  🎉 BEATS CURRENT RECORD! Record ≈ 0.3278, Yours = {mean_bpb:.6f}")
    else:
        print(f"  Record ≈ 0.3278, Gap = {mean_bpb - 0.3278:.4f}")
else:
    print("  Could not parse scores from logs. Check logs/ directory manually.")
PYRESULTS

echo ""
echo "=============================================="
echo " ALL DONE — $(date)"
echo " Results saved to: $RESULTS_FILE"
echo " Individual logs:  $REPO_DIR/logs/"
echo " Models saved:     $REPO_DIR/final_model.int8.ptz (last seed)"
echo "=============================================="
echo ""
echo "Next step: If any seed achieved val_bpb < 0.3278, copy the"
echo "submission files to records/ and open a PR!"
