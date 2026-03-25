#!/usr/bin/env bash
# ── GolfStudent v2 — 8×H100 run script ──────────────────────────────────────
# Paste this into a fresh Lambda Labs / RunPod / Vast.ai 8×H100 instance.
# Cost: ~$5 per run (10-min wall-clock cap). Run 3× for p<0.01 significance.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RUN_ID="${1:-run1}"
LOG_FILE="logs_${RUN_ID}.txt"

echo "=== GolfStudent v2 | RUN_ID=$RUN_ID | $(date) ===" | tee "$LOG_FILE"

# ── 1. Install deps ───────────────────────────────────────────────────────────
pip install -q sentencepiece tiktoken tqdm numpy torch huggingface_hub datasets --upgrade

# ── 2. Clone the fork (your branch) ──────────────────────────────────────────
if [ ! -d "parameter-golf" ]; then
  git clone --branch feat/alan-samaha-golf --single-branch \
    https://github.com/whitestone1121-web/parameter-golf.git
fi
cd parameter-golf

# ── 3. Download training data (~2GB FineWeb sp1024 shards) ───────────────────
echo "--- Downloading data ---" | tee -a "../$LOG_FILE"
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# ── 4. Run training (10-min hard wall-clock cap enforced by script) ───────────
echo "--- Starting training ---" | tee -a "../$LOG_FILE"
RUN_ID="$RUN_ID" \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 \
    records/alan_samaha_golf/train_gpt.py 2>&1 | tee -a "../$LOG_FILE"

# ── 5. Extract final score ────────────────────────────────────────────────────
echo "--- Final results ---" | tee -a "../$LOG_FILE"
VAL_BPB=$(grep -oP 'val_bpb=\K[\d.]+' "../$LOG_FILE" | tail -1)
VAL_LOSS=$(grep -oP 'val_loss=\K[\d.]+' "../$LOG_FILE" | tail -1)
COMPRESSED=$(ls -l records/alan_samaha_golf/*.bin 2>/dev/null | awk '{print $5}' | tail -1)

echo "val_bpb     = $VAL_BPB"    | tee -a "../$LOG_FILE"
echo "val_loss    = $VAL_LOSS"   | tee -a "../$LOG_FILE"
echo "compressed  = $COMPRESSED bytes" | tee -a "../$LOG_FILE"

# ── 6. Auto-update submission.json ────────────────────────────────────────────
cat > records/alan_samaha_golf/submission.json << EOF
{
  "name": "Alan Samaha",
  "github_id": "whitestone1121-web",
  "val_bpb": $VAL_BPB,
  "val_loss": $VAL_LOSS,
  "compressed_bytes": $COMPRESSED,
  "description": "16.4M Hybrid LM: LinearRecurrence O(L) + Attention every 3rd layer, d=288, L=14, vocab=1024 weight-tied. Muon optimizer + EMA decay=0.999 + warmdown LR. GPTQ-lite per-row MSE INT8 quantization.",
  "model_params": 16400000,
  "architecture": "Hybrid LinearRecurrence + Attention",
  "d_model": 288,
  "num_layers": 14,
  "vocab_size": 1024,
  "weight_tied": true,
  "optimizer": "Muon (matrices) + Adam (embeddings/scalars)",
  "ema_decay": 0.999,
  "warmdown_iters": 1200,
  "run_id": "$RUN_ID",
  "hardware": "8x H100 SXM 80GB",
  "wall_clock_seconds": 600
}
EOF

echo "--- submission.json updated ---" | tee -a "../$LOG_FILE"
echo "--- Log saved to $LOG_FILE ---"
echo "Done. Copy $LOG_FILE back to your fork and attach to the PR."
