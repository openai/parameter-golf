#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

PYTHON=/home/frosty40/ml-lab/.venv-nightly/bin/python
export PYTHONPATH="${REPO_DIR}/local_shims:${PYTHONPATH:-}"

# ── Shared settings ──
COMMON=(
  MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=4
  CRAWLER_MLP_MULT=4
  TRIGRAM_VOCAB_SIZE=8192 TRIGRAM_DIM=128
  ROPE_DIMS=16 LN_SCALE=1
  TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
  TRAIN_BATCH_TOKENS=131072
  MAX_WALLCLOCK_SECONDS=1800
  WARMDOWN_ITERS=3500 WARMUP_STEPS=20
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
  MUON_MOMENTUM=0.99 MUON_WD=0.04
  VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100
  EVAL_STRIDE=64 SEED=1337
  VE_ENABLED=0 DTG_ENABLED=0
  TTT_BURST_ENABLED=0 DISTILL_ENABLED=0
  POLAR_ENABLED=0 TS_PD_ENABLED=0
  QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15
  SWA_ENABLED=1 SWA_EVERY=50
  DIAG_FAST_VAL=1 DIAG_FIXED_CADENCE=0
)

echo "═══════════════════════════════════════════════════════════════"
echo "H4: CRAWLER BANK AT U-NET BOTTLENECK — DGX Spark"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── ARM A: Control — 6 flat layers, no crawler ──
echo "────────────────────────────────────────────────────────────"
echo "[A] 6 flat, 0 crawler — CONTROL"
echo "    6 stored blocks, 6 effective depth"
echo "────────────────────────────────────────────────────────────"
RUN_ID_A="H4_A_6flat_$(date +%Y%m%d_%H%M%S)"
env "${COMMON[@]}" \
  RUN_ID="$RUN_ID_A" \
  NUM_FLAT_LAYERS=6 NUM_CRAWLER_LAYERS=0 CRAWLER_LOOPS=1 \
  XSA_LAST_N=0 \
  DIAG_CSV_PATH="experiments/H4_crawler_bank_on_unet/results/${RUN_ID_A}_diag.csv" \
  $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 \
    train_gpt_h4_bottleneck_crawler.py

echo ""
echo "[A] DONE: $RUN_ID_A"
echo ""

# ── ARM B: Test — 5 flat + 1 crawler x2 at bottleneck ──
echo "────────────────────────────────────────────────────────────"
echo "[B] 5 flat + 1 crawler x2 at BOTTLENECK"
echo "    6 stored blocks, 7 effective depth (1 free from sharing)"
echo "────────────────────────────────────────────────────────────"
RUN_ID_B="H4_B_5f1cx2_btn_$(date +%Y%m%d_%H%M%S)"
env "${COMMON[@]}" \
  RUN_ID="$RUN_ID_B" \
  NUM_FLAT_LAYERS=5 NUM_CRAWLER_LAYERS=1 CRAWLER_LOOPS=2 \
  XSA_LAST_N=1 \
  DIAG_CSV_PATH="experiments/H4_crawler_bank_on_unet/results/${RUN_ID_B}_diag.csv" \
  $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 \
    train_gpt_h4_bottleneck_crawler.py

echo ""
echo "[B] DONE: $RUN_ID_B"
echo ""

# ── ARM C: Test — 5 flat + 1 crawler x3 at bottleneck (more loops) ──
echo "────────────────────────────────────────────────────────────"
echo "[C] 5 flat + 1 crawler x3 at BOTTLENECK"
echo "    6 stored blocks, 8 effective depth (2 free from sharing)"
echo "────────────────────────────────────────────────────────────"
RUN_ID_C="H4_C_5f1cx3_btn_$(date +%Y%m%d_%H%M%S)"
env "${COMMON[@]}" \
  RUN_ID="$RUN_ID_C" \
  NUM_FLAT_LAYERS=5 NUM_CRAWLER_LAYERS=1 CRAWLER_LOOPS=3 \
  XSA_LAST_N=1 \
  DIAG_CSV_PATH="experiments/H4_crawler_bank_on_unet/results/${RUN_ID_C}_diag.csv" \
  $PYTHON -m torch.distributed.run --standalone --nproc_per_node=1 \
    train_gpt_h4_bottleneck_crawler.py

echo ""
echo "[C] DONE: $RUN_ID_C"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "H4 COMPLETE — check logs/ for detailed results"
echo "═══════════════════════════════════════════════════════════════"
