#!/usr/bin/env bash
# =============================================================================
# RunPod Launch Script — Parameter Golf Auto-Evolve
#
# Run this ONCE on the RunPod machine after SSH-ing in.
# It sets up the workspace, downloads data, and starts the evolve loop
# inside a tmux session so it survives SSH disconnects.
#
# Usage:
#   bash autoevolve/runpod_launch.sh [--nproc 1] [--model gpt-5.4] [--max-iters 0] [--dry-run]
#
# Then detach: Ctrl+B then D
# Re-attach:   tmux attach -t evolve
# =============================================================================
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
NPROC=1
MODEL="gpt-5.4"
MAX_ITERS=0
DRY_RUN=0
SESSION="evolve"
WORKSPACE="/workspace"
REPO="$WORKSPACE/parameter-golf"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc)     NPROC="$2";     shift 2 ;;
    --model)     MODEL="$2";     shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1;      shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo ""
echo "==========================================================="
echo "  Parameter Golf Auto-Evolve — RunPod Setup"
if [ "$NPROC" -eq 1 ]; then
  RUN_MODE="scout"
else
  RUN_MODE="full"
fi
if [ "$MAX_ITERS" -eq 0 ]; then
  MAX_ITERS_LABEL="until manually stopped"
else
  MAX_ITERS_LABEL="$MAX_ITERS"
fi
echo "  GPUs: $NPROC  |  Mode: $RUN_MODE  |  Model: $MODEL  |  Max iters: $MAX_ITERS_LABEL  |  Dry run: $DRY_RUN"
echo "==========================================================="

# ── 1. Check repo ─────────────────────────────────────────────────────────────
if [ ! -d "$REPO" ]; then
  echo ""
  echo "ERROR: Repo not found at $REPO"
  echo "Please clone first:"
  echo "  cd /workspace && git clone https://github.com/zanwenfu/parameter-golf.git"
  exit 1
fi
cd "$REPO"
echo ""
echo "[1/6] Repo found at $REPO"
git log --oneline -3

# ── 2. Check .env ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Checking .env ..."
ENV_FILE="$REPO/autoevolve/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "  .env not found. Creating template..."
  cat > "$ENV_FILE" <<'EOF'
OPENAI_API_KEY=sk-YOUR_OPENAI_KEY_HERE
GITHUB_TOKEN=ghp_YOUR_GITHUB_PAT_HERE
EOF
  echo ""
  echo "  ⚠  Edit $ENV_FILE with your keys, then re-run:"
  echo "     nano $ENV_FILE"
  echo ""
  echo "  OPENAI_API_KEY : your OpenAI API key (sk-proj-...)"
  echo "  GITHUB_TOKEN   : GitHub Personal Access Token with 'repo' scope"
  echo "                   Create at: https://github.com/settings/tokens/new"
  exit 1
fi
if grep -q "YOUR_OPENAI_KEY_HERE" "$ENV_FILE"; then
  echo "  ⚠  .env still has placeholder keys. Edit it:"
  echo "     nano $ENV_FILE"
  exit 1
fi
if ! grep -q "GITHUB_TOKEN" "$ENV_FILE" || grep -q "YOUR_GITHUB_PAT_HERE" "$ENV_FILE"; then
  echo "  ⚠  GITHUB_TOKEN missing or not set in .env"
  echo "     Results won't be pushed to GitHub — pod data will be lost on shutdown!"
  echo "     Add: GITHUB_TOKEN=ghp_... to $ENV_FILE"
  echo "     Create token at: https://github.com/settings/tokens/new  (select 'repo' scope)"
  echo ""
  read -p "  Continue without GitHub push? [y/N] " yn
  if [[ "$yn" != "y" && "$yn" != "Y" ]]; then exit 1; fi
fi
echo "  .env OK"

# ── Configure git identity (required for commits on RunPod) ───────────────────
git config user.email "autoevolve@parameter-golf" 2>/dev/null || true
git config user.name  "AutoEvolve Bot"             2>/dev/null || true

# ── 3. Install Python deps ─────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Python dependencies ..."
pip install --quiet openai python-dotenv zstandard 2>&1 | tail -3
echo "  Deps OK"

# ── 4. Download data ───────────────────────────────────────────────────────────
echo ""
echo "[4/6] Checking training data ..."
DATA_DIR="$REPO/data/datasets/fineweb10B_sp1024"
TOKENIZER="$REPO/data/tokenizers/fineweb_1024_bpe.model"

if [ ! -f "$TOKENIZER" ] || [ $(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l) -lt 2 ]; then
  echo "  Downloading FineWeb data (this takes ~5-10 minutes)..."
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
  echo "  Data downloaded."
else
  echo "  Data already present ($(ls "$DATA_DIR"/fineweb_train_*.bin | wc -l) shards)"
fi

# ── 5. Create autoevolve/logs dir (not gitignored) ────────────────────────────
mkdir -p "$REPO/autoevolve/logs"

# ── 6. Launch in tmux ──────────────────────────────────────────────────────────
echo ""
echo "[6/6] Launching auto-evolve in tmux session '$SESSION' ..."

# Kill existing session if it's stuck
tmux kill-session -t "$SESSION" 2>/dev/null || true

ITER_FLAG="--max-iters $MAX_ITERS"

DRY_FLAG=""
if [ "$DRY_RUN" -eq 1 ]; then
  DRY_FLAG="--dry-run"
fi

# Build the command
CMD="cd $REPO && python3 autoevolve/evolve.py --nproc $NPROC --model $MODEL $ITER_FLAG $DRY_FLAG 2>&1 | tee autoevolve/logs/evolve_main.log"

tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "$CMD" Enter

echo ""
echo "==========================================================="
echo "  Auto-evolve is running! Session: '$SESSION'"
echo ""
echo "  Monitor commands:"
echo "    tmux attach -t $SESSION          # watch live output"
echo "    python3 autoevolve/monitor.py    # full dashboard"
echo "    python3 autoevolve/monitor.py --summary   # one-liner"
echo "    python3 autoevolve/monitor.py --tail      # tail latest experiment"
echo "    python3 autoevolve/monitor.py --watch 60  # auto-refresh every 60s"
echo ""
echo "  Detach from tmux: Ctrl+B then D"
echo "  Stop gracefully:  Ctrl+C inside tmux session"
echo ""
echo "  Budget reminder: 1xH100 ≈ \$2.49/hr → \$25 ≈ 10 hrs ≈ 25 experiments"
echo "==========================================================="
