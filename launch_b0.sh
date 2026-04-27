#!/usr/bin/env bash
# launch_b0.sh — Sprint 002 launcher
#
# Defaults to B0 baseline replication. Accepts any ablation row via --row + --config,
# so the same script handles A1-A6 and C1-C2 once B0 lands.
#
# Bundles:
#   1. Idempotent data-prep check (downloads + tokenizes only if shards missing).
#   2. Env hygiene (unsets all known toggles, then re-exports only --config ones).
#   3. torchrun launch into logs/{RUN_ID}.txt.
#   4. record_run.py invocation on success.
# Does NOT auto-commit or push — print the suggested git commands and let you review.
#
# Usage:
#   ./launch_b0.sh                                                    # B0 seed 1337, 1 GPU
#   ./launch_b0.sh --seed 4242                                        # B0 different seed
#   NPROC=8 ./launch_b0.sh --seed 1337                                # B0 on 8xH100 SXM
#   ./launch_b0.sh --row A1 --seed 1337 --config "QUANTIZE_WEIGHTS=none"
#   ./launch_b0.sh --row C1 --seed 1337 --config "QUANTIZE_WEIGHTS=none NUM_KV_HEADS=8"
#   ./launch_b0.sh --dry-run                                          # show config, no spend
#
# Exit codes:
#   0   training + record succeeded
#   2   bad CLI args
#   3   pre-flight check failed (wrong dir, missing deps, etc.)
#   4   training crashed (record_run.py NOT invoked)
#   5   record_run.py failed

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ROW="B0"
SEED="1337"
CONFIG=""
NPROC="${NPROC:-1}"
DRY_RUN=0

# All ablation toggles — must mirror KNOWN_TOGGLES in record_run.py.
KNOWN_TOGGLES="QUANTIZE_WEIGHTS QUANT_SCHEME SDPA_BACKEND OPTIMIZER NUM_KV_HEADS TIE_EMBEDDINGS"
VALID_ROWS="B0 A1 A2 A3 A4 A5 A6 C1 C2"

DATA_DIR="data/datasets/fineweb10B_sp1024"
TOKENIZER="data/tokenizers/fineweb_1024_bpe.model"
VAL_SHARD="${DATA_DIR}/fineweb_val_000000.bin"
TRAIN_SHARD="${DATA_DIR}/fineweb_train_000000.bin"

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --row)      ROW="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --config)   CONFIG="$2"; shift 2 ;;
        --nproc)    NPROC="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage 0 ;;
        *) echo "Unknown arg: $1" >&2; usage 2 ;;
    esac
done

# Validate --row.
case " ${VALID_ROWS} " in
    *" ${ROW} "*) ;;
    *) echo "FATAL: --row=${ROW} not in {${VALID_ROWS}}" >&2; exit 2 ;;
esac

# Validate --nproc divides 8 (train_gpt.py constraint).
case "$NPROC" in
    1|2|4|8) ;;
    *) echo "FATAL: --nproc=${NPROC} must divide 8 (one of 1, 2, 4, 8)" >&2; exit 2 ;;
esac

# Validate --config keys are in KNOWN_TOGGLES (string-only check; export happens later).
if [[ -n "$CONFIG" ]]; then
    for kv in $CONFIG; do
        if [[ "$kv" != *"="* ]]; then
            echo "FATAL: --config token '${kv}' missing '=' (expected KEY=VAL)" >&2
            exit 2
        fi
        key="${kv%%=*}"
        case " ${KNOWN_TOGGLES} " in
            *" ${key} "*) ;;
            *) echo "FATAL: --config key '${key}' not in known toggles {${KNOWN_TOGGLES}}" >&2; exit 2 ;;
        esac
    done
fi

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if [[ ! -f "train_gpt.py" || ! -f "record_run.py" ]]; then
    echo "FATAL: must run from repo root (train_gpt.py + record_run.py expected here)" >&2
    exit 3
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [[ "$CURRENT_BRANCH" != "sprint-002-ablation-study" ]]; then
    echo "WARN: current git branch is '${CURRENT_BRANCH}', not 'sprint-002-ablation-study'" >&2
    echo "      results will be tagged with this branch's commit; continue if intentional" >&2
fi

# Runtime-only checks: skipped on --dry-run since dry runs don't actually execute.
if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -x "$(command -v torchrun)" ]]; then
        echo "FATAL: torchrun not on PATH; pip install torch" >&2
        exit 3
    fi
    if ! python -c "import sentencepiece" 2>/dev/null; then
        echo "FATAL: sentencepiece not installed; pip install sentencepiece" >&2
        exit 3
    fi
fi

# ---------------------------------------------------------------------------
# Data prep (idempotent)
# ---------------------------------------------------------------------------
if [[ ! -f "$VAL_SHARD" || ! -f "$TRAIN_SHARD" || ! -f "$TOKENIZER" ]]; then
    echo "==> Data prep needed (shards or tokenizer missing). Running download_hf_docs_and_tokenize.py..."
    echo "    This is one-time per network volume; takes ~30-60min on a 1xH100 pod."

    # Disk-space safety: HF cache defaults to /root/.cache/huggingface (container
    # disk, often only ~50GB) but the docs blob from willdepueoai/parameter-golf
    # is ~48GB. If we don't redirect, hf_hub_download fails mid-fetch. Route the
    # whole HF state tree to the network volume parent of $DATA_DIR so it
    # persists across pod sessions and never hits the container disk.
    if [[ -z "${HF_HOME:-}" ]]; then
        export HF_HOME="$(cd "$(dirname "$DATA_DIR")/.." && pwd)/.hf_cache"
        mkdir -p "$HF_HOME"
        echo "    HF_HOME (auto-set): $HF_HOME"
    else
        echo "    HF_HOME (inherited): $HF_HOME"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "    [DRY RUN] Would run: python data/download_hf_docs_and_tokenize.py --output-root data"
    else
        if ! python -c "import huggingface_hub" 2>/dev/null; then
            pip install --quiet huggingface_hub
        fi
        # --output-root is REQUIRED by the upstream script. Layout it produces:
        #   data/tokenizers/fineweb_1024_bpe.model
        #   data/datasets/fineweb10B_sp1024/fineweb_{train,val}_*.bin
        # Defaults for --repo-id (willdepueoai/parameter-golf) and --num-val-docs
        # (sidecar-pinned, ~50k) are intentional — overriding either desyncs val_loss
        # from the upstream 0.9485 baseline.
        python data/download_hf_docs_and_tokenize.py --output-root data 2>&1 | tee data_prep.log
        if [[ ! -f "$VAL_SHARD" || ! -f "$TOKENIZER" ]]; then
            echo "FATAL: data prep finished but shards/tokenizer still missing — see data_prep.log" >&2
            exit 3
        fi
    fi
else
    echo "==> Data already present:"
    echo "    $TOKENIZER"
    ls -lh "$DATA_DIR"/*.bin | head -3
fi

# ---------------------------------------------------------------------------
# Env hygiene
# ---------------------------------------------------------------------------
# Wipe any leaked toggle from the shell rc / previous run.
for k in $KNOWN_TOGGLES; do
    unset "$k"
done

# Apply --config toggles (already validated above).
if [[ -n "$CONFIG" ]]; then
    for kv in $CONFIG; do
        export "$kv"
    done
fi

# Per-run identifiers.
ROW_LOWER="$(echo "$ROW" | tr '[:upper:]' '[:lower:]')"
export RUN_ID="${ROW_LOWER}_seed${SEED}"
export SEED

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo
echo "================================================================"
echo "Sprint 002 launch"
echo "  ROW:           $ROW"
echo "  SEED:          $SEED"
echo "  RUN_ID:        $RUN_ID"
echo "  NPROC:         $NPROC"
echo "  CONFIG:        ${CONFIG:-<all defaults — pure baseline>}"
echo "  GIT BRANCH:    $CURRENT_BRANCH"
echo "  GIT COMMIT:    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  LOG:           logs/${RUN_ID}.txt"
echo "================================================================"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] Would invoke:"
    echo "  torchrun --standalone --nproc-per-node=${NPROC} train_gpt.py"
    echo "  python record_run.py logs/${RUN_ID}.txt --row ${ROW} --seed ${SEED} --config \"${CONFIG}\""
    exit 0
fi

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
mkdir -p logs
TRAIN_START=$(date +%s)
set +e
torchrun --standalone --nproc-per-node="$NPROC" train_gpt.py 2>&1 | tee "logs/${RUN_ID}.txt"
TRAIN_RC=${PIPESTATUS[0]}
set -e
TRAIN_END=$(date +%s)
TRAIN_WALL=$((TRAIN_END - TRAIN_START))
echo
echo "==> Training wallclock: ${TRAIN_WALL}s, exit ${TRAIN_RC}"

if [[ "$TRAIN_RC" -ne 0 ]]; then
    echo "FATAL: torchrun exited ${TRAIN_RC}; record_run.py NOT invoked." >&2
    echo "       Inspect logs/${RUN_ID}.txt and fix locally before re-launching." >&2
    exit 4
fi

# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------
echo "==> Recording run to results/runs.jsonl ..."
if ! python record_run.py "logs/${RUN_ID}.txt" --row "$ROW" --seed "$SEED" --config "$CONFIG"; then
    echo "FATAL: record_run.py failed; row not written." >&2
    exit 5
fi

echo
echo "================================================================"
echo "B0/${ROW} seed ${SEED} done. Suggested next:"
echo "  cat results/runs.jsonl | tail -1 | python -m json.tool"
echo "  git add results/runs.jsonl"
echo "  git commit -m \"sprint 002: ${ROW} seed ${SEED} recorded\""
echo "  git push origin ${CURRENT_BRANCH}"
echo "  git push fork ${CURRENT_BRANCH}"
echo "================================================================"
