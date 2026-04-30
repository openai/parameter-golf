#!/usr/bin/env bash
# Reproduce and audit PR #1812 for a Field Guide aligned non-record package.
#
# Modes:
#   audit      Download exact PR #1812 files and run static Issue #1017 audit.
#   seed42     Run one independent seed-42 reproduction.
#   two-seed   Run seed 42 and seed 314 for an audit package.
#
# Usage on an 8xH100 RunPod pod from a parameter-golf checkout:
#   bash scripts/runpod_pr1812_audit_repro.sh audit
#   bash scripts/runpod_pr1812_audit_repro.sh seed42
#
# Data assumptions:
#   data/datasets/fineweb10B_sp8192 and data/tokenizers/fineweb_8192_bpe.model
#   already exist in this checkout. Override DATA_DIR if staged elsewhere.

set -euo pipefail

MODE="${1:-audit}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${ROOT}/runs/pr1812_audit_repro"
PR_DIR="${WORK_DIR}/upstream_pr1812"
DATA_DIR="${DATA_DIR:-${ROOT}/data}"

TRAIN_URL="${TRAIN_URL:-https://github.com/openai/parameter-golf/raw/1350423f2b26d20b3c384f194e8f66d06a6428c2/records%2Ftrack_10min_16mb%2F2026-04-25_SP8192_3LayerRecur_LegalTTT_4ep%2Ftrain_gpt.py}"
README_URL="${README_URL:-https://github.com/openai/parameter-golf/raw/1350423f2b26d20b3c384f194e8f66d06a6428c2/records%2Ftrack_10min_16mb%2F2026-04-25_SP8192_3LayerRecur_LegalTTT_4ep%2FREADME.md}"
SUBMISSION_URL="${SUBMISSION_URL:-https://github.com/openai/parameter-golf/raw/1350423f2b26d20b3c384f194e8f66d06a6428c2/records%2Ftrack_10min_16mb%2F2026-04-25_SP8192_3LayerRecur_LegalTTT_4ep%2Fsubmission.json}"

mkdir -p "${PR_DIR}"

fetch_pr_files() {
    if [ ! -f "${PR_DIR}/train_gpt.py" ]; then
        echo "[setup] downloading PR #1812 train_gpt.py"
        curl -fsSL -o "${PR_DIR}/train_gpt.py" "${TRAIN_URL}"
    fi
    if [ ! -f "${PR_DIR}/README.md" ]; then
        curl -fsSL -o "${PR_DIR}/README.md" "${README_URL}"
    fi
    if [ ! -f "${PR_DIR}/submission.json" ]; then
        curl -fsSL -o "${PR_DIR}/submission.json" "${SUBMISSION_URL}"
    fi
}

run_audit() {
    fetch_pr_files
    python3 "${ROOT}/scripts/pgolf_field_guide_audit.py" \
        "${PR_DIR}/train_gpt.py" \
        --write-decoded "${PR_DIR}/train_gpt.decoded.py" \
        --output "${WORK_DIR}/field_guide_static_audit.json"
}

check_data() {
    local sp_dir="${DATA_DIR}/datasets/fineweb10B_sp8192"
    local tok="${DATA_DIR}/tokenizers/fineweb_8192_bpe.model"
    if [ ! -d "${sp_dir}" ]; then
        echo "FATAL: missing ${sp_dir}" >&2
        echo "Download first: MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192" >&2
        exit 1
    fi
    if [ ! -f "${tok}" ]; then
        echo "FATAL: missing ${tok}" >&2
        exit 1
    fi
}

run_seed() {
    local seed="$1"
    local out_dir="${WORK_DIR}/seed${seed}"

    fetch_pr_files
    check_data
    mkdir -p "${out_dir}"

    echo ""
    echo "=== PR #1812 reproduction: seed=${seed} ==="
    echo "Output: ${out_dir}"

    (
        cd "${PR_DIR}"
        export DATA_DIR
        export SEED="${seed}"
        export TTT_ENABLED=1
        export TTT_LR=0.005
        export TTT_EPOCHS=4
        export RUN_ID="pr1812_audit_s${seed}"
        export ARTIFACT_DIR="${out_dir}"
        export PYTHONUNBUFFERED=1
        torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "${out_dir}/console.log"
    )

    grep -E "val_bpb|eval_time|Serialized model|Total submission size|train_time|stopping_early" \
        "${out_dir}/console.log" | tail -40 || true
}

case "${MODE}" in
    audit)
        run_audit
        ;;
    seed42)
        run_audit
        run_seed 42
        ;;
    two-seed)
        run_audit
        run_seed 42
        run_seed 314
        ;;
    *)
        echo "Usage: $0 [audit|seed42|two-seed]" >&2
        exit 2
        ;;
esac
