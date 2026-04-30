#!/bin/bash
set -euo pipefail

# PR #1950 Long-Train Artifact Scaling Experiment
# NON-RECORD: trains > 600s, not record-track compliant

export SEED=${SEED:-42}
export GPTQ_RESERVE_SECONDS=5.5
export COMPRESSOR=pergroup
export EMBED_WD=0.06
export MATRIX_CLIP_SIGMAS=12.85
export ATTN_CLIP_SIGMAS=12.0
export MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7
export EMBED_CLIP_SIGMAS=12.0
export MATRIX_LR=0.026
export MIN_LR=0.1
export CASEOPS_ENABLED=1
export SMEAR_GATE_ENABLED=1
export GATE_WINDOW=12
export LQER_ENABLED=1
export LQER_RANK=4
export LQER_TOP_K=3
export LQER_FACTOR_BITS=4
export LQER_ASYM_ENABLED=1
export LQER_ASYM_GROUP=64
export PHASED_TTT_PREFIX_DOCS=2000
export PHASED_TTT_NUM_PHASES=3
export TTT_WARM_START_A=1
export SPARSE_ATTN_GATE_ENABLED=1
export FUSED_CE_ENABLED=1
export NCCL_NET=Socket

# Long-train specific
export LONGTRAIN_EXPORT_MINUTES="${LONGTRAIN_EXPORT_MINUTES:-10,20,30,45,60}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}"
export NON_RECORD_LONGTRAIN=1
export EXPORT_MODE="${EXPORT_MODE:-light}"

# Data paths (set externally for RunPod)
export DATA_PATH="${DATA_PATH:-/root/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/root/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-/root/rehearsal_out/seed${SEED}}"

mkdir -p "${ARTIFACT_DIR}"

echo "=== PR #1950 Long-Train Artifact Scaling Experiment ==="
echo "NON-RECORD: Training for ${MAX_WALLCLOCK_SECONDS}s ($(echo "${MAX_WALLCLOCK_SECONDS}/60" | bc)min)"
echo "Checkpoints at: ${LONGTRAIN_EXPORT_MINUTES} minutes"
echo "Export mode: ${EXPORT_MODE}"
echo "Seed: ${SEED}"
echo "Start: $(date -u)"

# Print all env vars for reproducibility
env | sort | grep -E '^(SEED|GPTQ|COMPRESSOR|EMBED|MATRIX|ATTN|MLP|CASEOPS|SMEAR|GATE|LQER|PHASED|TTT|SPARSE|FUSED|NCCL|LONGTRAIN|MAX_WALL|NON_RECORD|EXPORT|DATA|TOKENIZER|ARTIFACT)' || true

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "${ARTIFACT_DIR}/train_seed${SEED}_longtrain.log"

echo "=== Training complete: $(date -u) ==="
