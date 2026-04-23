#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <034cA|034cB|034cC> [seed]" >&2
  exit 2
fi

rung="$1"
seed="${2:-314}"

case "$rung" in
  034cA) min_lr="0.05" ;;
  034cB) min_lr="0.10" ;;
  034cC) min_lr="0.15" ;;
  *)
    echo "unknown rung: $rung" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
record_dir="$repo_root/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT"
artifact_dir="/workspace/runs/034c-min-lr-on-034/${rung}/seed_${seed}"
baseline_log="/workspace/runs/034-frozen-direct-carry-from-031a/seed_314/train.log"

mkdir -p /workspace/.torch_inductor_cache "$artifact_dir"

export TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache
export DATA_DIR=/workspace/parameter-golf/data
export ARTIFACT_DIR="$artifact_dir"
export RUN_ID="${rung}_seed${seed}_corrected"
export SEED="$seed"
export MIN_LR="$min_lr"

# Inherit the validated 034 runtime stack exactly; MIN_LR is the only allowed diff.
export CASEOPS_ENABLED=1
export GATED_ATTN_ENABLED=1
export GATED_ATTN_INIT_STD=0.005
export GATED_ATTN_QUANT_GATE=1
export EMBED_BITS=7
export EMBED_CLIP_SIGMAS=15.0
export MLP_CLIP_SIGMAS=12.0
export TRAIN_LOG_EVERY=100
export DIRECT_CARRY_MODE=frozen_edge_self
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export TTT_ENABLED=1
export PHASED_TTT_PREFIX_DOCS=2000
export PHASED_TTT_NUM_PHASES=1
export MAX_WALLCLOCK_SECONDS=1200

cd "$record_dir"
python3 "$repo_root/research/scripts/verify_034c_inheritance.py" \
  --baseline-log "$baseline_log" \
  --artifact-dir "$artifact_dir" \
  --expected-min-lr "$min_lr" \
  --label "$rung"
exec torchrun --standalone --nproc_per_node=4 train_gpt.py
