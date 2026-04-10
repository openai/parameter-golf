#!/usr/bin/env bash
# Parallel fast eval: stride=64 SLOT=50 (half the SLOT cost, ±0.001 noise)
# Runs 4 evals in parallel. Sequential batches for 7 variants → 2 rounds.
# Each round ~30 min (instead of 50 min for SLOT=100). 2 rounds = ~60 min.

SCRIPT=records/track_10min_16mb/2026-04-09_v62_phase5a_sota_trivial/train_gpt.py

run_batch() {
  local gpu_base="$1"; shift
  local names=("$@")
  pids=()
  gpu=$gpu_base
  for name in "${names[@]}"; do
    RUN_NAME="v62_${name}_s1337"
    CKPT="runs/${RUN_NAME}/model.rans.ptz"
    LOGDIR="logs/v62_${name}_s1337"
    mkdir -p "$LOGDIR"
    if [[ ! -f "$CKPT" ]]; then
      echo "[$name] ckpt missing, skip"
      continue
    fi
    extra_env=""
    case "$name" in
      *bg4096_hm5) extra_env="BIGRAM_VOCAB=4096 HIDDEN_MULT=5.0";;
      *bg4096)     extra_env="BIGRAM_VOCAB=4096";;
      *hm5)        extra_env="HIDDEN_MULT=5.0";;
      *bg8192)     extra_env="BIGRAM_VOCAB=8192";;
      *nl12)       extra_env="NUM_LAYERS=12";;
      *ve4)        extra_env="VE_LAYERS=7,8,9,10";;
    esac
    echo "[$name] GPU $gpu ($extra_env) SLOT=50"
    CUDA_VISIBLE_DEVICES=$gpu env EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1 \
      QK_GAIN_INIT=5.0 MUON_EQ_R=1 $extra_env \
      nohup python -u "$SCRIPT" --eval --checkpoint "$CKPT" \
        --stride 64 --batch-seqs 32 --seq-len 1024 --compile \
        --slot-lr 0.1 --slot-steps 50 --slot-lr-min 0.001 \
        --data-dir data/datasets/fineweb10B_sp1024 \
        --tokenizer data/tokenizers/fineweb_1024_bpe.model \
        > "${LOGDIR}/eval_fast.log" 2>&1 &
    pids+=($!)
    gpu=$((gpu + 1))
  done
  echo "Round PIDs: ${pids[@]}"
  wait "${pids[@]}" 2>/dev/null
  echo "Round done"
}

# Round 1: 4 variants on GPUs 0-3
run_batch 0 p5a p5a_bg4096 p5a_hm5 p5a_bg4096_hm5
# Round 2: remaining 3 variants on GPUs 0-2
run_batch 0 p5a_bg8192 p5a_nl12 p5a_ve4

echo "ALL EVALS DONE"
echo ""
echo "=== SUMMARY ==="
for name in p5a p5a_bg4096 p5a_hm5 p5a_bg4096_hm5 p5a_bg8192 p5a_nl12 p5a_ve4; do
  LOGDIR="logs/v62_${name}_s1337"
  if [[ -f "${LOGDIR}/eval_fast.log" ]]; then
    b=$(grep -oP 'val_bpb:\s*\K[0-9.]+' "${LOGDIR}/eval_fast.log" 2>/dev/null | tail -1)
    printf "  %-20s  bpb=%s\n" "$name" "${b:-?}"
  fi
done
