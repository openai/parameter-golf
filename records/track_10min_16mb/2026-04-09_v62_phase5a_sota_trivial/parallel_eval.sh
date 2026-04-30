#!/usr/bin/env bash
# Parallel eval: run stride=64 SLOT=100 eval on up to 8 models at once, one per GPU.
# Usage: bash parallel_eval.sh <comma-separated variant names>
# Example: bash parallel_eval.sh p5a,p5a_bg4096,p5a_hm5,p5a_bg4096_hm5

SCRIPT=records/track_10min_16mb/2026-04-09_v62_phase5a_sota_trivial/train_gpt.py
VARIANTS="${1:-p5a,p5a_bg4096,p5a_hm5,p5a_bg4096_hm5,p5a_bg8192,p5a_nl12}"

IFS=',' read -r -a names <<< "$VARIANTS"
gpu=0
pids=()
for name in "${names[@]}"; do
  RUN_NAME="v62_${name}_s1337"
  CKPT="runs/${RUN_NAME}/model.rans.ptz"
  LOGDIR="logs/v62_${name}_s1337"
  mkdir -p "$LOGDIR"
  if [[ ! -f "$CKPT" ]]; then
    echo "[$name] ckpt missing: $CKPT, skipping"
    continue
  fi

  # Phase 4 env: re-materialize the model architecture with right bigram/hidden/etc.
  extra_env=""
  case "$name" in
    *bg4096_hm5) extra_env="BIGRAM_VOCAB=4096 HIDDEN_MULT=5.0";;
    *bg4096)     extra_env="BIGRAM_VOCAB=4096";;
    *hm5)        extra_env="HIDDEN_MULT=5.0";;
    *bg8192)     extra_env="BIGRAM_VOCAB=8192";;
    *nl12)       extra_env="NUM_LAYERS=12";;
    *ve4)        extra_env="VE_LAYERS=7,8,9,10";;
  esac

  echo "[$name] launching on GPU $gpu (env: $extra_env)"
  CUDA_VISIBLE_DEVICES=$gpu env EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1 \
    QK_GAIN_INIT=5.0 MUON_EQ_R=1 $extra_env \
    nohup python -u "$SCRIPT" --eval --checkpoint "$CKPT" \
      --stride 64 --batch-seqs 32 --seq-len 1024 --compile \
      --slot-lr 0.1 --slot-steps 100 --slot-lr-min 0.001 \
      --data-dir data/datasets/fineweb10B_sp1024 \
      --tokenizer data/tokenizers/fineweb_1024_bpe.model \
      > "${LOGDIR}/eval_par.log" 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done

echo "Launched ${#pids[@]} evals on GPUs 0..$((gpu-1))"
echo "PIDs: ${pids[@]}"
wait "${pids[@]}" 2>/dev/null
echo "ALL EVALS DONE"

# Summary
echo ""
echo "=== SUMMARY ==="
for name in "${names[@]}"; do
  LOGDIR="logs/v62_${name}_s1337"
  if [[ -f "${LOGDIR}/eval_par.log" ]]; then
    b=$(grep -oP 'val_bpb:\s*\K[0-9.]+' "${LOGDIR}/eval_par.log" 2>/dev/null | tail -1)
    printf "  %-20s  bpb=%s\n" "$name" "${b:-?}"
  fi
done
