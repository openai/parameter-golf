#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-${REPO_ROOT}/.venv/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CACHE_DIR="${SCRIPT_DIR}/cache"

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(42 0 1234)
fi

ensure_dirs() {
  local out_dir="$1"
  mkdir -p \
    "${out_dir}" \
    "${CACHE_DIR}/tmp" \
    "${CACHE_DIR}/xdg" \
    "${CACHE_DIR}/torchinductor" \
    "${CACHE_DIR}/triton" \
    "${CACHE_DIR}/cuda"
}

run_train_seed() {
  local seed_value="$1"
  local run_id="shortchunk16_t2048_local0875_seed${seed_value}"
  local out_dir="${SCRIPT_DIR}/${run_id}"
  local run_log="${out_dir}/train.log"
  local force_train="${FORCE_TRAIN:-${FORCE:-0}}"

  ensure_dirs "${out_dir}"

  if [ "${force_train}" != "1" ] \
    && [ -f "${out_dir}/final_model.int6.ptz" ] \
    && { grep -q "Total submission size quantized+pergroup" "${run_log}" 2>/dev/null \
      || grep -q "Total submission size quantized+pergroup" "${out_dir}/${run_id}.txt" 2>/dev/null; } \
    && { grep -q "diagnostic quantized" "${run_log}" 2>/dev/null \
      || grep -q "diagnostic quantized" "${out_dir}/${run_id}.txt" 2>/dev/null; }; then
    echo "train cache hit for seed ${seed_value}: ${out_dir}"
    return 0
  fi

  (
    cd "${REPO_ROOT}"
    SEED="${seed_value}" \
    RUN_ID="${run_id}" \
    ARTIFACT_DIR="${out_dir}" \
    TMPDIR="${CACHE_DIR}/tmp" \
    XDG_CACHE_HOME="${CACHE_DIR}/xdg" \
    TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torchinductor" \
    TRITON_CACHE_DIR="${CACHE_DIR}/triton" \
    CUDA_CACHE_PATH="${CACHE_DIR}/cuda" \
    DATA_PATH=data/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=data/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    VOCAB_SIZE=8192 \
    NUM_LAYERS=11 \
    NUM_KV_HEADS=4 \
    MODEL_DIM=512 \
    NUM_HEADS=8 \
    MLP_MULT=4 \
    TRAIN_SEQ_LEN=2048 \
    EVAL_SEQ_LEN=2048 \
    TRAIN_BATCH_TOKENS=786432 \
    VAL_BATCH_TOKENS=524288 \
    VAL_LOSS_EVERY=0 \
    TRAIN_LOG_EVERY=500 \
    WALLCLOCK_CHECK_START_STEP=4000 \
    WARMUP_STEPS=20 \
    WARMDOWN_FRAC=0.85 \
    MIN_LR=0.1 \
    MATRIX_LR=0.026 \
    SCALAR_LR=0.02 \
    TIED_EMBED_LR=0.03 \
    MUON_MOMENTUM=0.97 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    MUON_ROW_NORMALIZE=1 \
    BETA2=0.99 \
    GRAD_CLIP_NORM=0.3 \
    ADAM_WD=0.02 \
    MUON_WD=0.095 \
    EMBED_WD=0.085 \
    EMA_DECAY=0.9965 \
    QK_GAIN_INIT=5.0 \
    ROPE_DIMS=16 \
    ROPE_TRAIN_SEQ_LEN=2048 \
    XSA_LAST_N=11 \
    NUM_LOOPS=2 \
    LOOP_START=3 \
    LOOP_END=5 \
    ENABLE_LOOPING_AT=0.35 \
    PARALLEL_START_LAYER=8 \
    PARALLEL_FINAL_LANE=mean \
    CASEOPS_ENABLED=1 \
    SKIP_GATES_ENABLED=1 \
    SMEAR_GATE_ENABLED=1 \
    SPARSE_ATTN_GATE_ENABLED=1 \
    GATE_WINDOW=12 \
    SPARSE_ATTN_GATE_SCALE=0.5 \
    SPARSE_ATTN_GATE_INIT_STD=0.0 \
    GATED_ATTN_QUANT_GATE=1 \
    FUSED_CE_ENABLED=1 \
    MATRIX_BITS=6 \
    EMBED_BITS=7 \
    MATRIX_CLIP_SIGMAS=12.85 \
    EMBED_CLIP_SIGMAS=14.0 \
    MLP_CLIP_SIGMAS=11.5 \
    ATTN_CLIP_SIGMAS=13.0 \
    COMPRESSOR=pergroup \
    GPTQ_CALIBRATION_BATCHES=16 \
    GPTQ_RESERVE_SECONDS=0.5 \
    LQER_ENABLED=1 \
    LQER_TOP_K=3 \
    LQER_RANK=4 \
    LQER_FACTOR_BITS=4 \
    LQER_ASYM_ENABLED=1 \
    LQER_ASYM_GROUP=64 \
    TTT_ENABLED=0 \
    PHASED_TTT_PREFIX_DOCS=2750 \
    PHASED_TTT_NUM_PHASES=3 \
    "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/train_gpt.py"
  ) 2>&1 | tee "${run_log}"
}

run_eval_seed() {
  local seed_value="$1"
  local artifact_run_id="shortchunk16_t2048_local0875_seed${seed_value}"
  local run_id="${artifact_run_id}_eval"
  local out_dir="${SCRIPT_DIR}/${artifact_run_id}"
  local run_log="${out_dir}/eval.log"
  local force_eval="${FORCE_EVAL:-${FORCE:-0}}"

  if [ ! -f "${out_dir}/final_model.int6.ptz" ]; then
    echo "missing quantized artifact: ${out_dir}/final_model.int6.ptz" >&2
    return 1
  fi

  ensure_dirs "${out_dir}"

  if [ "${force_eval}" != "1" ] \
    && [ -f "${run_log}" ] \
    && grep -q "quantized_ttt_phased" "${run_log}" \
    && grep -q "total_eval_time" "${run_log}"; then
    echo "eval cache hit for seed ${seed_value}: ${out_dir}"
    return 0
  fi

  (
    cd "${REPO_ROOT}"
    SEED="${seed_value}" \
    RUN_ID="${run_id}" \
    ARTIFACT_DIR="${out_dir}" \
    TMPDIR="${CACHE_DIR}/tmp" \
    XDG_CACHE_HOME="${CACHE_DIR}/xdg" \
    TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torchinductor" \
    TRITON_CACHE_DIR="${CACHE_DIR}/triton" \
    CUDA_CACHE_PATH="${CACHE_DIR}/cuda" \
    PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
    TTT_EVAL_ONLY=1 \
    TTT_LORA_FASTPATH=1 \
    DESERIALIZE_CACHE=1 \
    DATA_PATH=data/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=data/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    VOCAB_SIZE=8192 \
    NUM_LAYERS=11 \
    NUM_KV_HEADS=4 \
    MODEL_DIM=512 \
    NUM_HEADS=8 \
    MLP_MULT=4 \
    TRAIN_SEQ_LEN=2048 \
    EVAL_SEQ_LEN=2560 \
    TRAIN_BATCH_TOKENS=786432 \
    VAL_BATCH_TOKENS=524288 \
    VAL_LOSS_EVERY=0 \
    TRAIN_LOG_EVERY=500 \
    WARMUP_STEPS=20 \
    WARMDOWN_FRAC=0.85 \
    MIN_LR=0.1 \
    MATRIX_LR=0.026 \
    SCALAR_LR=0.02 \
    TIED_EMBED_LR=0.03 \
    MUON_MOMENTUM=0.97 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 \
    MUON_ROW_NORMALIZE=1 \
    BETA2=0.99 \
    GRAD_CLIP_NORM=0.3 \
    ADAM_WD=0.02 \
    MUON_WD=0.095 \
    EMBED_WD=0.085 \
    EMA_DECAY=0.9965 \
    QK_GAIN_INIT=5.0 \
    ROPE_DIMS=16 \
    ROPE_TRAIN_SEQ_LEN=2048 \
    XSA_LAST_N=11 \
    NUM_LOOPS=2 \
    LOOP_START=3 \
    LOOP_END=5 \
    ENABLE_LOOPING_AT=0.35 \
    PARALLEL_START_LAYER=8 \
    PARALLEL_FINAL_LANE=mean \
    CASEOPS_ENABLED=1 \
    SKIP_GATES_ENABLED=1 \
    SMEAR_GATE_ENABLED=1 \
    SPARSE_ATTN_GATE_ENABLED=1 \
    GATE_WINDOW=12 \
    SPARSE_ATTN_GATE_SCALE=0.5 \
    SPARSE_ATTN_GATE_INIT_STD=0.0 \
    GATED_ATTN_QUANT_GATE=1 \
    FUSED_CE_ENABLED=1 \
    MATRIX_BITS=6 \
    EMBED_BITS=7 \
    MATRIX_CLIP_SIGMAS=12.85 \
    EMBED_CLIP_SIGMAS=14.0 \
    MLP_CLIP_SIGMAS=11.5 \
    ATTN_CLIP_SIGMAS=13.0 \
    COMPRESSOR=pergroup \
    GPTQ_CALIBRATION_BATCHES=16 \
    GPTQ_RESERVE_SECONDS=0.5 \
    GPTQ_PACKING_FASTPATH=0 \
    LQER_ENABLED=1 \
    LQER_TOP_K=3 \
    LQER_RANK=4 \
    LQER_FACTOR_BITS=4 \
    LQER_ASYM_ENABLED=1 \
    LQER_ASYM_GROUP=64 \
    TTT_ENABLED=1 \
    TTT_LORA_RANK=224 \
    TTT_LORA_LR=0.00007 \
    TTT_LORA_ALPHA=144 \
    TTT_CHUNK_SIZE=48 \
    TTT_EVAL_SEQ_LEN=2560 \
    TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-64}" \
    TTT_GRAD_STEPS=1 \
    TTT_LOCAL_LR_MULT=0.875 \
    TTT_WEIGHT_DECAY=0.5 \
    TTT_BETA1=0 \
    TTT_BETA2=0.99 \
    TTT_Q_LORA=1 \
    TTT_V_LORA=1 \
    TTT_K_LORA=1 \
    TTT_MLP_LORA=1 \
    TTT_O_LORA=1 \
    TTT_OPTIMIZER=adam \
    SHORT_TTT_THRESHOLD=2048 \
    SHORT_TTT_CHUNK_SIZE=16 \
    SHORT_TTT_LR_MULT=1.0 \
    SHORT_TTT_GRAD_STEPS=1 \
    SHORT_TTT_GRAD_THRESHOLD=2048 \
    PHASED_TTT_PREFIX_DOCS=0 \
    PHASED_TTT_NUM_PHASES=3 \
    GLOBAL_TTT_LR=0.001 \
    GLOBAL_TTT_MOMENTUM=0.9 \
    GLOBAL_TTT_EPOCHS=1 \
    GLOBAL_TTT_CHUNK_TOKENS=32768 \
    GLOBAL_TTT_BATCH_SEQS=32 \
    GLOBAL_TTT_GRAD_CLIP=1.0 \
    GLOBAL_TTT_RESPECT_DOC_BOUNDARIES=1 \
    "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/train_gpt_eval.py"
  ) 2>&1 | tee "${run_log}"
}

for seed in "${SEEDS[@]}"; do
  run_train_seed "${seed}"
  run_eval_seed "${seed}"
done

python3 "${SCRIPT_DIR}/summarize_results.py" "${SEEDS[@]}" | tee "${SCRIPT_DIR}/results_summary.txt"
