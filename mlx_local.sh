#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage:
  bash mlx_local.sh setup
  bash mlx_local.sh download [train_shards]
  bash mlx_local.sh run [standard|sliding]
  bash mlx_local.sh compare <model_path> [stride]

Examples:
  bash mlx_local.sh setup
  bash mlx_local.sh download 1
  bash mlx_local.sh run standard
  bash mlx_local.sh run sliding
  bash mlx_local.sh compare logs/my_run_mlx_model.int8.ptz

Optional environment variables for `run`:
  RUN_ID
  DATA_PATH
  TOKENIZER_PATH
  ITERATIONS
  TRAIN_BATCH_TOKENS
  VAL_LOSS_EVERY
  VAL_BATCH_SIZE
  TRAIN_SEQ_LEN
  EVAL_STRIDE
  EVAL_BATCH_SEQS
  MUON_WEIGHT_DECAY
  ADAM_WEIGHT_DECAY
  SEED
  OUT_DIR
EOF
}

ensure_venv_exists() {
    if [[ ! -d .venv ]]; then
        echo ".venv not found. Run: bash mlx_local.sh setup" >&2
        exit 1
    fi
}

activate_venv() {
    # shellcheck disable=SC1091
    source .venv/bin/activate
}

check_download_deps() {
    if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
import sentencepiece
import huggingface_hub
import datasets
import tqdm
PY
    then
        echo "Required packages are missing in .venv. Run: bash mlx_local.sh setup" >&2
        exit 1
    fi
}

check_run_deps() {
    if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
import sentencepiece
import mlx.core
import mlx.nn
PY
    then
        echo "Required MLX packages are missing in .venv. Run: bash mlx_local.sh setup" >&2
        exit 1
    fi
}

cmd_setup() {
    if [[ ! -d .venv ]]; then
        python3 -m venv .venv
    fi

    activate_venv
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache}"

    python -m pip install --upgrade pip
    python -m pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

    echo
    echo "MLX local environment is ready."
    echo "Next steps:"
    echo "  bash mlx_local.sh download 1"
    echo "  bash mlx_local.sh run standard"
    echo "  bash mlx_local.sh run sliding"
}

cmd_download() {
    local train_shards="${1:-1}"
    ensure_venv_exists
    activate_venv
    check_download_deps

    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${train_shards}"

    echo
    echo "Downloaded FineWeb sp1024 data with train_shards=${train_shards}."
    echo "Dataset path: ./data/datasets/fineweb10B_sp1024"
    echo "Tokenizer path: ./data/tokenizers/fineweb_1024_bpe.model"
}

cmd_run() {
    local mode="${1:-standard}"
    ensure_venv_exists

    if [[ ! -d ./data/datasets/fineweb10B_sp1024 ]]; then
        echo "Dataset not found. Run: bash mlx_local.sh download 1" >&2
        exit 1
    fi

    activate_venv
    check_run_deps

    local data_path="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
    local tokenizer_path="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
    local train_seq_len="${TRAIN_SEQ_LEN:-1024}"
    local iterations="${ITERATIONS:-200}"
    local train_batch_tokens="${TRAIN_BATCH_TOKENS:-8192}"
    local val_loss_every="${VAL_LOSS_EVERY:-0}"
    local val_batch_size="${VAL_BATCH_SIZE:-8192}"
    local eval_batch_seqs="${EVAL_BATCH_SEQS:-32}"
    local muon_weight_decay="${MUON_WEIGHT_DECAY:-0.0}"
    local adam_weight_decay="${ADAM_WEIGHT_DECAY:-0.0}"
    local seed="${SEED:-1337}"
    local run_id="${RUN_ID:-mlx_${mode}_$(date +%Y%m%d_%H%M%S)}"
    local eval_stride

    case "${mode}" in
        standard)
            eval_stride="${EVAL_STRIDE:-$train_seq_len}"
            ;;
        sliding)
            eval_stride="${EVAL_STRIDE:-64}"
            ;;
        *)
            echo "Usage: bash mlx_local.sh run [standard|sliding]" >&2
            exit 1
            ;;
    esac

    echo "mode=${mode}"
    echo "run_id=${run_id}"
    echo "iterations=${iterations}"
    echo "train_batch_tokens=${train_batch_tokens}"
    echo "val_batch_size=${val_batch_size}"
    echo "train_seq_len=${train_seq_len}"
    echo "eval_stride=${eval_stride}"
    echo "eval_batch_seqs=${eval_batch_seqs}"
    echo "muon_weight_decay=${muon_weight_decay}"
    echo "adam_weight_decay=${adam_weight_decay}"

    RUN_ID="${run_id}" \
    DATA_PATH="${data_path}" \
    TOKENIZER_PATH="${tokenizer_path}" \
    ITERATIONS="${iterations}" \
    TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
    VAL_LOSS_EVERY="${val_loss_every}" \
    VAL_BATCH_SIZE="${val_batch_size}" \
    TRAIN_SEQ_LEN="${train_seq_len}" \
    EVAL_STRIDE="${eval_stride}" \
    EVAL_BATCH_SEQS="${eval_batch_seqs}" \
    MUON_WEIGHT_DECAY="${muon_weight_decay}" \
    ADAM_WEIGHT_DECAY="${adam_weight_decay}" \
    SEED="${seed}" \
    python3 train_gpt_mlx.py
}

cmd_compare() {
    local model_path="${1:-}"
    local sliding_stride="${2:-64}"
    ensure_venv_exists

    if [[ -z "${model_path}" ]]; then
        echo "Usage: bash mlx_local.sh compare <model_path> [stride]" >&2
        exit 1
    fi
    if [[ ! -f "${model_path}" ]]; then
        echo "Model file not found: ${model_path}" >&2
        exit 1
    fi
    if [[ ! -d ./data/datasets/fineweb10B_sp1024 ]]; then
        echo "Dataset not found. Run: bash mlx_local.sh download 1" >&2
        exit 1
    fi

    activate_venv
    check_run_deps

    local data_path="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
    local tokenizer_path="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
    local train_seq_len="${TRAIN_SEQ_LEN:-1024}"
    local iterations="${ITERATIONS:-0}"
    local train_batch_tokens="${TRAIN_BATCH_TOKENS:-8192}"
    local val_loss_every="${VAL_LOSS_EVERY:-0}"
    local val_batch_size="${VAL_BATCH_SIZE:-8192}"
    local eval_batch_seqs="${EVAL_BATCH_SEQS:-32}"
    local seed="${SEED:-1337}"
    local out_dir="${OUT_DIR:-logs}"
    local run_base="${RUN_ID_BASE:-mlx_compare_$(date +%Y%m%d_%H%M%S)}"
    local std_run_id="${run_base}_standard"
    local slide_run_id="${run_base}_sliding"
    local std_log="${out_dir}/${std_run_id}.txt"
    local slide_log="${out_dir}/${slide_run_id}.txt"

    echo "compare_model=${model_path}"
    echo "standard_run_id=${std_run_id}"
    echo "sliding_run_id=${slide_run_id}"
    echo "sliding_stride=${sliding_stride}"
    echo "eval_batch_seqs=${eval_batch_seqs}"

    RUN_ID="${std_run_id}" \
    OUT_DIR="${out_dir}" \
    DATA_PATH="${data_path}" \
    TOKENIZER_PATH="${tokenizer_path}" \
    ITERATIONS="${iterations}" \
    TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
    VAL_LOSS_EVERY="${val_loss_every}" \
    VAL_BATCH_SIZE="${val_batch_size}" \
    TRAIN_SEQ_LEN="${train_seq_len}" \
    EVAL_STRIDE="${train_seq_len}" \
    EVAL_BATCH_SEQS="${eval_batch_seqs}" \
    SEED="${seed}" \
    EVAL_ONLY=1 \
    LOAD_MODEL_PATH="${model_path}" \
    python3 train_gpt_mlx.py

    RUN_ID="${slide_run_id}" \
    OUT_DIR="${out_dir}" \
    DATA_PATH="${data_path}" \
    TOKENIZER_PATH="${tokenizer_path}" \
    ITERATIONS="${iterations}" \
    TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
    VAL_LOSS_EVERY="${val_loss_every}" \
    VAL_BATCH_SIZE="${val_batch_size}" \
    TRAIN_SEQ_LEN="${train_seq_len}" \
    EVAL_STRIDE="${sliding_stride}" \
    EVAL_BATCH_SEQS="${eval_batch_seqs}" \
    SEED="${seed}" \
    EVAL_ONLY=1 \
    LOAD_MODEL_PATH="${model_path}" \
    python3 train_gpt_mlx.py

    STD_LOG="${std_log}" SLIDE_LOG="${slide_log}" python3 - <<'PY'
import pathlib
import re
import os

std_log = pathlib.Path(os.environ["STD_LOG"])
slide_log = pathlib.Path(os.environ["SLIDE_LOG"])
pat = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:(\\S+) val_bpb:(\\S+)")
time_pat = re.compile(r"final_int8_zlib_roundtrip val_loss:\\S+ val_bpb:\\S+ eval_time:(\\S+)ms")

def extract(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    m = pat.search(text)
    t = time_pat.search(text)
    if not m or not t:
        raise SystemExit(f"Could not parse final metrics from {path}")
    return float(m.group(1)), float(m.group(2)), float(t.group(1))

std_loss, std_bpb, std_ms = extract(std_log)
slide_loss, slide_bpb, slide_ms = extract(slide_log)
print("")
print("Comparison")
print(f"  standard loss={std_loss:.8f} bpb={std_bpb:.8f} eval_ms={std_ms:.0f}")
print(f"  sliding  loss={slide_loss:.8f} bpb={slide_bpb:.8f} eval_ms={slide_ms:.0f}")
print(f"  delta_loss={slide_loss - std_loss:+.8f}")
print(f"  delta_bpb={slide_bpb - std_bpb:+.8f}")
print(f"  eval_time_ratio={slide_ms / std_ms:.3f}x")
PY
}

main() {
    local cmd="${1:-}"
    case "${cmd}" in
        setup)
            shift
            cmd_setup "$@"
            ;;
        download)
            shift
            cmd_download "$@"
            ;;
        run)
            shift
            cmd_run "$@"
            ;;
        compare)
            shift
            cmd_compare "$@"
            ;;
        ""|-h|--help|help)
            usage
            ;;
        *)
            echo "Unknown command: ${cmd}" >&2
            echo >&2
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
