#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-Frosty40/10k_caseops_golfer}"
OUT_ROOT="${OUT_ROOT:-/home/frosty40/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets}"
HF_BIN="${HF_BIN:-$(command -v hf)}"
ACTION="${1:-check}"

TOKENIZER_MODEL="${OUT_ROOT}/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model"
TOKENIZER_VOCAB="${OUT_ROOT}/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.vocab"
DATASET_DIR="${OUT_ROOT}/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved"
MANIFEST="${OUT_ROOT}/caseops_manifest.json"

if [[ -z "${HF_BIN}" || ! -x "${HF_BIN}" ]]; then
  echo "FATAL: hf CLI not found. Set HF_BIN=/path/to/hf." >&2
  exit 1
fi

"${HF_BIN}" auth whoami >/dev/null

missing=0
for path in "${TOKENIZER_MODEL}" "${TOKENIZER_VOCAB}" "${DATASET_DIR}" "${MANIFEST}"; do
  if [[ ! -e "${path}" ]]; then
    echo "MISSING: ${path}" >&2
    missing=1
  fi
done

train_count=$(find "${DATASET_DIR}" -maxdepth 1 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l || true)
val_count=$(find "${DATASET_DIR}" -maxdepth 1 -name 'fineweb_val_*.bin' 2>/dev/null | wc -l || true)
val_bytes_count=$(find "${DATASET_DIR}" -maxdepth 1 -name 'fineweb_val_bytes_*.bin' 2>/dev/null | wc -l || true)

echo "repo=${REPO}"
echo "out_root=${OUT_ROOT}"
echo "train_shards=${train_count} val_shards=${val_count} val_byte_sidecars=${val_bytes_count}"

if [[ "${missing}" -ne 0 || "${train_count}" -eq 0 || "${val_count}" -eq 0 || "${val_bytes_count}" -eq 0 ]]; then
  echo "Dataset is not upload-ready yet." >&2
  exit 2
fi

if [[ "${ACTION}" == "check" ]]; then
  echo "Upload-ready. To upload:"
  echo "  REPO=${REPO} OUT_ROOT=${OUT_ROOT} HF_BIN=${HF_BIN} $0 upload"
  exit 0
fi

if [[ "${ACTION}" != "upload" ]]; then
  echo "Usage: $0 [check|upload]" >&2
  exit 64
fi

"${HF_BIN}" repo create "${REPO}" --repo-type dataset --exist-ok
"${HF_BIN}" upload-large-folder "${REPO}" "${OUT_ROOT}" --repo-type dataset
echo "DONE: https://huggingface.co/datasets/${REPO}"
