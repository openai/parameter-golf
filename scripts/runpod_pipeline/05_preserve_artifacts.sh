#!/usr/bin/env bash
# Stage 5: Tarball runs/ + checkpoints/, upload to configurable destination
# Set UPLOAD_TARGET before running:
#   hf:<repo_id>:<path_in_repo>   → HuggingFace Hub dataset upload
#   rsync:<user@host>:<remote>    → rsync to remote (e.g. Pegasus)
# If UPLOAD_TARGET is unset, tarball is saved locally only.
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
RUNS_DIR="${REPO_DIR}/runs"
CKPT_BASE="/workspace/checkpoints"
PYTHON="/opt/pg-venv/bin/python"
UPLOAD_TARGET="${UPLOAD_TARGET:-}"
TARBALL="/workspace/runs_$(date +%Y%m%d_%H%M).tar.gz"

mkdir -p "${RUNS_DIR}"
exec > >(tee -a "${RUNS_DIR}/05_preserve.log") 2>&1

echo "=== Stage 5: Artifact preservation === $(date)"

# Free space check before creating tarball
FREE_GB=$(df -BG /workspace --output=avail 2>/dev/null | tail -1 | tr -d 'G ')
[ "${FREE_GB:-0}" -lt 5 ] && { echo "ERROR: < 5G free — cannot create tarball" >&2; exit 1; }

# Provenance capture — write BEFORE tarball so files land inside it
echo ""
echo "=== Capturing provenance fingerprints ==="
cd "${REPO_DIR}"

# Ancestry guard: HEAD must descend from the warmup-fix anchor (a33191f)
ANCHOR_SHA="a33191f572430566b88c4d61badb0369e1e6f9a3"
if ! git merge-base --is-ancestor "${ANCHOR_SHA}" HEAD 2>/dev/null; then
    echo "FATAL: HEAD ($(git rev-parse HEAD)) is not a descendant of ${ANCHOR_SHA}" >&2
    echo "       Tarball would lie about provenance; refusing to continue." >&2
    exit 1
fi

{
  git rev-parse HEAD
  git log -1 --format='%H %s'
  git log -1 --format='  author: %an <%ae>%n  date:   %ai'
} > "${RUNS_DIR}/commit_sha.txt"

nvidia-smi > "${RUNS_DIR}/hardware_info.txt" 2>&1 || \
  echo "nvidia-smi unavailable" > "${RUNS_DIR}/hardware_info.txt"

"${PYTHON}" - > "${RUNS_DIR}/env_fingerprint.txt" <<'PY'
import torch
try:
    import flash_attn_interface
    fa_ver = getattr(flash_attn_interface, "__version__", "unknown")
except Exception as e:
    fa_ver = f"import-failed: {e!r}"
print(f"torch                 {torch.__version__}")
print(f"cuda (torch-reported) {torch.version.cuda}")
print(f"flash_attn_interface  {fa_ver}")
print(f"python                {__import__('sys').version.split()[0]}")
PY

# Non-empty guard — catches silent failures
for f in "${RUNS_DIR}/commit_sha.txt" "${RUNS_DIR}/hardware_info.txt" "${RUNS_DIR}/env_fingerprint.txt"; do
    [ -s "$f" ] || { echo "FATAL: $f is empty — provenance capture failed" >&2; exit 1; }
done

echo "Provenance fingerprints captured:"
wc -l "${RUNS_DIR}/commit_sha.txt" "${RUNS_DIR}/hardware_info.txt" "${RUNS_DIR}/env_fingerprint.txt"

cd /workspace

echo "Creating tarball: ${TARBALL}"
echo "Including: parameter-golf/runs/ + checkpoints/"
tar czf "${TARBALL}" \
    --exclude="parameter-golf/data" \
    --exclude="parameter-golf/.venv" \
    --exclude="parameter-golf/.hf" \
    --exclude="parameter-golf/.cache" \
    parameter-golf/runs/ \
    checkpoints/ \
    2>/dev/null

TARBALL_SIZE=$(du -sh "${TARBALL}" | cut -f1)
echo "Tarball created: ${TARBALL} (${TARBALL_SIZE})"

# Print what's inside
echo ""
echo "Contents summary:"
tar tzf "${TARBALL}" | grep -E "\.json$|\.txt$|\.ptz$|\.pt$" | head -40
echo "(full listing: tar tzf ${TARBALL})"

if [ -z "${UPLOAD_TARGET}" ]; then
    echo ""
    echo "WARNING: UPLOAD_TARGET not set. Tarball saved locally only."
    echo "  Set UPLOAD_TARGET and re-run before terminating the pod:"
    echo "    UPLOAD_TARGET=hf:<repo_id>:<path> bash scripts/runpod_pipeline/05_preserve_artifacts.sh"
    echo "    UPLOAD_TARGET=rsync:<user@host>:<remote_path> bash scripts/runpod_pipeline/05_preserve_artifacts.sh"
    echo ""
    echo "  Tarball path: ${TARBALL}"
    exit 0
fi

TARGET_TYPE="${UPLOAD_TARGET%%:*}"
TARGET_PATH="${UPLOAD_TARGET#*:}"
TARBALL_BASENAME=$(basename "${TARBALL}")

echo ""
echo "Uploading to: ${UPLOAD_TARGET}"

case "${TARGET_TYPE}" in

hf)
    # hf:<repo_id>:<path_in_repo>
    HF_REPO="${TARGET_PATH%%:*}"
    HF_SUBPATH="${TARGET_PATH#*:}"
    DEST="${HF_REPO}/${HF_SUBPATH}/${TARBALL_BASENAME}"
    echo "HuggingFace Hub: ${DEST}"
    "${PYTHON}" - "${TARBALL}" "${HF_REPO}" "${HF_SUBPATH}/${TARBALL_BASENAME}" <<'PY'
import sys, os
from huggingface_hub import HfApi

local_path    = sys.argv[1]
repo_id       = sys.argv[2]
path_in_repo  = sys.argv[3]

size_mb = os.path.getsize(local_path) / 1_000_000
print(f"Uploading {local_path} ({size_mb:.1f} MB) → {repo_id}/{path_in_repo}")
api = HfApi()
url = api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="model",
)
print(f"Upload complete: {url}")
PY
    ;;

rsync)
    # rsync:<user@host>:<remote_path>
    REMOTE_HOST="${TARGET_PATH%%:*}"
    REMOTE_PATH="${TARGET_PATH#*:}"
    echo "rsync: ${REMOTE_HOST}:${REMOTE_PATH}/${TARBALL_BASENAME}"
    rsync -avz --progress "${TARBALL}" "${REMOTE_HOST}:${REMOTE_PATH}/"
    echo "rsync complete."
    ;;

*)
    echo "ERROR: Unknown UPLOAD_TARGET type '${TARGET_TYPE}'." >&2
    echo "  Supported: hf:<repo_id>:<path>  |  rsync:<user@host>:<path>" >&2
    exit 1
    ;;
esac

echo ""
echo "=== Artifact preservation: DONE ==="
echo "Tarball: ${TARBALL} (${TARBALL_SIZE})"
echo "Uploaded to: ${UPLOAD_TARGET}"
echo ""
echo "Pod can now be safely terminated."
