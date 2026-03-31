#!/bin/bash
set -euo pipefail
# bandit_wagon_choke: Crawler MLP per-loop bottleneck choke sweep
#
# Config locked to confirmed-optimal + pending results:
#   NUM_FLAT_LAYERS=4      (BW5F confirmed)
#   XSA_LAST_N=11          (baseline; XSA=15 pending full combination)
#   CRAWLER_LOOPS=3        (CL1)
#   CRAWLER_MLP_MULT=6.0   (CL3)
#   CRAWLER_QUANT_INT8=1   (CL1: mandatory)
#   SKIP_GPTQ=1            (CL3)
#   SKIP_EMA=1             (Ablations_v1)
#   MLP_LEAKY_SLOPE=0.5    (flat blocks — LOCKED)
#   CRAWLER_MLP_LEAKY_SLOPE=0.5  (pending crawler_mlp results; use control value here)
#
# Primary lever: CRAWLER_MLP_CHOKE_DIM
#   0   = disabled (standard MLP, control)
#   32  = extreme choke (same as inst_dim FLOW bottleneck)
#   128 = moderate
#   256 = conservative
#   512 = minimal choke (matches model_dim)
#
# Override: CRAWLER_MLP_CHOKE_DIM=128 bash experiments/bandit_wagon_choke/run.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"
CRAWLER_MLP_CHOKE_DIM="${CRAWLER_MLP_CHOKE_DIM:-0}"
CRAWLER_MLP_LEAKY_SLOPE="${CRAWLER_MLP_LEAKY_SLOPE:-0.5}"

echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || echo "  WARNING: zstandard not found"

echo "[preflight] patching torch inductor AttrsDescriptor bug (if present)..."
python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('torch._inductor.runtime.hints')
if spec and spec.origin:
    p = pathlib.Path(spec.origin)
    txt = p.read_text()
    old = 'attr_desc_fields = {f.name for f in fields(AttrsDescriptor)}'
    if old in txt:
        import attr
        new = 'import attr as _attr; attr_desc_fields = {f.name for f in _attr.fields(AttrsDescriptor)}'
        p.write_text(txt.replace(old, new))
        print('  patched OK')
    else:
        print('  no patch needed')
" 2>/dev/null || echo "  WARNING: could not patch hints.py"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} detected — want FA3')
" 2>/dev/null || echo "  WARNING: no flash_attn found"

echo "============================================"
echo "  bandit_wagon_choke — crawler MLP per-loop choke sweep"
echo "  Seed: ${SEED}"
echo "  MODEL_DIM=512 | inst_dim=32 FLOW | 4F+1C x 3 loops | DN=0"
echo "  mlp_mult=3.0 (flat) | CRAWLER_MLP_MULT=6.0 | XSA_LAST_N=11"
echo "  MLP_LEAKY_SLOPE=0.5 (flat, locked) | CRAWLER_MLP_LEAKY_SLOPE=${CRAWLER_MLP_LEAKY_SLOPE}"
echo "  CRAWLER_MLP_CHOKE_DIM=${CRAWLER_MLP_CHOKE_DIM} (0=disabled/standard MLP)"
echo "  SKIP_GPTQ=1 | CRAWLER_QUANT_INT8=1"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE} | NITRUST_STRICT=${NITRUST_STRICT}"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=2000 \
MLP_LEAKY_SLOPE=0.5 \
CRAWLER_MLP_LEAKY_SLOPE="${CRAWLER_MLP_LEAKY_SLOPE}" \
CRAWLER_MLP_CHOKE_DIM="${CRAWLER_MLP_CHOKE_DIM}" \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
LATE_QAT_THRESHOLD=0 \
MATRIX_LR=0.03 \
TORCHDYNAMO_OPTIMIZE_DDP=0 \
COMPILE_FULLGRAPH=0 \
MODEL_DIM=512 \
USE_CRAWLER=1 \
NUM_FLAT_LAYERS=4 \
NUM_CRAWLER_LAYERS=1 \
CRAWLER_LOOPS=3 \
CRAWLER_MLP_MULT=6.0 \
INST_DIM=32 \
CRAWLER_QUANT_INT8=1 \
DELTA_NET_HEADS=0 \
SKIP_EMA=1 \
SKIP_GPTQ=1 \
LOOP_AWARE_GPTQ=0 \
NITRUST_ENABLE="${NITRUST_ENABLE}" \
NITRUST_STRICT="${NITRUST_STRICT}" \
NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/bwchoke_dim${CRAWLER_MLP_CHOKE_DIM}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
