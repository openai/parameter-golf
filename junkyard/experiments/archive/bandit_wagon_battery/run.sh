#!/bin/bash
set -euo pipefail
# bandit_wagon_battery: Per-loop RoPE scale sweep (crawler as sparse attention battery)
#
# CRAWLER_LOOP_ROPE_SCALES="1,1,1"   standard (control)
# CRAWLER_LOOP_ROPE_SCALES="1,3,9"   moderate ascending: loop 0 local, loop 2 9x wider
# CRAWLER_LOOP_ROPE_SCALES="1,5,25"  aggressive ascending
# CRAWLER_LOOP_ROPE_SCALES="9,3,1"   descending: loop 0 global, loop 2 local
#
# scale > 1 divides inv_freq by scale → lower frequencies → wider attention range
# scale=1 is identical to standard behavior

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"
CRAWLER_LOOP_ROPE_SCALES="${CRAWLER_LOOP_ROPE_SCALES:-1,1,1}"
# All other features disabled by default for clean single-variable testing
CRAWLER_MLP_CHOKE_DIM="${CRAWLER_MLP_CHOKE_DIM:-0}"
CRAWLER_LOOP_SMEAR="${CRAWLER_LOOP_SMEAR:-0}"
CRAWLER_TAP_DIM="${CRAWLER_TAP_DIM:-0}"

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
echo "  bandit_wagon_battery — per-loop RoPE scale sweep"
echo "  Seed: ${SEED}"
echo "  MODEL_DIM=512 | inst_dim=32 FLOW | 4F+1C x 3 loops | DN=0"
echo "  CRAWLER_LOOP_ROPE_SCALES=${CRAWLER_LOOP_ROPE_SCALES}"
echo "  CRAWLER_MLP_CHOKE_DIM=${CRAWLER_MLP_CHOKE_DIM} | CRAWLER_LOOP_SMEAR=${CRAWLER_LOOP_SMEAR} | CRAWLER_TAP_DIM=${CRAWLER_TAP_DIM}"
echo "  SKIP_GPTQ=1 | CRAWLER_QUANT_INT8=1"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=2000 \
MLP_LEAKY_SLOPE=0.5 \
CRAWLER_MLP_LEAKY_SLOPE=0.5 \
CRAWLER_MLP_CHOKE_DIM="${CRAWLER_MLP_CHOKE_DIM}" \
CRAWLER_LOOP_SMEAR="${CRAWLER_LOOP_SMEAR}" \
CRAWLER_TAP_DIM="${CRAWLER_TAP_DIM}" \
CRAWLER_TAP_LOOP_SPECIFIC=1 \
CRAWLER_TAP_LAYERS=all \
CRAWLER_LOOP_ROPE_SCALES="${CRAWLER_LOOP_ROPE_SCALES}" \
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
    2>&1 | tee "logs/bwbat_scales${CRAWLER_LOOP_ROPE_SCALES//,/_}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
