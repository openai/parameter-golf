#!/bin/bash
set -euo pipefail
# MEDUSA_IV: late-start EMA + loop-aware GPTQ
#
# Insight from CC_VII: EMA wasn't just lagging — it was smoothing weights to be
# quantization-friendly. Live model after warmdown has spiky weights that GPTQ
# can't fully compensate (+0.636 gap vs EMA's +0.206 gap).
#
# Fix: late-start EMA.
#   EMA_START_STEP=4400  — re-initialize EMA at SWA/warmdown onset, skip early steps
#   EMA_DECAY=0.99       — fast decay: closely tracks warmdown weights (~400 steps)
#   LOOP_AWARE_GPTQ=1    — 2-phase crawler calibration (from CC_VII)
#
# Expected: BPB close to live (~0.47-0.52), weights smooth enough for GPTQ (+0.20 gap)
# Baseline: CC_II 1.0427 BPB | CC_VII roundtrip 1.1879 BPB

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"

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

echo "[preflight] checking fla.ops.delta_rule (canonical DeltaNet kernel)..."
python3 -c "
from fla.ops.delta_rule import chunk_delta_rule
print('  chunk_delta_rule OK — CANONICAL kernel active')
" 2>/dev/null || echo "  WARNING: fla.ops not found — will fall back to Python DeltaNet loop (slow, non-canonical)"

echo "============================================"
echo "  MEDUSA_IV — late-start EMA (step 4400) + loop-aware GPTQ"
echo "  Seed: ${SEED}"
echo "  inst_dim=32 FLOW | 4 flat + 1 crawler x 4 loops"
echo "  DELTA_NET_HEADS=4 | chunk_delta_rule | short_conv=True"
echo "  EMA_START_STEP=4400 | EMA_DECAY=0.99 | LOOP_AWARE_GPTQ=1"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE} | NITRUST_STRICT=${NITRUST_STRICT}"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=2000 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
LATE_QAT_THRESHOLD=0 \
MATRIX_LR=0.03 \
TORCHDYNAMO_OPTIMIZE_DDP=0 \
COMPILE_FULLGRAPH=0 \
NGRAM_EVAL_ORDER=0 \
USE_CRAWLER=1 \
NUM_FLAT_LAYERS=4 \
NUM_CRAWLER_LAYERS=1 \
CRAWLER_LOOPS=4 \
INST_DIM=32 \
CRAWLER_QUANT_INT8=1 \
DELTA_NET_HEADS=4 \
EMA_START_STEP=4400 \
EMA_DECAY=0.99 \
LOOP_AWARE_GPTQ=1 \
NITRUST_ENABLE="${NITRUST_ENABLE}" \
NITRUST_STRICT="${NITRUST_STRICT}" \
NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/medusa4_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
