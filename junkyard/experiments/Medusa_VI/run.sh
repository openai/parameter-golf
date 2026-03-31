#!/bin/bash
set -euo pipefail
# MEDUSA_VI: state dtype fix + DeltaNet CastedLinear QAT
#
# Medusa_V proved the dtype fix dramatically improved training (0.2508 FP BPB)
# but made unravel worse (1.0463 int6 SW) because DeltaNet projections used
# plain nn.Linear — invisible to QAT (CastedLinear._qat_enabled).
#
# Fix: DeltaNet q/k/v/o_proj → CastedLinear in both DeltaNet classes.
# Effect: QAT now covers crawler weights during warmdown. The 4-loop structure
# gives the crawler 4x QAT gradient signal per step vs flat layers — exactly
# proportional to the 4x error compounding that caused unravel.
# b_proj kept as nn.Linear (bias=True, not exported via GPTQ).
#
# Medusa_V seed 44: FP 0.2508 → int6 SW 1.0463 (gap +0.788)
# Medusa_VI target: same FP quality, gap < 0.4

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
echo "  MEDUSA_VI — state dtype fix + DeltaNet CastedLinear QAT"
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
    2>&1 | tee "logs/medusa6_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
