#!/bin/bash
set -euo pipefail
# bandit_wagon_XSA: XSA coverage sweep on confirmed-optimal 4F+1C config
#
# Config locked to CL3/BW5F proven findings:
#   NUM_FLAT_LAYERS=4      (BW5F confirmed optimal)
#   CRAWLER_LOOPS=3        (CL1)
#   CRAWLER_MLP_MULT=6.0   (CL3)
#   CRAWLER_QUANT_INT8=1   (CL1: mandatory)
#   SKIP_GPTQ=1            (CL3)
#   SKIP_EMA=1             (Ablations_v1)
#   COMPILE_FULLGRAPH=0    (CL3)
#
# Primary lever: XSA_LAST_N
#   4F+1C x3 = 15 total blocks
#   XSA_LAST_N=11 → 73%  (current SOTA config)
#   XSA_LAST_N=13 → 87%  (BWXSA-01)
#   XSA_LAST_N=15 → 100% (BWXSA-02, ceiling)
#
# Override: XSA_LAST_N=13 bash experiments/bandit_wagon_XSA/run.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"
XSA_LAST_N="${XSA_LAST_N:-11}"

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
echo "  bandit_wagon_XSA — XSA coverage sweep"
echo "  Seed: ${SEED}"
echo "  MODEL_DIM=512 | inst_dim=32 FLOW | 4F+1C x 3 loops | DN=0"
echo "  mlp_mult=6.0 | XSA_LAST_N=${XSA_LAST_N} (of 15 blocks) | SKIP_GPTQ=1 | CRAWLER_QUANT_INT8=1"
echo "  (BW5F confirmed: 4F+1C optimal, quant gap is the lever)"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE} | NITRUST_STRICT=${NITRUST_STRICT}"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=2000 \
XSA_LAST_N="${XSA_LAST_N}" \
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
    2>&1 | tee "logs/bwxsa_xsa${XSA_LAST_N}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
