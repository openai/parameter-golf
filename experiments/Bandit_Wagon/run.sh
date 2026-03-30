#!/bin/bash
set -euo pipefail
# BANDIT_WAGON: Bandit (crawler + X-WING oracle) with extra headroom ablations
#
# Bandit baseline: 0.4961 BPB @ 9.35 MB (6.65 MB headroom to 16 MB limit)
# Ablation arms — one variable at a time:
#   BW-00  dim=512  4F+1C×4  (anchor == Bandit)
#   BW-01  dim=576  4F+1C×4  (+2.28 MB — width lever)
#   BW-02  dim=640  4F+1C×4  (+4.83 MB — width lever max)
#   BW-03  dim=512  5F+1C×4  (+1.68 MB — depth lever)
#   BW-04  dim=512  6F+1C×4  (+3.36 MB — depth lever max)
#
# Override: MODEL_DIM=640 NUM_FLAT_LAYERS=4 bash experiments/Bandit_Wagon/run.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"
MODEL_DIM="${MODEL_DIM:-512}"
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-4}"

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
echo "  BANDIT_WAGON — crawler + X-WING oracle (headroom ablation)"
echo "  Seed: ${SEED}"
echo "  MODEL_DIM=${MODEL_DIM} | inst_dim=32 FLOW | ${NUM_FLAT_LAYERS} flat + 1 crawler x 4 loops | DN=0"
echo "  EMA_START_STEP=4400 | EMA_DECAY=0.99 | LOOP_AWARE_GPTQ=1"
echo "  NGRAM_EVAL_ORDER=9 | CUBRIC_CADENCE=32 | COMPLEMENT_ALPHA=0.5"
echo "  Shared n-gram tables | 3D Cubric 54-cell warm-start"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE} | NITRUST_STRICT=${NITRUST_STRICT}"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=2000 \
COMPLEMENT_ALPHA=0.5 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
LATE_QAT_THRESHOLD=0 \
MATRIX_LR=0.03 \
TORCHDYNAMO_OPTIMIZE_DDP=0 \
COMPILE_FULLGRAPH=0 \
MODEL_DIM="${MODEL_DIM}" \
USE_CRAWLER=1 \
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS}" \
NUM_CRAWLER_LAYERS=1 \
CRAWLER_LOOPS=4 \
INST_DIM=32 \
CRAWLER_QUANT_INT8=1 \
DELTA_NET_HEADS=0 \
EMA_START_STEP=4400 \
EMA_DECAY=0.99 \
LOOP_AWARE_GPTQ=1 \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.20 \
NGRAM_EVAL_ALPHA_MAX=0.75 \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS=8388608 \
CUBRIC_CADENCE=32 \
NITRUST_ENABLE="${NITRUST_ENABLE}" \
NITRUST_STRICT="${NITRUST_STRICT}" \
NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/bandit_wagon_d${MODEL_DIM}_f${NUM_FLAT_LAYERS}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
