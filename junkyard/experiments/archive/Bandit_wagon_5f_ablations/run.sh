#!/bin/bash
set -euo pipefail
# BANDIT_WAGON_5F_ABLATIONS: 4F vs 5F direct comparison with XSA coverage sweep
#
# Config locked to CL3/BW proven findings:
#   CRAWLER_LOOPS=3        (CL1-01)
#   CRAWLER_MLP_MULT=6.0   (CL3)
#   CRAWLER_QUANT_INT8=1   (CL1-08: mandatory)
#   SKIP_GPTQ=1            (CL3)
#   SKIP_EMA=1             (Ablations_v1)
#   COMPILE_FULLGRAPH=0    (CL3)
#
# Key overrides vs BW run.sh:
#   XSA_LAST_N is now an env var (was hardcoded to 11 in BW)
#   NUM_FLAT_LAYERS is the primary lever
#
# Arms:
#   BW2-00  4F+1C, XSA_LAST_N=11  (control — missing from BW ablation)
#   BW2-01  5F+1C, XSA_LAST_N=14  (proportional XSA for 18-block model)
#   (BW-03  5F+1C, XSA_LAST_N=11  reference, already run → 1.54404)
#
# Override: NUM_FLAT_LAYERS=5 XSA_LAST_N=14 bash experiments/Bandit_wagon_5f_ablations/run.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"
MODEL_DIM="${MODEL_DIM:-512}"
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-4}"
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
echo "  BANDIT_WAGON_5F — 4F vs 5F + XSA sweep"
echo "  Seed: ${SEED}"
echo "  MODEL_DIM=${MODEL_DIM} | inst_dim=32 FLOW | ${NUM_FLAT_LAYERS}F+1C x 3 loops | DN=0"
echo "  mlp_mult=6.0 | XSA_LAST_N=${XSA_LAST_N} | SKIP_GPTQ=1 | CRAWLER_QUANT_INT8=1"
echo "  (CL3 proven: 1.18742 mean BPB, 3-seed)"
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
MODEL_DIM="${MODEL_DIM}" \
USE_CRAWLER=1 \
NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS}" \
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
    2>&1 | tee "logs/bw5f_f${NUM_FLAT_LAYERS}_xsa${XSA_LAST_N}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
