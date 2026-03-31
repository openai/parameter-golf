#!/bin/bash
set -euo pipefail
# CLOWNCAR_VII: ClownCar_II base — EMA disabled, loop-aware GPTQ
#
# Same arch as ClownCar_II. Two changes only:
#   SKIP_EMA=1       — use live model weights (CC_II EMA dragged 0.47 → 0.73 BPB)
#   LOOP_AWARE_GPTQ=1 — 2-phase GPTQ: flat Hessians (phase1) then crawler Hessians
#                       with quantized-flat activations (phase2). Crawler GPTQ now
#                       compensates against the real drifted inputs it sees at inference.
#
# Hypothesis: Medusa (naive int6, no EMA) got 1.51 BPB roundtrip because:
#   1. naive int6 sent crawler weights through int6 (not int8) — 4x error amplification
#   2. GPTQ Hessians for crawler calibrated on fp16 inter-loop activations, not
#      quantized-flat activations — crawler fixed-point unravels under distribution shift
# Fix: re-enable GPTQ with loop-aware 2-phase calibration.
#
# Baseline: ClownCar_II sliding window 1.0427 BPB (int6+GPTQ, EMA applied)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

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
echo "  CLOWNCAR_VII — live weights, loop-aware GPTQ"
echo "  Seed: ${SEED}"
echo "  inst_dim=32 FLOW | 4 flat + 1 crawler x 4 loops"
echo "  DELTA_NET_HEADS=4 | chunk_delta_rule | short_conv=True"
echo "  SKIP_EMA=1 | LOOP_AWARE_GPTQ=1 | ngram eval DISABLED"
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
SKIP_EMA=1 \
LOOP_AWARE_GPTQ=1 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/clowncar7_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
