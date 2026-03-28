#!/bin/bash
set -euo pipefail
# CLOWNCAR_IV: Canonical FLA DeltaNet + Crawler — GPTQ bypass + state dtype fix
#
# Changes from ClownCar_II:
#   1. new_state.to(dtype) — fixes chunk_delta_rule returning Float32 state in BF16
#      training, which caused torch.compile recompiles on every rank at eval time.
#   2. SKIP_GPTQ=1 — uses naive int6 instead of GPTQ. ClownCar_II showed GPTQ
#      degrading 0.7278 → 0.9340 (0.2062 BPB gap). Hypothesis: DeltaNet state
#      matrices have outlier weights that GPTQ handles poorly.
#
# ClownCar_II seed 1337: 1.0427 BPB (sliding window, int6+GPTQ)
# ClownCar_IV target:    beat 1.0427 with naive int6

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
" 2>/dev/null || echo "  WARNING: fla.ops not found"

echo "============================================"
echo "  CLOWNCAR_IV — Canonical FLA DeltaNet + GPTQ bypass"
echo "  Seed: ${SEED}"
echo "  inst_dim=32 FLOW | 4 flat + 1 crawler x 4 loops"
echo "  DELTA_NET_HEADS=4 | chunk_delta_rule | SKIP_GPTQ=1"
echo "  state dtype fix: new_state.to(dtype)"
echo "  ngram eval DISABLED — sliding window submission only"
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
SKIP_GPTQ=1 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/clowncar4_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
