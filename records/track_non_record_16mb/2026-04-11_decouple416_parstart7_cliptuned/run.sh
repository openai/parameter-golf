#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run.sh — exp118: Decouple416 + ParStart7 + ClipTuned + TTT Freeze
#
# Data-informed design from exp114 .pt analysis:
#
# 1. EMBEDDING_DIM=416 (gap=96d, saves 360K params / ~0.4MB)
#    - Decouples tied embedding → frees boundary blocks
#    - exp114 proved: B10 +42.5% eff, routing normalized, V-shape flattened
#    - 416 is milder than 384 (only -360K vs -655K params)
#
# 2. PARALLEL_RESIDUAL_START=7 (was 8)
#    - With B10 now active (decoupled), 4 blocks get two-lane routing (B7-B10)
#    - Parent couldn't benefit: B10 was dead. Now it's alive.
#
# 3. MATRIX_CLIP_SIGMAS=12.0 (was 12.85)
#    - Rate-distortion optimal: 0.4MB saved by smaller tok_emb → spend on
#      tighter clipping → lower quant error (11.2% vs 11.97%)
#
# 4. TTT_FREEZE_BLOCKS=2
#    - Freeze B0-B1 during TTT (still weakest after decoupling)
#    - Focus TTT gradient on activated middle+boundary blocks
#    - Proven 2.4x better TTT gains in competition research
#
# Hardware: 1 node x 8 H100 80GB SXM (600s wallclock)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

echo "=== Installing dependencies ==="
pip install -q brotli sentencepiece
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ \
  2>/dev/null || echo "WARN: flash_attn_3 wheel install failed -- check CUDA/PyTorch versions"

echo "=== Downloading SP8192 tokenizer & FineWeb shards ==="
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

echo "=== Copying experiment files to repo root ==="
cp "$SCRIPT_DIR/train_gpt.py" "$REPO_ROOT/train_gpt.py"
if [ -d "$SCRIPT_DIR/cutlass_evt_fusion" ]; then
    cp -r "$SCRIPT_DIR/cutlass_evt_fusion" "$REPO_ROOT/cutlass_evt_fusion"
    echo "Copied cutlass_evt_fusion/"
fi

echo "=== Starting training on 8xH100 ==="
SEED="${SEED:-42}" \
TTT_ENABLED=1 \
HASH_EMBED_ENABLED=1 \
TTT_LR=0.01 \
TTT_FREEZE_BLOCKS=2 \
MUON_MOMENTUM=0.97 \
PARALLEL_RESIDUAL_START=7 \
EMBEDDING_DIM=416 \
MATRIX_CLIP_SIGMAS=12.0 \
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train_gpt.py

echo "=== Done ==="
