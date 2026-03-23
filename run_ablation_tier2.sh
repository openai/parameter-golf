#!/bin/bash
# TIER2 Architecture Ablations — 3-min proxy runs on 1xH100
# Compare val_bpb at step ~2000 against baseline to pick best architecture
#
# Usage: bash run_ablation_tier2.sh [A|B|C|D|E|baseline]

set -e
cd /workspace/parameter-golf

ABLATION=${1:-baseline}

# Shared config
export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=256 DOC_ISOLATED_EVAL=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=4 VE_ENABLED=1
export VALUE_RESIDUAL=1 GATED_ATTENTION=1
export PERLAYER_TRAIN_LR=1 PROJ_LR_MULT=1.5 FC_LR_MULT=0.7
export STAR_RELU=1 TRIGRAM_HASH=1
export BIGRAM_HASH_BUCKETS=8192
export EMA_ENABLED=0 SWA=0 QAT=0
export TTT_ENABLED=0 TTT_CAUSAL=0
export TIER2_MODE=1
export SEED=1337

# Defaults (overridden per ablation)
export NUM_LAYERS=11
export NUM_KV_HEADS=4
export MLP_HIDDEN=0  # 0 = use mlp_mult * dim = 1536
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export GRAD_CLIP_NORM=0.0
export TRAIN_BATCH_TOKENS=786432

case "$ABLATION" in
    baseline)
        echo "=== TIER2: BASELINE (11L kv4 h1536) ==="
        ;;
    A)
        echo "=== TIER2: A — Full MHA (kv8) ==="
        export NUM_KV_HEADS=8
        ;;
    B)
        echo "=== TIER2: B — Wider MLP (h1792) ==="
        export MLP_HIDDEN=1792
        ;;
    C)
        echo "=== TIER2: C — Full MHA + Wider MLP (match PR #505) ==="
        export NUM_KV_HEADS=8
        export MLP_HIDDEN=1792
        ;;
    D)
        echo "=== TIER2: D — Higher lr + lower momentum (match PR #486) ==="
        export MATRIX_LR=0.04
        export SCALAR_LR=0.04
        export MUON_MOMENTUM=0.95
        export MUON_MOMENTUM_WARMUP_START=0.85
        export MUON_MOMENTUM_WARMUP_STEPS=500
        ;;
    E)
        echo "=== TIER2: E — 13 Layers ==="
        export NUM_LAYERS=13
        ;;
    *)
        echo "Usage: bash run_ablation_tier2.sh [A|B|C|D|E|baseline]"
        echo "  baseline: 11L kv4 h1536 (our current)"
        echo "  A: Full MHA (kv8)"
        echo "  B: Wider MLP (h1792)"
        echo "  C: Full MHA + Wider MLP (PR #505 arch)"
        echo "  D: Higher lr + lower momentum (PR #486 optimizer)"
        echo "  E: 13 Layers"
        exit 1
        ;;
esac

echo "NUM_LAYERS=$NUM_LAYERS NUM_KV_HEADS=$NUM_KV_HEADS MLP_HIDDEN=$MLP_HIDDEN"
echo "MATRIX_LR=$MATRIX_LR MUON_MOMENTUM=$MUON_MOMENTUM"
echo "================================================"

python3 records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
