#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Causal JEPA ablation sweep on 1×A100/H100 — 3 configs × 10 min each
#
# Google Colab setup:
#   !git clone -b trident-neural-memory-ttt https://github.com/<you>/parameter-golf.git
#   %cd parameter-golf
#   !pip install zstandard
#   !mkdir -p data/byte260_export logs
#   !cat > data/tokenizer_specs_byte260.json << 'SPEC'
#   [{"kind":"byte","name":"pure_byte_260","dataset_suffix":"byte260"}]
#   SPEC
#   !python data/download_hf_docs_and_tokenize.py \
#       --output-root data/byte260_export \
#       --tokenizer-config data/tokenizer_specs_byte260.json
#   !PYTHONUNBUFFERED=1 bash run_a100_causal_jepa.sh 2>&1 | tee logs/ablation_sweep.txt
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

export DATA_PATH="${DATA_PATH:-./data/byte260_export/datasets/fineweb10B_byte260}"

# ── shared defaults ──────────────────────────────────────────────────────────
set_defaults() {
    export MAX_WALLCLOCK_SECONDS=600
    export ITERATIONS=200000
    export WARMUP_STEPS=10
    export WARMDOWN_ITERS=200

    export TRAIN_SEQ_LEN=2048
    export TRAIN_BATCH_TOKENS=524288
    export GRAD_ACCUM_STEPS=4

    export VOCAB_SIZE=260
    export MODEL_DIM=512
    export NUM_LAYERS=12
    export NUM_HEADS=8
    export NUM_KV_HEADS=4
    export MLP_MULT=3
    export PARTIAL_ROPE_DIM=16
    export ROPE_BASE=10000
    export LOGIT_SOFTCAP=30.0
    export BIGRAM_VOCAB_SIZE=4096
    export BIGRAM_DIM=64

    export JEPA_WEIGHT=0.1
    export JEPA_LATENT_DIM=256
    export JEPA_HORIZON=32
    export JEPA_DECAY_FRAC=0.5
    export EMA_DECAY=0.997

    export EMBED_LR=0.6
    export MATRIX_LR=0.025
    export SCALAR_LR=0.04
    export MUON_MOMENTUM=0.99
    export MUON_MOMENTUM_WARMUP_START=0.92
    export MUON_MOMENTUM_WARMUP_STEPS=1500
    export GRAD_CLIP_NORM=0.3

    export INT6_ENABLED=1
    export USE_ZSTD=1
    export ZSTD_LEVEL=22

    export VAL_LOSS_EVERY=200
    export MAX_VAL_TOKENS=131072
    export EVAL_STRIDE=64
    export EVAL_BATCH_SEQS=16

    export TTT_ENABLED=1
    export TTT_LR=0.002
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_GRAD_CLIP=1.0
    export TTT_MOMENTUM=0.9

    export TRAIN_LOG_EVERY=20
}

# ── Ablation A: Gentle TTT ───────────────────────────────────────────────────
# Same JEPA decay, but much softer TTT to stop the overfitting regression
# TTT_LR 0.002→0.0005, TTT_EPOCHS 3→1
echo "================================================================"
echo "ABLATION A: Gentle TTT (lr=0.0005, epochs=1)"
echo "================================================================"
set_defaults
export RUN_ID=ablation_A_gentle_ttt
export TTT_LR=0.0005
export TTT_EPOCHS=1
python ./jepa/train_gpt.py

# ── Ablation B: Faster JEPA decay + Gentle TTT ──────────────────────────────
# JEPA off at 30% instead of 50% — more pure-CE training steps
# Combined with gentle TTT
echo "================================================================"
echo "ABLATION B: Fast JEPA decay (30%) + Gentle TTT"
echo "================================================================"
set_defaults
export RUN_ID=ablation_B_fast_jepa_decay
export JEPA_DECAY_FRAC=0.3
export TTT_LR=0.0005
export TTT_EPOCHS=1
python ./jepa/train_gpt.py

# ── Ablation C: Lower JEPA weight + Gentle TTT ──────────────────────────────
# Halve JEPA weight (0.05) to reduce gradient competition while keeping
# the regularization benefit. Standard 50% decay. Gentle TTT.
echo "================================================================"
echo "ABLATION C: Lower JEPA weight (0.05) + Gentle TTT"
echo "================================================================"
set_defaults
export RUN_ID=ablation_C_low_jepa_weight
export JEPA_WEIGHT=0.05
export TTT_LR=0.0005
export TTT_EPOCHS=1
python ./jepa/train_gpt.py

echo "================================================================"
echo "ALL ABLATIONS COMPLETE"
echo "Results in: logs/ablation_A_gentle_ttt.txt"
echo "            logs/ablation_B_fast_jepa_decay.txt"
echo "            logs/ablation_C_low_jepa_weight.txt"
echo "================================================================"
