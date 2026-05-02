#!/bin/bash
# ============================================================
# exp106: meta-TTT = cross-chunk split + delta loss + MetaSGD scales
# Branched from exp101_poscond-bigram-trigram_from_exp95 (1.1159 TRIGRAM=0 baseline).
#
# Goal: test whether a re-formulated meta-TTT produces a *differentiated*
# adaptation advantage where the exp101 FOMAML flavor plateaued at the same
# ~0.023 bpb TTT delta as no-meta training.
#
# Three changes, all inside train_gpt.py's meta_ttt_step (no arch change):
#   (A) META_TTT_SPLIT=batch  — cross-sample inner/outer split.
#       Inner/outer draw from DIFFERENT sequences in the same batch, so they
#       come from different fineweb10B documents. Matches deployment-time TTT
#       statistical regime instead of the legacy "same-doc prefix/suffix" split
#       whose inner/outer correlation was too high to produce real meta signal.
#   (B) META_TTT_DELTA_WEIGHT=0.3 — outer loss = (post_w + delta_w) * loss_post
#       - delta_w * loss_pre. Actively rewards the backbone for developing
#       features where SGD-on-banks has headroom to move (loss_pre > loss_post).
#       Main training loss keeps loss_pre grounded; delta term widens the gap.
#   (C) META_SGD_ENABLED=1  — learn per-layer-per-bank inner-loop LR scales
#       (meta_sgd_{qo,kv,up,down}, ~6*num_layers total scalars, ~66 params).
#       Excluded from final_model.pt so they don't touch the 16MB budget.
#       Inner update becomes upd = bank.detach() - lr * scale * g. Built as a
#       differentiable non-leaf so a single backward populates both the
#       MetaSGD scale grads (via leaf autograd) and the FOMAML bank grads
#       (via retain_grad + manual copy to bank.grad).
#
# Single diff vs exp101-no-tri is these three env vars + the new train_gpt.py
# logic. Also keeps TRIGRAM=0 so the baseline matches the 1.1159 point the
# ablation run (exp105a) is being compared against.
#
# Decision thresholds (vs exp105a's no-meta-TTT baseline, not yet run):
#   > 0.002 bpb improvement over exp105a      -> meta-TTT genuinely helps; keep
#   [0, 0.002 bpb] over exp105a               -> marginal, not worth compute
#   <= exp105a                                -> meta reformulation ALSO fails;
#                                                  pivot to hypernet banks / prompt-vector TTT
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="exp106_metasgd-crosschunk-delta_from_exp101"
cd /workspace/parameter-golf

# --- 8xH100 simulation ---
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-4800}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export ITERATIONS="${ITERATIONS:-7500}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-2500}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

# --- Eval ---
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=128
export SEED="${SEED:-42}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-3000}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"

# --- Architecture ---
export NUM_LAYERS=11
export XSA_LAST_N=11
export ROPE_DIMS=16
export LN_SCALE=1

# --- Smaller bigram (saves ~1.5 MB → eliminates ±1 pruning) ---
export BIGRAM_VOCAB_SIZE=4096
export BIGRAM_DIM=64

# --- Bigram layout (inherited from exp101 with TRIGRAM=0 to match 1.1159 ref) ---
# POS_CONDITIONAL_BIGRAM=1: split buckets ws/non-ws (see BigramHashEmbedding docstring)
# TRIGRAM=0: exp101-no-tri baseline, same as exp105a so results are directly comparable
export POS_CONDITIONAL_BIGRAM=1
export TRIGRAM=0

# --- Wider Value Embeddings (layers 7-10, was 9-10) ---
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="7,8,9,10"

# --- Earlier Late QAT (threshold 0.25, was 0.15) ---
export QAT_ENABLED=0
export LATE_QAT_THRESHOLD=0.25

# --- Adaptive Warmdown ---
export ADAPTIVE_WARMDOWN=1
export ADAPTIVE_WARMDOWN_EMA=0.99
export ADAPTIVE_WARMDOWN_THRESHOLD=0.0005
export ADAPTIVE_WARMDOWN_MIN_STEPS=2000

# --- Learning rates ---
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035

# --- Weight decay ---
export MUON_WD=0.04
export ADAM_WD=0.04

# --- EMA (tighter focus on converged weights) ---
export EMA_ENABLED=1
export EMA_DECAY=0.998
export EMA_UPDATE_EVERY=10

# --- SWA ---
export SWA_ENABLED=1
export SWA_EVERY=50

# --- Fixed momentum 0.99 (meta-TTT needs stable high momentum) ---
# Cycling would dilute the weak FOMAML gradient signal (3x faster forgetting at 0.97)
export MOMENTUM_CYCLIC=0
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500

# --- Newton-Schulz ---
export MUON_BACKEND_STEPS=5

# --- Grad clipping ---
export GRAD_CLIP_NORM=0.3

# --- GPTQ ---
export GPTQ_CALIB_BATCHES=256
export GPTQ_BLOCK_SIZE=128
export TARGET_MB=15.9

# --- Meta-TTT (FOMAML + exp106 A/B/C extensions) ---
# Base FOMAML (unchanged from exp101)
export META_TTT_ENABLED=1
export META_TTT_INNER_LR=0.002
export META_TTT_EVERY=4
export META_TTT_LOSS_WEIGHT=0.5
export META_TTT_FREEZE_BLOCKS=2
# (A) Cross-chunk split: "batch" = inner/outer from different sequences (different docs).
#     Falls back to seq-half split if batch size < 2.
export META_TTT_SPLIT=batch
# (B) Delta-loss weight. outer = (post_w + delta_w) * loss_post - delta_w * loss_pre.
#     0.3 is a moderate setting — strong enough to shape the backbone without fighting
#     the main loss. Bump to 0.5 if delta stays flat; reduce to 0.1 if pre-loss drifts up.
export META_TTT_DELTA_WEIGHT=0.3
# (C) MetaSGD learned per-layer inner-loop LR scales. ~66 params, excluded from export.
export META_SGD_ENABLED=1
export META_SGD_LR=0.0

# --- TTT (eval time) — AdamW, flat LR, larger chunks ---
export TTT_ENABLED=1
export TTT_LR=0.004
export TTT_EPOCHS=4
export TTT_CHUNK_TOKENS=65536
export TTT_FREEZE_BLOCKS=2
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=16
export TTT_GRAD_CLIP=1.0

export RUN_ID="${EXP_NAME}_seed${SEED}"
echo "=== ${EXP_NAME} seed=${SEED} ==="
echo "=== exp106: cross-chunk split (A) + delta loss (B) + MetaSGD scales (C) ==="
echo "=== META_TTT_SPLIT=${META_TTT_SPLIT} DELTA_WEIGHT=${META_TTT_DELTA_WEIGHT} META_SGD=${META_SGD_ENABLED} ==="
python3 "${SCRIPT_DIR}/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/logs_seed${SEED}.txt"
echo "=== ${EXP_NAME} COMPLETE ==="
