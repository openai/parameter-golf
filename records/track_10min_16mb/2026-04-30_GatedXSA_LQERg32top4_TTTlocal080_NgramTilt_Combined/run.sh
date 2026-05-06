#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Gated XSA + CaseOps + LQER g32/top4 + In-Timer N-gram Tilt
#
# Architecture & code: PR #2018 (simon-marcus) — Gated XSA + LQER + token-only
#   in-timer N-gram tilt + LeakyReLU 0.3, V21 lineage from PR #1967.
#   Verified pre-quant 1.05028, post-TTT 1.04722 BPB (3-seed mean).
#
# Quant tuning ported from PR #2060 (S0urC10ud):
#   MATRIX_LR=0.028, LQER_RANK=2, LQER_ASYM_GROUP=32, LQER_TOP_K=4.
#
# IMPORTANT — knobs from PR #2060 that DO NOT EXIST in PR #2018's train_gpt.py
# and would be silently ignored have been REMOVED from this script:
#   TTT_MASK, TTT_Q_LORA, TTT_V_LORA, TTT_LOCAL_LR_MULT,
#   PHASED_TTT_ENABLED, LQER_GROUP_SIZE.
# These were also present in PR #2018's own run.sh but had no effect there
# either; PR #2018's 1.04722 was achieved with Q+V LoRA enabled by default
# and TTT_LORA_LR=0.0001 (no local multiplier).
#
# Included final g32/top4 logs:
#   seed 42: 1.05872644 BPB
#   seed 0:  1.06035228 BPB
#   mean:    1.05953936 BPB
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-combined_gxsa_lqerg32top4_ngram_seed${SEED}}"

# ── Data / tokenizer (CaseOps SP8192 lossless-caps) ──────────────────────────
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${SCRIPT_DIR}/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"

# ── Compute budget ───────────────────────────────────────────────────────────
ITERATIONS=20000
MAX_WALLCLOCK_SECONDS=600
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# ─── PORTED FROM PR #2060 (the 4 knobs PR #2018's code actually honors) ──────
# These are the LQER quantization tweaks that PR #2060 used on its V21/LongCtx
# parent (PR #2007), giving −0.00107 BPB at p≈0.011.
MATRIX_LR=0.028                # parent (PR #2018): 0.026 → 0.028
LQER_RANK=2                    # parent (PR #2018): 4    → 2  (half-rank, ~half size per tensor)
LQER_ASYM_GROUP=32             # parent (PR #2018): 64   → 32 (finer asym groups)
LQER_TOP_K=4                   # parent (PR #2018): 1    → 4  (more correctors; net bytes ≈ same)

# ── PR #2018 architecture toggles ────────────────────────────────────────────
GATED_XSA=1                    # learned per-head tanh(α) gate on XSA, zero-init
SKYLIGHT_MUON=0                # destabilized this stack per PR #2018 ablation

# ── PR #2018 in-timer n-gram tilt (causal token-16 expert only, C1-compliant) ─
NGRAM_TILT_ENABLED=1
NGRAM_HINT_PRECOMPUTE_OUTSIDE=0  # MUST be 0 — A2 merged-record precedent
TOKEN_ORDER=16
TOKEN_THRESHOLD=0.800
TOKEN_BOOST=2.625
WITHIN_TAU=999 WITHIN_BOOST=0    # within-word expert disabled (C1 violation in PR #1967)
WORD_TAU=999   WORD_BOOST=0      # word-level   expert disabled (C1 violation in PR #1967)
AGREE_ADD_BOOST=0

# ── TTT (PR #2018 phased-1 / 1000-prefix; needed to fit n-gram in 600s eval) ─
TTT_ENABLED=1
PHASED_TTT_NUM_PHASES=1          # phased TTT is auto-on whenever NUM_PHASES > 0
PHASED_TTT_PREFIX_DOCS=1000
TTT_LORA_RANK=80
EVAL_SEQ_LEN=2560
TTT_EVAL_SEQ_LEN=2560

# ── Architecture knobs (V21 + Gated XSA stack) ───────────────────────────────
QK_GAIN_INIT=5.25
MIN_LR=0.1
EMBED_BITS=7
GRAD_CLIP_NORM=0.3
MATRIX_CLIP_SIGMAS=12.85
ATTN_CLIP_SIGMAS=13.0
MLP_CLIP_SIGMAS=11.5
EMBED_CLIP_SIGMAS=14.0
FUSED_CE_ENABLED=1
SMEAR_GATE_ENABLED=1
GATE_WINDOW=12
SPARSE_ATTN_GATE_ENABLED=1
LQER_ENABLED=1
LQER_ASYM_ENABLED=1
AWQ_LITE_ENABLED=1
ASYM_LOGIT_RESCALE=1
GPTQ_RESERVE_SECONDS=4.0
GPTQ_CALIBRATION_BATCHES=16
COMPRESSOR=pergroup
CASEOPS_ENABLED=1
VOCAB_SIZE=8192

# ─────────────────────────────────────────────────────────────────────────────
export SEED RUN_ID DATA_PATH TOKENIZER_PATH \
       ITERATIONS MAX_WALLCLOCK_SECONDS \
       MATRIX_LR LQER_RANK LQER_ASYM_GROUP LQER_TOP_K \
       GATED_XSA SKYLIGHT_MUON \
       NGRAM_TILT_ENABLED NGRAM_HINT_PRECOMPUTE_OUTSIDE \
       TOKEN_ORDER TOKEN_THRESHOLD TOKEN_BOOST \
       WITHIN_TAU WITHIN_BOOST WORD_TAU WORD_BOOST AGREE_ADD_BOOST \
       TTT_ENABLED PHASED_TTT_NUM_PHASES PHASED_TTT_PREFIX_DOCS \
       TTT_LORA_RANK EVAL_SEQ_LEN TTT_EVAL_SEQ_LEN \
       QK_GAIN_INIT MIN_LR EMBED_BITS GRAD_CLIP_NORM \
       MATRIX_CLIP_SIGMAS ATTN_CLIP_SIGMAS MLP_CLIP_SIGMAS EMBED_CLIP_SIGMAS \
       FUSED_CE_ENABLED SMEAR_GATE_ENABLED GATE_WINDOW SPARSE_ATTN_GATE_ENABLED \
       LQER_ENABLED LQER_ASYM_ENABLED \
       AWQ_LITE_ENABLED ASYM_LOGIT_RESCALE \
       GPTQ_RESERVE_SECONDS GPTQ_CALIBRATION_BATCHES \
       COMPRESSOR CASEOPS_ENABLED VOCAB_SIZE

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/train_gpt.py"
