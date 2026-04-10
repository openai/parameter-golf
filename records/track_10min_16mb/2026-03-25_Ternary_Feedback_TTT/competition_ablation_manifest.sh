#!/bin/bash
# ============================================================================
# COMPETITION ABLATION MANIFEST — SKC SP8192
# Branch: skc_competition_sp8192
# ============================================================================
# This script documents the mandatory ablation matrix.
# It does NOT launch runs automatically — it is a reference for
# what to run and what metric to record for each ablation.
#
# Run each ablation using the 2-GPU proxy script and record:
#   - steps completed in 599s budget
#   - final roundtrip BPB (canonical metric)
#   - augmented BPB (with TTT + ngram)
#   - artifact size (MB)
#   - any NaN / divergence events
#
# Promote a change only if it improves final roundtrip BPB under
# the real artifact + timing constraints.
# ============================================================================

# ── Ablation 1: Tokenizer regime ─────────────────────────────────────────────
# Compare SP1024 (research baseline) vs SP8192 (competition target)
# Everything else held at research defaults except tokenizer/data.
#
# SP1024 baseline run (use existing run_small_skc_2gpu.sh):
#   DATA_PATH=.../fineweb10B_sp1024
#   TOKENIZER_PATH=.../fineweb_1024_bpe.model  VOCAB_SIZE=1024
#
# SP8192 run (use run_skc_competition_2gpu_proxy.sh with TTT_ENABLED=0,
#   SKC_PARALLEL_RESIDUAL=0, RECURRENCE_DEPTH=0):
#   DATA_PATH=.../fineweb10B_sp8192
#   TOKENIZER_PATH=.../fineweb_8192_bpe.model  VOCAB_SIZE=8192
ABLATION_1_BASELINE_BPB=""   # fill in after run
ABLATION_1_SP8192_BPB=""

# ── Ablation 2: SKC serial vs SKC recurrence ──────────────────────────────────
# SP8192, no parallel residual, vary recurrence
# RECURRENCE_DEPTH=0 (serial) vs RECURRENCE_DEPTH=2 RECURRENCE_START_FRACTION=0.35
ABLATION_2_SERIAL_BPB=""
ABLATION_2_RECUR_BPB=""

# ── Ablation 3: SKC serial vs SKC parallel residual ───────────────────────────
# SP8192, no recurrence, vary parallel residual
# SKC_PARALLEL_RESIDUAL=0 vs SKC_PARALLEL_RESIDUAL=1
ABLATION_3_SERIAL_BPB=""
ABLATION_3_PARALLEL_BPB=""

# ── Ablation 4: MoE on vs off ─────────────────────────────────────────────────
# SP8192, no recurrence, no parallel, vary MoE
# MOE_ENABLED=0 vs MOE_ENABLED=1
ABLATION_4_NOMOE_BPB=""
ABLATION_4_MOE_BPB=""

# ── Ablation 5: Export path ───────────────────────────────────────────────────
# EXPORT_MODE=ternary_lzma vs EXPORT_MODE=competition_gptq
# Both on same trained model (can reuse checkpoint from ablation 1/SP8192 run)
ABLATION_5_TERNARY_BPB=""
ABLATION_5_TERNARY_SIZE_MB=""
ABLATION_5_GPTQ_BPB=""
ABLATION_5_GPTQ_SIZE_MB=""

# ── Ablation 6: TTT off vs legal score-first TTT ─────────────────────────────
# TTT_ENABLED=0 vs TTT_ENABLED=1 TTT_SCOPE=skc_safe
# (augmented BPB is the relevant metric here, not roundtrip)
ABLATION_6_NOTTT_AUGMENTED_BPB=""
ABLATION_6_TTT_AUGMENTED_BPB=""
ABLATION_6_TTT_WALL_CLOCK_OVERHEAD_S=""

# ── Ablation 7: Research HP vs frontier HP ────────────────────────────────────
# Research: MATRIX_LR=0.02 MUON_WD=0.04 QK_GAIN_INIT=2.25 WARMDOWN_FRACTION=0.20
# Frontier: MATRIX_LR=0.025 MUON_WD=0.08 QK_GAIN_INIT=3.0 WARMDOWN_FRACTION=0.15
ABLATION_7_RESEARCH_BPB=""
ABLATION_7_FRONTIER_BPB=""

# ── Sweep grid for HP search ──────────────────────────────────────────────────
# Run in this order (most impactful first). Fix best value, then sweep next.
# 1. QK_GAIN_INIT: 2.25 2.5 3.0 3.5
# 2. MUON_WD: 0.04 0.06 0.08 0.10
# 3. WARMDOWN_FRACTION: 0.10 0.15 0.20 0.25
# 4. MATRIX_LR: 0.018 0.022 0.025 0.030
# 5. MoE on/off (after best HP found)
# 6. RECURRENCE_START_FRACTION: 0.25 0.30 0.35 0.40

echo "Ablation manifest loaded. Fill in BPB values as runs complete."
echo "Promote to best_known_competition.env only when roundtrip BPB improves."
