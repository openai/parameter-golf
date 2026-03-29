#!/bin/bash
# Master ablation sequence. Run on 1×H100 for fast iteration.
set -euo pipefail

export NPROC=1
export MAX_WALLCLOCK_SECONDS=120
export SEED=1337

echo "=========================================="
echo "PARAMETER GOLF ABLATION SUITE"
echo "=========================================="

./run_ablation.sh \
  "p1_base_569" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q4" \
  "none" \
  NOTES="PR #569 base unmodified"

./run_ablation.sh \
  "p1_base_569_optrot" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569" \
  NOTES="+OptRot pre-quantization rotation" ENABLE_OPTROT=1

./run_ablation.sh \
  "p1_optrot_hybridnorm" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A15,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="+HybridNorm mixed Pre/Post" ENABLE_HYBRIDNORM=1

./run_ablation.sh \
  "p1_starelu_test" \
  "A1,A3,A4,A5,A6,A7,A8,A10,A11,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="Star-ReLU (GEPA) instead of LeakyReLU²" ACTIVATION=star_relu

./run_ablation.sh \
  "p1_gepa_gated_skips" \
  "A1,A4,A5,A6,A7,A8,A10,A11,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="GEPA gated skips + Star-ReLU (full GEPA)" ACTIVATION=star_relu SKIP_TYPE=gated

./run_ablation.sh \
  "p1_xsa_all" \
  "A1,A2,A3,A4,A5,A7,A8,A9,A10,A12,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="+XSA on all 11 layers" XSA_LAST_N=11

./run_ablation.sh \
  "p1_mlp35_int5" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A13,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,Q6,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="+MLP 3.5× (1792) with int5 quant" MLP_HIDDEN=1792 QUANT_BITS=5

./run_ablation.sh \
  "p1_mtp_heads" \
  "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T15,Q5,Q7,Q8,Q9,Q4" \
  "p1_base_569_optrot" \
  NOTES="+Multi-token prediction (2 aux heads)" MTP_NUM_HEADS=2

echo ""
echo "=========================================="
echo "PHASE 1 ABLATION RESULTS"
echo "=========================================="
python3 ablation.py leaderboard
python3 ablation.py ablations

echo ""
echo "DECISION POINT: Review results above."
echo "Pick the best combination of Phase 1 techniques."
echo "Then run Phase 2 (eval-time) on the winning checkpoint."
