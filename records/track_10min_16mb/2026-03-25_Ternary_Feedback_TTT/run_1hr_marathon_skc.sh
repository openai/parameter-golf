#!/bin/bash
# ============================================================================
# 1-Hour Marathon — SKC Convergence Study on Apple Silicon M3
# ============================================================================
# Purpose: Understand long-run convergence of SKC beyond our 10-min proxy runs.
#
# Key questions being answered:
#   1. Does BPB keep improving after 10 minutes or plateau?
#   2. Are there instabilities at 15/30/45/60 minutes?
#   3. What BPB does 12L dim=512 reach at 60 min?
#      → Extrapolates to 24L dim=512 on H100 (same width, 2× depth)
#   4. What is the convergence rate (BPB/hour)? Informs H100 time allocation.
#
# Config rationale:
#   • 12L dim=512: same width as H100 target, 50% depth → cleanest scale extrapolation
#   • seq=1024: key insight — memory is CONSTANT regardless of seq_len because
#     TRAIN_BATCH_TOKENS=8192 is fixed: 8 seqs×1024 uses identical activation memory
#     to 32 seqs×256 (201MB either way). Compute per step scales as O(T log T) via WHT
#     blocks: seq=1024 → 64 blocks/seq → ~4.7s/step → ~770 steps in 1hr.
#     seq=1024 chosen over seq=512 (1000 steps) because: 2× context teaches long-range
#     dependencies, better approximates competition seq=2048, still ~8 val checkpoints.
#     Over seq=2048 (515 steps): more gradient updates, slightly less risky scan loop.
#   • batch=8192: same as all ablations for fair comparison
#   • ALL proven best features (full EL combo + UNet caps skip always-on)
#   • Memory: ~1.1GB of M3's 16GB — very safe regardless of seq_len
#
# Expected outcomes:
#   • 8L_dim256@10min = 1.6552 BPB (best MLX validated)
#   • 12L_dim512@10min: ~1.60-1.65 BPB estimate (deeper + wider)
#   • 12L_dim512@60min: ~1.45-1.55 BPB estimate (if log-linear convergence holds)
# ============================================================================
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR"

TS=$(date +%s)
LOGFILE="${EXPDIR}/marathon_1hr_skc_${TS}.log"

echo "====== 1-HOUR SKC MARATHON START: $(date) ======"
echo "====== 1-HOUR SKC MARATHON START: $(date) ======" > "$LOGFILE"

env \
  ARCHITECTURE=skc \
  NUM_LAYERS=12 \
  MODEL_DIM=512 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=4 \
  MLP_MULT=4 \
  \
  SKC_BLOCK_SIZE=16 \
  SKC_NUM_CAPSULES=16 \
  SKC_CAPSULE_DIM=128 \
  SKC_CONV_KERNEL=4 \
  \
  SEED=42 \
  TRAIN_SEQ_LEN=1024 \
  TRAIN_BATCH_TOKENS=8192 \
  MAX_WALLCLOCK_SECONDS=3600 \
  ITERATIONS=1000000 \
  \
  CURRICULUM_ENABLED=1 \
  CURRICULUM_PHASE1_SEQ=64 \
  CURRICULUM_PHASE2_SEQ=256 \
  CURRICULUM_PHASE1_FRAC=0.03 \
  CURRICULUM_PHASE2_FRAC=0.12 \
  \
  TKO_ENABLED=0 \
  LAWA_ENABLED=1 \
  LAWA_K=5 \
  SWA_ENABLED=1 \
  SMEARGATE_ENABLED=1 \
  \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.015 \
  MUON_MOMENTUM=0.95 \
  MUON_MOMENTUM_WARMUP_START=0.85 \
  MUON_MOMENTUM_WARMUP_STEPS=0 \
  MUON_BACKEND_STEPS=5 \
  GRAD_CLIP_NORM=0.3 \
  WARMDOWN_FRACTION=0.25 \
  \
  BIGRAM_HASH_ENABLED=1 \
  BIGRAM_HASH_BUCKETS=1024 \
  BIGRAM_HASH_DIM=128 \
  ENGRAM_NUM_HEADS=4 \
  ENGRAM_NUM_ORDERS=3 \
  ENGRAM_INJECT_LAYER=1 \
  \
  NGRAM_CACHE_ENABLED=1 \
  NGRAM_MAX_ORDER=5 \
  NGRAM_ALPHA_BASE=0.05 \
  NGRAM_ALPHA_SCALE=0.55 \
  NGRAM_ENTROPY_CENTER=4.0 \
  \
  CAPSULE_ENABLED=0 \
  FEEDBACK_ENABLED=0 \
  KOOPMAN_SPECULATOR_ENABLED=0 \
  VRL_ENABLED=0 \
  TTT_ENABLED=0 \
  EMA_ENABLED=0 \
  SHARED_BLOCKS=0 \
  \
  SLIDING_EVAL=1 \
  SLIDING_EVAL_STRIDE=64 \
  VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=10 \
  TEMP_SCALING=0 \
  \
  TURBO_QUANT_EXPORT=1 \
  TURBO_QUANT_TRAIN=0 \
  \
  bash run_mlx_reasoner.sh 2>&1 | tee -a "$LOGFILE"

echo "====== 1-HOUR SKC MARATHON DONE: $(date) ======" | tee -a "$LOGFILE"
echo "Log: $LOGFILE"

# Extract convergence curve
/opt/homebrew/Cellar/python@3.12/3.12.3/Frameworks/Python.framework/Versions/3.12/bin/python3.12 << PYEOF
import re
with open("$LOGFILE") as f:
    c = f.read()

bpbs  = re.findall(r'val_bpb:([\d.]+)', c)
steps = re.findall(r'step:(\d+)/', c)
times = re.findall(r'train_time:(\d+)ms', c)

print("\n=== CONVERGENCE CURVE ===")
print(f"{'Step':>8} {'Time(min)':>10} {'val_bpb':>10}")
print("-" * 32)
for s, t, b in zip(steps, times, bpbs):
    print(f"{int(s):>8} {int(t)/60000:>10.1f} {float(b):>10.4f}")

if bpbs:
    print(f"\nFinal BPB: {float(bpbs[-1]):.4f}")
    print(f"Best  BPB: {min(float(b) for b in bpbs):.4f}")
    if len(bpbs) >= 2:
        first_bpb = float(bpbs[0])
        last_bpb  = float(bpbs[-1])
        print(f"Total improvement: {first_bpb - last_bpb:.4f} BPB over run")

# Compare to 10-min reference
print("\n=== REFERENCE POINTS ===")
print("  8L_dim256 @10min = 1.6552 BPB  (best MLX validated)")
print("  Expected 12L_dim512 @10min: see first few checkpoints above")
PYEOF
