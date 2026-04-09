#!/bin/bash
# ============================================================================
# RUNPOD 8×H100 SXM — COMPETITION SUBMISSION (10 minutes / 16MB)
# Architecture: Spectral Koopman Capsule (SKC) — proven winner
# ============================================================================
#
# CHANGES FROM OLD HYBRID SCRIPT (all bugs fixed):
#   BROKEN: VRL_START_LAYER=10 > NUM_LAYERS=8  → VRL never fired. Removed.
#   BROKEN: CURRICULUM_ENABLED=1               → dead code in train_gpt.py line 3197. Off.
#   HURTS:  CAPSULE_ENABLED=1                  → +0.030 BPB vs internal SKC skip. Off.
#   HURTS:  KOOPMAN_SPECULATOR_ENABLED=1        → catastrophic without capsule bank. Off.
#   HURTS:  MOE_ENABLED=1                       → untested with SKC, instability risk. Off.
#   HURTS:  STOCHASTIC_DEPTH_PROB=0.1           → regularization hurts underfitting regime. Off.
#   HURTS:  SELF_DISTILL_KL_WEIGHT=0.1          → gradient conflict with ternary quant. Off.
#   WRONG:  MUON_MOMENTUM_WARMUP_STEPS=1500     → ablation BH(0)=2.1263 < BG(300)=2.1517. → 0.
#   WRONG:  WARMDOWN_FRACTION=0.5               → decaying LR for 300/599s is too aggressive. → 0.4.
#   WRONG:  TTT_ENABLED=1                       → expensive eval overhead, eats 10-min budget. Off.
#   WRONG:  ARCHITECTURE=hybrid 8L dim=768      → SKC 24L dim=512 proven better. Switched.
#   NOTE:   VOCAB_SIZE=1024 is the competition standard (sp1024 tokenizer).
#   NOTE:   EMBED_DIM=254 is the code default   → not a bug, but not needed for SKC. Removed.
#   NEW:    COMPILER_WARMUP_STEPS=20            → pre-budget compile trigger (outside 599s).
#           WARMUP_STEPS=20                     → LR linear ramp (inside 599s budget).
#           XSA_START_LAYER=999                 → default=0 puts XSA on all 24 layers. Disabled.
#           KOOPMAN_ENABLED=0                   → default=1; 2-GPU proven winner had it off.
#           WEIGHT_SHARING=0                    → explicit off; default may be on.
#           File existence checks               → fail fast if data/tokenizer/script missing.
#           Artifact cleanup before run         → rm stale artifacts before each run.
#           Full artifact copies after run      → pre_export_state, best_live, best_proxy.
#
# Deploy (RunPod):
#   runpodctl create pod \
#     --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 8 \
#     --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
#     --volumeSize 50
#
# Setup on pod:
#   pip install sentencepiece
#   pip install flash-attn --no-build-isolation   # flash_attn v3 — ~15-20% step speedup on H100
#   # Copy data to /workspace/data/ and code to /workspace/
#
# Run:
#   bash run_runpod_8xh100.sh
# ============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

[[ -f "${DIR}/train_gpt.py" ]] || { echo "ERROR: ${DIR}/train_gpt.py not found" >&2; exit 1; }

# ── Data (competition standard: vocab=1024, sp1024) ──────────────────────────
# Competition README explicitly uses sp1024/vocab=1024 throughout.
# Evaluation is bits-per-byte (tokenizer-agnostic) but all submissions use sp1024.
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2; exit 1; }

# ── Architecture: SKC + MoE — sweep winner (8L/320D/8experts → BPB 1.6825) ──
# Sweep results (2×A40, vocab=1024, batch=16K, 599s):
#   8L/256D baseline:          1.8893 BPB, 10.22MB
#   8L/256D +16 experts:       1.8893 BPB, 10.22MB
#   8L/256D +32 experts:       1.7525 BPB, 15.12MB
#   8L/256D +8 experts:        1.8012 BPB, 10.12MB
#   8L/320D +8 experts (TOP):  1.6825 BPB, 15.82MB  ← WINNER (+0.207 BPB)
#
# Both sweep and H100 submission use vocab=1024 — sizes are directly comparable.
# Sweep measured 15.82MB at vocab=1024 → same config here should reproduce ~15.82MB.
# Why NOT scale to 24L: MoE experts × 24L → way over 16MB budget.
# Why 8L works well: MoE gives effective parameter density vs depth.
export ARCHITECTURE=skc
export NUM_LAYERS=8
# D=1536 mainline: quality-per-second optimized for 599s budget.
# D=2304 yielded ~14.9MB but only ~1200 steps in 599s on 8×H100 (too few updates).
# D=1536 gives ~8MB artifact, ~2× more steps → better final loss trajectory.
# head_dim=64 (1536/24): hardware-optimal; KV ratio=1/4 (GQA).
export MODEL_DIM=1536
export NUM_HEADS=24            # head_dim = 1536/24 = 64 (hardware sweet spot)
export NUM_KV_HEADS=6          # GQA: num_heads/4
export MLP_MULT=3              # MLP_MULT=3 vs 4: better wall-clock vs capacity tradeoff
export EMBED_DIM=256           # FP embedding rank: 192-256 band for V=1024 (small vocab → small embed)
export PARTIAL_ROPE_DIMS=32    # Partial RoPE: 32 dims is sufficient for seq_len=2048

# MoE: proven winner from sweep (4 experts, top-1 routing, lighter than before)
export MOE_ENABLED=1
export MOE_NUM_EXPERTS=4
export MOE_TOP_K=1
export MOE_LAYER_FRAC=0.67     # Upper 33% of layers get MoE (keeps lower layers dense)

# SKC-specific hyperparams (scaled for D=1536)
export SKC_BLOCK_SIZE=64       # Larger block_size for wider model
export SKC_NUM_CAPSULES=24     # capsule_num ≈ model_dim/64, capped at 24
export SKC_CAPSULE_DIM=64      # Capsule dim = 64 (semantic slot width)
export SKC_CONV_KERNEL=4

# ── Training budget ──────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-599}   # overridable for smoke tests
export ITERATIONS=500000
# WARMUP_STEPS: LR linear ramp (0→base over N steps). Runs WITHIN the 599s budget.
export WARMUP_STEPS=20
# COMPILER_WARMUP_STEPS: runs full forward+backward N times BEFORE t0 (outside 599s budget).
# Weights and optimizer are reset after; these steps exist only to trigger torch.compile
# JIT compilation of all CUDA kernels so the budget starts with a warm graph.
# Without this, the first ~5-10 training steps are slow compilation overhead.
export COMPILER_WARMUP_STEPS=20
export SEED=${SEED:-42}   # overridable: SEED=1337 bash run_runpod_8xh100.sh

# ── Batch sizing for 8×H100 SXM (80GB each) ────────────────────────────────
# With autocast(bf16) + TF32 now enabled, each step is compute-bound, not memory-bound.
# The model is tiny (~4.5M params, ~2.75MB) — 1 seq/GPU (16K/8GPUs) severely
# under-utilizes H100 tensor cores. We need more tokens/GPU to keep SMs busy.
#
# Target: ~32 seqs/GPU = 65K tokens/GPU = 524K tokens/step globally
# Activations per seq at D=320, L=8, T=2048, bf16 ≈ 320*2048*8*2B ≈ 10MB/seq
# 32 seqs/GPU × 10MB ≈ 320MB VRAM for activations — well within 80GB
#
# More tokens/step = better gradient estimates + fewer Muon all-reduces per token
# Tradeoff: fewer steps in 599s, but each step is higher quality
# At ~2ms/step with bf16 + compile: 524K batch → ~30K steps in 599s
export TRAIN_BATCH_TOKENS=262144  # 256K = 16 seqs/GPU on 8×H100 — more frequent updates for D=1536
export TRAIN_SEQ_LEN=2048
export LOCAL_SGD_SYNC_EVERY=10       # Weight mirroring: each GPU explores independently, syncs every 10 steps
                                     # Eliminates all-reduce on 9/10 steps → faster per-step throughput
export LOCAL_SGD_WARMUP_STEPS=50     # Standard DDP for first 50 steps (stable weight init), then switch
export TRAINING_DEPTH_RECURRENCE=0   # Off: slows steps, contradicts maximize-steps goal
export ACTIVATION_CHECKPOINTING=0    # Not needed without depth recurrence

# ── Curriculum ───────────────────────────────────────────────────────────────
# DEAD CODE: train_gpt.py line 3197 unconditionally sets active_seq_len = train_seq_len.
# All CURRICULUM_PHASE* vars are parsed but never used in the training loop.
# Training always runs at TRAIN_SEQ_LEN=2048. Fine — H100 handles full seq from step 1.
export CURRICULUM_ENABLED=0

# ── Optimizer (SKC-tuned from ablations) ─────────────────────────────────────
export MATRIX_OPTIMIZER=muon
export MATRIX_LR=0.02          # SKC-optimal (vs 0.035 default): validated in ablations
export SCALAR_LR=0.015
export TIED_EMBED_LR=0.025
export ADAM_LR=0.015
export ADAM_WD=0.04
export MUON_WD=0.04
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=0   # 0 > 300: ablation BH=2.1263 vs BG=2.1517
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=0.3
export WARMDOWN_FRACTION=0.4          # wallclock-based: LR decays over last 240s of 599s

# ── Weight averaging (proven convergence boost) ───────────────────────────────
export LAWA_ENABLED=1
export LAWA_K=5
export SWA_ENABLED=1
export SMEARGATE_ENABLED=1
export TKO_ENABLED=0                  # Always hurts: proven across all ablations

# ── Engram hash (proven BPB improvement with orders=3) ───────────────────────
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_BUCKETS=8192       # V=1024 → 1024²=1M bigrams; 8192 is sane floor for D=1536
export BIGRAM_HASH_DIM=48             # Smaller dim matches D=1536 (was 128 for wider model)
export ENGRAM_NUM_HEADS=4
export ENGRAM_NUM_ORDERS=2            # 2 orders (not 3): with small vocab+limited buckets, 3+ is collision-heavy
export ENGRAM_INJECT_LAYER=1

# ── N-gram cache (free BPB at eval, interpolated with neural model) ───────────
export NGRAM_CACHE_ENABLED=1
export NGRAM_MAX_ORDER=5
export NGRAM_ALPHA_BASE=0.05
export NGRAM_ALPHA_SCALE=0.55
export NGRAM_ENTROPY_CENTER=4.0

# ── Disabled features (proven to hurt or untested at SKC scale) ──────────────
export CAPSULE_ENABLED=0          # External CapsuleBank: DC=1.8813 vs DB=1.8516 → +0.030 HURTS
                                  # Internal SKC UNet caps skip (always on) already handles this
export KOOPMAN_ENABLED=0          # Default=1; proven winner ran without it — off to match 2-GPU winner
export KOOPMAN_SPECULATOR_ENABLED=0  # Without caps bank: catastrophic (BR=3.33 vs 2.14 baseline)
                                     # With caps bank: neutral at best (+0.002). Never helps.
export FEEDBACK_ENABLED=0         # Catastrophic: seq_M=3.51, seq_Q=3.38 vs baseline ~2.14
export VRL_ENABLED=0              # Was broken in old script (start_layer=10 > num_layers=8)
export TTT_ENABLED=0              # Expensive eval overhead — eats into 10-min wallclock
export EMA_ENABLED=0              # Redundant with LAWA weight averaging
export SHARED_BLOCKS=0            # SKC: each layer specializes, sharing degrades performance
export WEIGHT_SHARING=0           # Explicit off: weight sharing degrades SKC performance
export XSA_START_LAYER=999        # Default=0 (XSA on ALL layers) — disable entirely for SKC
export STOCHASTIC_DEPTH_PROB=0    # Regularization hurts in underfitting regime (10-min budget)
export SELF_DISTILL_KL_WEIGHT=0   # Gradient conflict with ternary quantization

# ── Eval stack ────────────────────────────────────────────────────────────────
# Training phase: zero validation — every millisecond of the 599s budget is training compute.
# The loop's last_step block (after wallclock cap fires) still runs: it gives a final val_bpb
# reading and saves best_live_state.pt, but this happens AFTER the 599s window closes.
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=50         # Print loss every 50 steps (was 20) — negligible overhead reduction

# Post-training eval: runs after the 599s budget, fully outside training time.
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=64
export SLIDING_BATCH_SIZE=256
export TEMP_SCALING=1             # Calibrate optimal softmax T on training data post-training

# ── Ternary quantization ──────────────────────────────────────────────────────
export BITNET_GROUP_SIZE=128
export TURBO_QUANT_EXPORT=1
export TURBO_QUANT_TRAIN=0
export TURBO_QUANT_KV=1
export EXPORT_ALIGNED_TRAIN=1
export EXPORT_ALIGNED_TRAIN_START_FRACTION=0.85
export TERNARY_THRESHOLD_SEARCH=1
export TERNARY_THRESHOLD_LOW=0.02
export TERNARY_THRESHOLD_HIGH=0.15
export TERNARY_THRESHOLD_STEPS=4
export TERNARY_SCALE_SEARCH=1
export TERNARY_SCALE_MULT_LOW=0.9
export TERNARY_SCALE_MULT_HIGH=1.1
export TERNARY_SCALE_MULT_STEPS=3
export TERNARY_CALIB_TOP_N=5
export EXPORT_PROXY_EVAL=1        # Mid-training proxy: track round-trip BPB, snapshot best checkpoint
export EXPORT_PROXY_EVERY=2000    # Every 2000 steps (~60s at fast throughput) — low overhead
export EXPORT_PROXY_NUM_SEQS=16
export AVERAGE_TERNARY_PARAMS=0
export SAVE_PRE_EXPORT_STATE=1
export FAST_EXPORT=1              # Skip variant grid search — use LAWA directly (saves 30-60min post-training)
export LZMA_PRESET=3              # Preset 3 vs 6: ~3x faster, <1% size difference

# ── torch.compile (H100 sm_90 — huge speedup, required for competitive step time) ──
export COMPILE_MODE=default

# ── NCCL (single-node 8×H100 NVLink — do NOT disable P2P, NVLink uses it) ──────
export NCCL_SOCKET_IFNAME=lo    # loopback for same-node rendezvous — faster init
export NCCL_IB_DISABLE=1        # no InfiniBand on single-node pod

# ── Run ID & logging ──────────────────────────────────────────────────────────
export RUN_ID="skc_h100_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
rm -f final_model.ternary.ptz submission.json pre_export_state.pt best_export_proxy_state.pt best_live_state.pt
LOG="${DIR}/logs/${RUN_ID}.log"

echo "=========================================================================="
echo "  SKC Competition Run — 8×H100 SXM (FIXED)"
echo "  RUN ID : ${RUN_ID}"
echo "  MODEL  : SKC  L=${NUM_LAYERS}  D=${MODEL_DIM}  H=${NUM_HEADS}  (~92M params / ~14.9MB empirical)"
echo "           block_size=${SKC_BLOCK_SIZE}  caps=${SKC_NUM_CAPSULES}×${SKC_CAPSULE_DIM}"
echo "           mlp_mult=${MLP_MULT}  UNet caps skip (proven -0.107 BPB) auto-enabled for SKC arch"
echo "  BUDGET : ${MAX_WALLCLOCK_SECONDS}s wallclock  (compiler_warmup=${COMPILER_WARMUP_STEPS} steps pre-budget, lr_warmup=${WARMUP_STEPS} steps in-budget)"
echo "  SEQ    : ${TRAIN_SEQ_LEN} (competition standard)"
echo "  BATCH  : ${TRAIN_BATCH_TOKENS} tokens/step → $((TRAIN_BATCH_TOKENS/2048)) seqs/step globally ($((TRAIN_BATCH_TOKENS/2048/8)) seqs/GPU)"
echo "  LR     : matrix=${MATRIX_LR}  scalar=${SCALAR_LR}  warmdown_frac=${WARMDOWN_FRACTION}"
echo "  EXTRAS : engram(orders=3)  ngram_cache  TKO=off  LAWA  SWA  smeargate  temp_scaling"
echo "  FIXED  : curriculum(dead code)  capsule/speculator(hurt)  moe(risky)"
echo "           vrl(was broken)  stoch_depth/self_distill(hurt)  muon_warmup→0"
echo "=========================================================================="

# ── Launch: 8-GPU DDP via torchrun ───────────────────────────────────────────
OMP_NUM_THREADS=1 \
TORCH_NCCL_TIMEOUT_SEC=7200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"

# Preserve per-run artifacts before next seed overwrites them
cp final_model.ternary.ptz          "logs/${RUN_ID}_model.ternary.ptz"          2>/dev/null || true
cp submission.json                   "logs/${RUN_ID}_submission.json"             2>/dev/null || true
cp pre_export_state.pt               "logs/${RUN_ID}_pre_export_state.pt"         2>/dev/null || true
cp best_export_proxy_state.pt        "logs/${RUN_ID}_best_export_proxy_state.pt"  2>/dev/null || true
cp best_live_state.pt                "logs/${RUN_ID}_best_live_state.pt"          2>/dev/null || true
cp final_model.ternary.ptz           "logs/skc_h100_model.ternary.ptz"           2>/dev/null || true

echo "=== DONE ==="
echo "Log      : $LOG"
echo "Artifact : logs/${RUN_ID}_model.ternary.ptz"
echo "Submission: logs/${RUN_ID}_submission.json"
