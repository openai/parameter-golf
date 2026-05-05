#!/bin/bash
# A/B Test Round 3 — all experiments sequential
# Each full training run takes ~20 min (10 min train + 10 min eval)
# Eval-only runs take ~10 min
# Total: ~3-4 hours for everything

export PATH=/data/backups/rganapa/pylibs/bin:$PATH
export TMPDIR=/data/backups/rganapa/tmp
export PYTHONPATH=/data/backups/rganapa/pylibs
export TRITON_CACHE_DIR=/data/backups/rganapa/triton_cache
export TORCH_HOME=/data/backups/rganapa/torch_home
export WANDB_DIR=/data/backups/rganapa
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export DATA_PATH=/data/backups/rganapa/parameter-golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/data/backups/rganapa/parameter-golf/data/tokenizers/fineweb_1024_bpe.model

cd /data/backups/rganapa/parameter-golf

# Common training config (matches exp199/exp200 baseline)
COMMON_TRAIN="NUM_LAYERS=14 MUON_WD=0.09 ADAM_WD=0.02 EMA_ENABLED=1 EMA_DECAY=0.997 \
BIGRAM_VOCAB_SIZE=8192 BIGRAM_DIM=64 MLP_ACTIVATION=leaky2 \
PRUNE_FRAC=0.0 ROPE_BASE=50000 SWA_ENABLED=0 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
GPTQ_ENABLED=1 GPTQ_SAMPLES=256 SEED=1337"

# Common TTT config (stride=76 for time budget)
COMMON_TTT="TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 \
TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 EVAL_STRIDE=76"

# Eval-only config (uses saved model)
EVAL_ONLY="EVAL_ONLY_MODEL=/data/backups/rganapa/parameter-golf/final_model_pre_ttt.pt"

echo "========================================="
echo "A/B TEST ROUND 3 — $(date)"
echo "========================================="

# --- EVAL-ONLY EXPERIMENTS (use saved model, ~10 min each) ---

# E1. Cosine-annealed TTT LR (vs flat lr=0.002)
# NOT IMPLEMENTED YET IN 201 — skip for now
# echo "--- E1: cosine TTT LR ---"

# E2. Per-layer TTT LR (later layers get higher LR)
# NOT IMPLEMENTED YET — skip for now

# --- FULL TRAINING EXPERIMENTS (~20 min each) ---

# T1. VRL (ResFormer) — already running as exp202_vrl_v2
# Will check results separately

# T2. No-VRL control (identical to exp199 but with stride=76 eval)
echo "--- T2: no-VRL control (exp199 baseline + stride=76) ---"
RUN_ID=exp203_novrl_control \
eval $COMMON_TRAIN $COMMON_TTT \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T2"
echo ""

# T3. Early QAT threshold=0.15
echo "--- T3: Early QAT threshold=0.15 ---"
RUN_ID=exp204_earlyqat015 \
eval $COMMON_TRAIN $COMMON_TTT \
QAT_ENABLED=1 LATE_QAT=1 QAT_THRESHOLD=0.15 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T3"
echo ""

# T4. Early QAT threshold=0.5 (more aggressive, PR #545 style)
echo "--- T4: Early QAT threshold=0.5 ---"
RUN_ID=exp205_earlyqat050 \
eval $COMMON_TRAIN $COMMON_TTT \
QAT_ENABLED=1 LATE_QAT=1 QAT_THRESHOLD=0.5 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T4"
echo ""

# T5. Muon WD=0.07 (less aggressive, might help at 14L)
echo "--- T5: Muon WD=0.07 ---"
RUN_ID=exp206_wd07 \
eval $COMMON_TRAIN $COMMON_TTT \
MUON_WD=0.07 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T5"
echo ""

# T6. Muon WD=0.11 (more aggressive)
echo "--- T6: Muon WD=0.11 ---"
RUN_ID=exp207_wd11 \
eval $COMMON_TRAIN $COMMON_TTT \
MUON_WD=0.11 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T6"
echo ""

# T7. Asymmetric U-Net (9 encoder + 5 decoder instead of 7+7)
# This would need code changes — skip for now unless implemented

# T8. BigramHash dim=96 (up from 64)
echo "--- T8: BigramHash dim=96 ---"
RUN_ID=exp208_bigdim96 \
eval $COMMON_TRAIN $COMMON_TTT \
BIGRAM_DIM=96 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T8"
echo ""

# T9. EMA decay=0.995 (faster averaging for deeper model)
echo "--- T9: EMA decay=0.995 ---"
RUN_ID=exp209_ema995 \
eval $COMMON_TRAIN $COMMON_TTT \
EMA_DECAY=0.995 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T9"
echo ""

# T10. EMA decay=0.999 (slower averaging)
echo "--- T10: EMA decay=0.999 ---"
RUN_ID=exp210_ema999 \
eval $COMMON_TRAIN $COMMON_TTT \
EMA_DECAY=0.999 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T10"
echo ""

# T11. Warmdown=4000 (longer cooldown)
echo "--- T11: warmdown=4000 ---"
RUN_ID=exp211_wd4000 \
eval $COMMON_TRAIN $COMMON_TTT \
WARMDOWN_ITERS=4000 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T11"
echo ""

# T12. Warmdown=3000 (shorter)
echo "--- T12: warmdown=3000 ---"
RUN_ID=exp212_wd3000 \
eval $COMMON_TRAIN $COMMON_TTT \
WARMDOWN_ITERS=3000 \
torchrun --nproc_per_node=8 clean_train_202_novrl.py 2>&1 | tail -10
echo "DONE T12"
echo ""

echo "========================================="
echo "ALL A/B TESTS COMPLETE — $(date)"
echo "========================================="
