#!/bin/bash
# H100 Runs — Based on 32 local experiments (2026-03-27)
#
# Key findings from local queue:
#   1. Seq_len 2048 is a MASSIVE win (-0.4 BPB vs 1024)
#   2. Muon LR 0.025 gives slight edge over default 0.04
#   3. TTT gives -0.04 to -0.11 BPB on H100 (eval is free/unlimited time)
#   4. GQA (4KV) keeps artifact under 16MB with ~0.01 BPB cost
#   5. No RoPE slightly better than partial RoPE
#   6. 13L/576d >> 11L/512d but artifact borderline
#
# Previous best: 1.2642 BPB (13L/576d, seq1024, TTT=7, 15.7MB artifact)
#
# Budget: ~$36 for 3 runs (~$12 each)

set -e
cd /workspace/parameter-golf

export HF_HOME=/workspace/.hf_cache
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model

# ============================================================
# RUN A: New Best Candidate — 13L/576d + Seq2048 + TTT=7
# ============================================================
# Combines our best architecture (13L/576d) with the biggest
# local discovery (seq_len 2048). Previous 13L/576d got 1.264
# with seq1024 — seq2048 should push significantly lower.
# Risk: artifact might exceed 16MB (previous was 15.7MB)
echo "=== RUN A: 13L/576d + Seq2048 + TTT=7 ==="
RUN_ID="runa_13L576d_seq2048" \
NUM_LAYERS=13 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=12000 WARMDOWN_ITERS=3500 WARMUP_STEPS=20 \
MATRIX_LR=0.025 \
BIGRAMHASH_BUCKETS=4096 \
SMEARGATE=1 UNET_SKIPS=1 INT6_QAT=1 TIE_EMBEDDINGS=1 \
ROPE_PARTIAL_DIMS=0 LN_SCALE=1 XSA_LAYERS=4 \
EMA_DECAY=0.997 \
TTT_ENABLED=1 TTT_EPOCHS=7 TTT_LR=0.002 TTT_CHUNK_TOKENS=32768 TTT_BATCH_SEQS=32 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
python train_gpt.py 2>&1 | tee /workspace/runa_13L576d_seq2048.log

# Save artifact with descriptive name
cp final_model.int6.ptz /workspace/artifact_runA_13L576d_seq2048.ptz 2>/dev/null || true
echo "Run A complete. Check log for BPB."
echo ""

# ============================================================
# RUN B: Safe Candidate — 11L/512d + Seq2048 + TTT=7
# ============================================================
# Same seq2048 win but with guaranteed-legal 11L/512d config.
# Previous 11L/512d got 1.352 (seq1024) — seq2048 should be ~1.25-1.30?
# Artifact: ~14-15MB (safely under 16MB limit)
echo "=== RUN B: 11L/512d + Seq2048 + TTT=7 ==="
RUN_ID="runb_11L512d_seq2048" \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=12000 WARMDOWN_ITERS=3500 WARMUP_STEPS=20 \
MATRIX_LR=0.025 \
BIGRAMHASH_BUCKETS=4096 \
SMEARGATE=1 UNET_SKIPS=1 INT6_QAT=1 TIE_EMBEDDINGS=1 \
ROPE_PARTIAL_DIMS=0 LN_SCALE=1 XSA_LAYERS=4 \
EMA_DECAY=0.997 \
TTT_ENABLED=1 TTT_EPOCHS=7 TTT_LR=0.002 TTT_CHUNK_TOKENS=32768 TTT_BATCH_SEQS=32 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
python train_gpt.py 2>&1 | tee /workspace/runb_11L512d_seq2048.log

cp final_model.int6.ptz /workspace/artifact_runB_11L512d_seq2048.ptz 2>/dev/null || true
echo "Run B complete. Check log for BPB."
echo ""

# ============================================================
# RUN C: Previous Best Rerun (baseline comparison)
# ============================================================
# Exact config from Run 4 that got 1.2642 BPB (seq1024).
# Use as baseline to measure how much seq2048 actually helps on H100.
echo "=== RUN C: 13L/576d + Seq1024 + TTT=7 (baseline) ==="
RUN_ID="runc_13L576d_seq1024_baseline" \
NUM_LAYERS=13 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 \
TRAIN_SEQ_LEN=1024 \
ITERATIONS=12000 WARMDOWN_ITERS=3500 WARMUP_STEPS=20 \
MATRIX_LR=0.025 \
BIGRAMHASH_BUCKETS=4096 \
SMEARGATE=1 UNET_SKIPS=1 INT6_QAT=1 TIE_EMBEDDINGS=1 \
ROPE_PARTIAL_DIMS=0 LN_SCALE=1 XSA_LAYERS=4 \
EMA_DECAY=0.997 \
TTT_ENABLED=1 TTT_EPOCHS=7 TTT_LR=0.002 TTT_CHUNK_TOKENS=32768 TTT_BATCH_SEQS=32 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
python train_gpt.py 2>&1 | tee /workspace/runc_13L576d_seq1024.log

cp final_model.int6.ptz /workspace/artifact_runC_13L576d_seq1024.ptz 2>/dev/null || true
echo "Run C complete. Check log for BPB."
echo ""

# ============================================================
# RUN D: Best Local Config (MHA) + Seq2048 + TTT=7
# ============================================================
# The best local config (BPB 1.628) used full MHA (8 KV heads)
# but was NEVER tested on H100. All H100 runs used GQA.
# Combining MHA with seq2048 could be the best overall.
# Risk: artifact may exceed 16MB (was 19.4MB locally at 5000 steps)
# but H100 trains longer + we can try compression locally.
echo "=== RUN D: 11L/512d MHA + Seq2048 + TTT=7 (best local config) ==="
RUN_ID="rund_11L512d_mha_seq2048" \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=8 \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=12000 WARMDOWN_ITERS=3500 WARMUP_STEPS=20 \
MATRIX_LR=0.025 \
BIGRAMHASH_BUCKETS=4096 \
SMEARGATE=1 UNET_SKIPS=1 INT6_QAT=1 TIE_EMBEDDINGS=1 \
ROPE_PARTIAL_DIMS=0 LN_SCALE=1 XSA_LAYERS=4 \
EMA_DECAY=0.997 \
TTT_ENABLED=1 TTT_EPOCHS=7 TTT_LR=0.002 TTT_CHUNK_TOKENS=32768 TTT_BATCH_SEQS=32 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
python train_gpt.py 2>&1 | tee /workspace/rund_11L512d_mha_seq2048.log

cp final_model.int6.ptz /workspace/artifact_runD_11L512d_mha_seq2048.ptz 2>/dev/null || true
echo "Run D complete. Check log for BPB."
echo ""

echo "=== ALL RUNS COMPLETE ==="
echo "Check artifacts in /workspace/artifact_run*.ptz"
echo "Logs in /workspace/run*.log"
echo ""
echo "Expected results:"
echo "  Run A (13L+seq2048): Target <1.22 BPB, artifact ~15-17MB"
echo "  Run B (11L+seq2048): Target <1.30 BPB, artifact ~14-15MB (safe)"
echo "  Run C (13L+seq1024): Expect ~1.264 BPB (baseline for comparison)"
