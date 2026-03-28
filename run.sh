#!/bin/bash
# Usage: bash run.sh <config>
# Configs:
#   baseline       - original train_gpt.py, 10min wallclock cap
#   baseline_full  - original train_gpt.py, 3000 steps no cap
#   v2_quick       - train_gpt_v2.py, 3000 steps (ablation signal)
#   v2_full        - train_gpt_v2.py, 20000 steps (full A100 run)
#   v2_h100        - train_gpt_v2.py, 8xH100 submission (10min cap)
#   11l            - v2 with 11 layers (Phase 2 bet A)
#   12l            - v2 with 12 layers (Phase 2 bet A)
#   trigram        - v2 + trigram hash on top of bigram (Phase 2 bet B)
#   seq4096        - v2 with seq_len=4096 (Phase 2 bet C)
#   kv2            - v2 with 2 KV heads instead of 4 (Phase 2 bet D)
#   bigram20k      - v2 with 20480 bigram buckets (Phase 2 bet E)
# Phase 3 (new SOTA stack):
#   v3_abl         - 12L INT4 bQAT + XSA4 + EMA + LeakyReLU² (1500 steps, ablation)
#   proxy_v3       - proxy run (10500 steps) for the v3 stack
#   v3_h100        - 8xH100 submission with full v3 stack

set -e
cd /vol/paraG/parameter-golf
source .venv/bin/activate

CONFIG=${1:-v2_quick}

export DATA_PATH=./data/datasets/fineweb10B_sp1024/
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024

case "$CONFIG" in

  baseline)
    echo "=== Running: BASELINE (10min wallclock cap) ==="
    export RUN_ID=baseline_ref
    torchrun --standalone --nproc_per_node=1 train_gpt.py
    ;;

  baseline_full)
    echo "=== Running: BASELINE full (3000 steps, no cap) ==="
    export RUN_ID=baseline_full ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0
    torchrun --standalone --nproc_per_node=1 train_gpt.py
    ;;

  v2_quick)
    echo "=== Running: V2 quick (3000 steps, no cap) ==="
    export RUN_ID=v2_quick ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  v2_full)
    echo "=== Running: V2 full A100 (20000 steps, no cap) ==="
    export RUN_ID=v2_full ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=0
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  v2_h100)
    echo "=== Running: V2 on 8xH100 (submission, 10min cap) ==="
    # WINNER CONFIG: 12L + INT4 MLP/Bigram QAT (trigram ablated out — hurt by 0.002 BPB)
    export RUN_ID=v2_h100_seed${SEED:-1} SEED=${SEED:-1} \
           NUM_LAYERS=12 MLP_QUANT_BITS=4
    torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
    ;;

  # ---- Phase 2 ablations (3000 steps each, ~35-60 min on A100) ----

  11l)
    echo "=== Phase 2 Bet A: 11 layers ==="
    export RUN_ID=abl_11l ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0 NUM_LAYERS=11
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  12l)
    echo "=== Phase 2 Bet A: 12 layers ==="
    export RUN_ID=abl_12l ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0 NUM_LAYERS=12
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  seq4096)
    echo "=== Phase 2 Bet C: seq_len=4096 ==="
    export RUN_ID=abl_seq4096 ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0 TRAIN_SEQ_LEN=4096
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  kv2)
    echo "=== Phase 2 Bet D: 2 KV heads ==="
    export RUN_ID=abl_kv2 ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0 NUM_KV_HEADS=2
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  bigram20k)
    echo "=== Phase 2 Bet E: 20480 bigram buckets ==="
    export RUN_ID=abl_bigram20k ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0 BIGRAM_VOCAB_SIZE=20480
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  noqat)
    echo "=== Ablation: v2 without QAT (isolate QAT impact) ==="
    export RUN_ID=abl_noqat ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=0 QAT_ENABLED=0
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  11l_kv2_b6k)
    echo "=== Phase 2 Combo: 11L + KV2 + bigram6144 ==="
    export RUN_ID=abl_11l_kv2_b6k ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=11 NUM_KV_HEADS=2 BIGRAM_VOCAB_SIZE=6144
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  12l_int4)
    echo "=== NOVEL BET: 12L + INT4 MLP QAT (est. ~14.8MB) ==="
    export RUN_ID=abl_12l_int4 ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  12l_int4_b9k)
    echo "=== WINNER CONFIG: 12L + INT4 + bigram9216 (size fix, target ~15.95MB) ==="
    export RUN_ID=abl_12l_int4_b9k ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 BIGRAM_VOCAB_SIZE=9216
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  12l_int4_bqat)
    echo "=== NOVEL: 12L + INT4 MLP + INT4 Bigram QAT (target ~15.75MB) ==="
    export RUN_ID=abl_12l_int4_bqat ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  proxy_12l_bqat)
    echo "=== PROXY: 12L + INT4 MLP + INT4 Bigram QAT, 10500 steps ==="
    export RUN_ID=proxy_12l_bqat ITERATIONS=10500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  proxy_10l)
    echo "=== PROXY: 10L INT5 MLP (baseline arch), 10500 steps ==="
    export RUN_ID=proxy_10l ITERATIONS=10500 MAX_WALLCLOCK_SECONDS=0
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  12l_bqat_trig)
    echo "=== ABLATION: 12L + INT4 bQAT + Trigram(512), 1500 steps ==="
    export RUN_ID=abl_12l_bqat_trig ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 TRIGRAM_VOCAB_SIZE=512
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  proxy_12l_bqat_trig)
    echo "=== PROXY: 12L + INT4 bQAT + Trigram(512), 10500 steps ==="
    export RUN_ID=proxy_12l_bqat_trig ITERATIONS=10500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 TRIGRAM_VOCAB_SIZE=512
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  13l_int4)
    echo "=== NOVEL BET: 13L + INT4 MLP QAT (est. ~16.0MB, risky) ==="
    export RUN_ID=abl_13l_int4 ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=13 MLP_QUANT_BITS=4
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  # ---- Phase 3: New SOTA stack (LeakyReLU² + XSA4 + EMA + GPTQ-lite) ----

  v3_abl)
    echo "=== Phase 3 ABLATION: 12L INT4 bQAT + XSA4 + EMA + LeakyReLU² + RoPE16 + LNScale (1500 steps) ==="
    # Full v4 stack: XSA + EMA + GPTQ-lite + LeakyReLU² + Partial RoPE(16) + LN Scale
    export RUN_ID=abl_v4_full ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  abl_emfix)
    echo "=== EMA-fix verification: 1500 steps, threshold=0.4 → QAT at step~300, 1200 QAT steps ==="
    # threshold=0.4 is proportional for 1500-step run (scale goes 0.5→0, so 0.4 fires at step~300)
    # EMA reset at QAT activation → 97.5% QAT-adapted by end
    # Verify: post-quant should be ~1.27-1.30, NOT 1.47
    export RUN_ID=abl_emfix ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.4 TTT_ENABLED=0
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  abl_ve)
    echo "=== ValueEmbedding ablation: 1500 steps, VE at last 3 layers ==="
    # ValueEmbedding reinjects token identity into V at last N layers (from SOTA #549)
    export RUN_ID=abl_ve ITERATIONS=1500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.4 VALUE_EMBED_LAYERS=3
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  proxy_v3)
    echo "=== Phase 3 PROXY: Full v4 stack + Late QAT + TTT, 10500 steps ==="
    export RUN_ID=proxy_v4_full ITERATIONS=10500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 TTT_ENABLED=1
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  proxy_v4)
    echo "=== Phase 4 PROXY: EMA fix (threshold=0.9 + EMA reset at QAT), 10500 steps ==="
    # Fix for EMA+LateQAT mismatch: threshold=0.9 → QAT starts at step ~7800 (2700 steps)
    # EMA also resets at QAT activation (code fix in train_gpt_v2.py)
    # Expected: post-quant close to fake-quant val (~1.163 instead of 1.356)
    export RUN_ID=proxy_v4_emfix ITERATIONS=10500 MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.9 TTT_ENABLED=1
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  v4_h100)
    echo "=== Phase 4 H100: EMA-fixed stack (8xH100 submission) ==="
    export RUN_ID=v4_h100_seed${SEED:-1} SEED=${SEED:-1} \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.9 TTT_ENABLED=1 \
           LATE_QAT_FRAC=0.65 VAL_LOSS_EVERY=1000
    torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
    ;;

  v6_parallel)
    echo "=== Phase 6 H100: Parallel Muon — DEPRECATED, reverted to DDP (virtual banking gave no speedup) ==="
    echo "Use v7_ve instead."
    exit 1
    ;;

  v7_ve)
    echo "=== Phase 7 H100: 12L + Value Embeddings (VE_DIM=128, last 2 layers) ==="
    # Value embeddings reinject token identity into V at layers 10,11 (SOTA uses this)
    # VE adds ~147KB compressed, fits in 16MB (790KB margin). Expect +0.005-0.015 BPB quality gain.
    # Fallback: SEED=X bash run.sh v4_h100 for vanilla 12L without VE
    export RUN_ID=v7_ve_seed${SEED:-1} SEED=${SEED:-1} \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.9 TTT_ENABLED=1 \
           LATE_QAT_FRAC=0.65 VAL_LOSS_EVERY=1000 \
           VALUE_EMBED_LAYERS=2 VALUE_EMBED_DIM=128
    torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
    ;;

  v8_static)
    echo "=== Phase 8 H100: VE + static_graph DDP (test VE overhead fix) ==="
    # Same as v7_ve but DDP(static_graph=True) — tests if overhead drops from 137ms to ~110ms
    # If step_avg drops: ~5200 steps vs 4380, potential +0.006 BPB
    export RUN_ID=v8_static_seed${SEED:-1} SEED=${SEED:-1} \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.9 TTT_ENABLED=1 \
           LATE_QAT_FRAC=0.65 VAL_LOSS_EVERY=1000 \
           VALUE_EMBED_LAYERS=2 VALUE_EMBED_DIM=128
    torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
    ;;

  v7_ve_small)
    echo "=== Phase 7 H100: 12L + Value Embeddings small (VE_DIM=64, last 1 layer) ==="
    # Half VE overhead, ~50-75% of quality gain. If 27ms overhead is embed-size-driven, this helps.
    export RUN_ID=v7_ve_small_seed${SEED:-1} SEED=${SEED:-1} \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.9 TTT_ENABLED=1 \
           LATE_QAT_FRAC=0.65 VAL_LOSS_EVERY=1000 \
           VALUE_EMBED_LAYERS=1 VALUE_EMBED_DIM=64
    torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
    ;;

  v5_rownorm)
    echo "=== Phase 5 H100: Rownorm backend — DEPRECATED, rownorm hurt quality ==="
    echo "Use v6_parallel instead."
    exit 1
    ;;

  proxy_v5_rownorm)
    echo "=== Proxy v5: Rownorm backend (1xH100, no wallclock cap) ==="
    export RUN_ID=proxy_v5_rownorm SEED=${SEED:-1} MAX_WALLCLOCK_SECONDS=0 \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.9 TTT_ENABLED=1 \
           VAL_LOSS_EVERY=500 MUON_BACKEND=rownorm
    torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
    ;;

  v3_h100)
    echo "=== Phase 3 H100: Full v4 stack + Late QAT + TTT (8xH100 submission) ==="
    export RUN_ID=v3_h100_seed${SEED:-1} SEED=${SEED:-1} \
           NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
           ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 TTT_ENABLED=1
    torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
    ;;

  *)
    echo "Unknown config: $CONFIG"
    echo "Available: baseline | baseline_full | v2_quick | v2_full | v2_h100"
    echo "Phase 2:   11l | 12l | seq4096 | kv2 | bigram20k | noqat | 11l_kv2_b6k"
    echo "Phase 3:   v3_abl | proxy_v3 | v3_h100"
    echo "Phase 4:   proxy_v4 | v4_h100  (EMA+LateQAT fix, threshold=0.9)"
    echo "Phase 6:   v6_parallel  (Parallel Muon: DEPRECATED)"
    echo "Phase 7:   v7_ve | v7_ve_small  (Value Embeddings)"
    echo "Phase 8:   v8_static  (VE + DDP static_graph, overhead fix test)"
    exit 1
    ;;
esac
