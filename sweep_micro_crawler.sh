#!/bin/bash
# Micro Crawler sweep — find the sweet spot
# Key insight: fewer stored blocks = wider dim = better per-step learning
# but also slower per step. Need to find the pareto frontier.
#
# Baseline to beat: fractal 5x2 cad=4 MLP4x = 2.1849 BPB

set -e
source .venv/bin/activate

COMMON="--lr 0.002 --grad-clip 5.0 --iterations 500 --eval-tokens 100000 --max-seconds 600 --batch-tokens 32768 --seq-len 1024 --seed 1337"

echo "============================================================"
echo "MICRO CRAWLER SWEEP — $(date)"
echo "============================================================"

# Config 1: 4flat + 1crawl x2 = 6 effective, 5 stored blocks (should auto-size wider)
echo -e "\n>>> CONFIG 1: 4f+1cx2 MLP4x trigram"
python train_micro_crawler.py $COMMON --num-flat-layers 4 --num-crawler-layers 1 --crawler-loops 2 \
  --flat-mlp-mult 4 --crawler-mlp-mult 4 --trigram-vocab 8192 --cadence 2 --run-id mc_4f1cx2_tri 2>&1 | tail -20

# Config 2: 5flat + 1crawl x2 = 7 effective, 6 stored (matches fractal 6x2 stored count)
echo -e "\n>>> CONFIG 2: 5f+1cx2 MLP4x trigram"
python train_micro_crawler.py $COMMON --num-flat-layers 5 --num-crawler-layers 1 --crawler-loops 2 \
  --flat-mlp-mult 4 --crawler-mlp-mult 4 --trigram-vocab 8192 --cadence 2 --run-id mc_5f1cx2_tri 2>&1 | tail -20

# Config 3: 3flat + 1crawl x3 = 6 effective, 4 stored (widest, 3 crawler firings)
echo -e "\n>>> CONFIG 3: 3f+1cx3 MLP4x trigram"
python train_micro_crawler.py $COMMON --num-flat-layers 3 --num-crawler-layers 1 --crawler-loops 3 \
  --flat-mlp-mult 4 --crawler-mlp-mult 4 --trigram-vocab 8192 --cadence 3 --run-id mc_3f1cx3_tri 2>&1 | tail -20

# Config 4: 4flat + 2crawl x2 = 8 effective, 6 stored (same stored as fractal, split architecture)
echo -e "\n>>> CONFIG 4: 4f+2cx2 MLP4x trigram"
python train_micro_crawler.py $COMMON --num-flat-layers 4 --num-crawler-layers 2 --crawler-loops 2 \
  --flat-mlp-mult 4 --crawler-mlp-mult 4 --trigram-vocab 8192 --cadence 3 --run-id mc_4f2cx2_tri 2>&1 | tail -20

# Config 5: 5flat + 1crawl x2, flat MLP3x / crawler MLP5x (invest in the crawler)
echo -e "\n>>> CONFIG 5: 5f+1cx2 flat3x/crawl5x trigram"
python train_micro_crawler.py $COMMON --num-flat-layers 5 --num-crawler-layers 1 --crawler-loops 2 \
  --flat-mlp-mult 3 --crawler-mlp-mult 5 --trigram-vocab 8192 --cadence 2 --run-id mc_5f1cx2_35_tri 2>&1 | tail -20

# Config 6: 3flat + 1crawl x2 = 5 effective, 4 stored (widest possible with crawl)
echo -e "\n>>> CONFIG 6: 3f+1cx2 MLP4x trigram"
python train_micro_crawler.py $COMMON --num-flat-layers 3 --num-crawler-layers 1 --crawler-loops 2 \
  --flat-mlp-mult 4 --crawler-mlp-mult 4 --trigram-vocab 8192 --cadence 2 --run-id mc_3f1cx2_tri 2>&1 | tail -20

# Config 7: flat-only control, 6 blocks no crawler (is the crawler even helping?)
echo -e "\n>>> CONFIG 7: 6flat control (no crawler)"
python train_micro_crawler.py $COMMON --num-flat-layers 6 --num-crawler-layers 0 --crawler-loops 0 \
  --flat-mlp-mult 4 --trigram-vocab 8192 --cadence 0 --run-id mc_6flat_ctrl 2>&1 | tail -20

# Config 8: 4flat + 1crawl x2 NO trigram (trigram ablation)
echo -e "\n>>> CONFIG 8: 4f+1cx2 MLP4x NO trigram"
python train_micro_crawler.py $COMMON --num-flat-layers 4 --num-crawler-layers 1 --crawler-loops 2 \
  --flat-mlp-mult 4 --crawler-mlp-mult 4 --trigram-vocab 0 --cadence 2 --run-id mc_4f1cx2_notri 2>&1 | tail -20

echo -e "\n============================================================"
echo "SWEEP COMPLETE — $(date)"
echo "============================================================"
echo "Logs in logs/mc_*.tsv"
