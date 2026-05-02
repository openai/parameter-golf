#!/bin/bash
# Round 1 eval-only: 8 parallel eval method A/B tests
# Prerequisites: GPU 0 baseline must have finished and saved final_model.pt + final_model.int6.ptz

export PATH=/data/backups/rganapa/pylibs/bin:$PATH
export PYTHONPATH=/data/backups/rganapa/pylibs
export TMPDIR=/data/backups/rganapa/tmp
export TRITON_CACHE_DIR=/data/backups/rganapa/triton_cache
export TORCH_HOME=/data/backups/rganapa/torch_home
export DATA_PATH=data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export TTT_ONLY=1
export SEED=1337

cd /data/backups/rganapa/parameter-golf
mkdir -p eight_parallel_logs

# GPU 0: Baseline (PR #834 exact, for comparison)
CUDA_VISIBLE_DEVICES=0 TTT_CHUNK_TOKENS=1048576 TTT_EPOCHS=4 \
  WANDB_RUN_NAME=r1_gpu0_baseline \
  nohup python3 pr834_train_gpt.py > eight_parallel_logs/r1_gpu0_baseline.log 2>&1 &
echo "GPU0: baseline"

# GPU 1: Higher order (order=10 instead of 7)
CUDA_VISIBLE_DEVICES=1 TTT_CHUNK_TOKENS=1048576 TTT_EPOCHS=4 \
  NGRAM_ORDER=10 \
  WANDB_RUN_NAME=r1_gpu1_order10 \
  nohup python3 pr834_train_gpt.py > eight_parallel_logs/r1_gpu1_order10.log 2>&1 &
echo "GPU1: order=10"

# GPU 2: Finer chunks (256K instead of 1M)
CUDA_VISIBLE_DEVICES=2 TTT_CHUNK_TOKENS=262144 TTT_EPOCHS=4 \
  WANDB_RUN_NAME=r1_gpu2_chunk256k \
  nohup python3 pr834_train_gpt.py > eight_parallel_logs/r1_gpu2_chunk256k.log 2>&1 &
echo "GPU2: chunk=256K"

# GPU 3: More TTT epochs (8 instead of 4)
CUDA_VISIBLE_DEVICES=3 TTT_CHUNK_TOKENS=1048576 TTT_EPOCHS=8 \
  WANDB_RUN_NAME=r1_gpu3_ttt8ep \
  nohup python3 pr834_train_gpt.py > eight_parallel_logs/r1_gpu3_ttt8ep.log 2>&1 &
echo "GPU3: ttt_epochs=8"

# GPU 4: Higher TTT LR (0.001 instead of 0.0005)
CUDA_VISIBLE_DEVICES=4 TTT_CHUNK_TOKENS=1048576 TTT_EPOCHS=4 TTT_LR=0.001 \
  WANDB_RUN_NAME=r1_gpu4_tttlr001 \
  nohup python3 pr834_train_gpt.py > eight_parallel_logs/r1_gpu4_tttlr001.log 2>&1 &
echo "GPU4: ttt_lr=0.001"

# GPU 5: No TTT (n-gram only, saves eval time)
CUDA_VISIBLE_DEVICES=5 TTT_CHUNK_TOKENS=1048576 TTT_EPOCHS=0 \
  WANDB_RUN_NAME=r1_gpu5_no_ttt \
  nohup python3 pr834_train_gpt.py > eight_parallel_logs/r1_gpu5_no_ttt.log 2>&1 &
echo "GPU5: no TTT"

# GPU 6: Larger hash buckets (4M instead of 1M)
CUDA_VISIBLE_DEVICES=6 TTT_CHUNK_TOKENS=1048576 TTT_EPOCHS=4 \
  NGRAM_BUCKETS=4194304 \
  WANDB_RUN_NAME=r1_gpu6_4Mbuckets \
  nohup python3 pr834_train_gpt.py > eight_parallel_logs/r1_gpu6_4Mbuckets.log 2>&1 &
echo "GPU6: 4M buckets"

# GPU 7: Lower alpha center (2.5 instead of default)
CUDA_VISIBLE_DEVICES=7 TTT_CHUNK_TOKENS=1048576 TTT_EPOCHS=4 \
  ALPHA_CENTER=2.5 \
  WANDB_RUN_NAME=r1_gpu7_alpha_center25 \
  nohup python3 pr834_train_gpt.py > eight_parallel_logs/r1_gpu7_alpha_center25.log 2>&1 &
echo "GPU7: alpha_center=2.5"

echo ""
echo "All 8 eval-only A/B tests launched."
echo "Monitor: for g in 0 1 2 3 4 5 6 7; do echo GPU\$g:; tail -1 eight_parallel_logs/r1_gpu\$g_*.log; done"
