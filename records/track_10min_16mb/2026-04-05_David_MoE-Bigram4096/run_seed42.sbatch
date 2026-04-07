#!/bin/bash
#SBATCH --job-name=golf_42
#SBATCH --partition=research
#SBATCH --gres=gpu:8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/weka/dghazaryan/parameter-golf/records/track_10min_16mb/2026-04-05_David_MoE-Bigram4096/slurm_seed42_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/dghazaryan/conda-pkgs
source /mnt/weka/dghazaryan/miniconda3/etc/profile.d/conda.sh
conda activate golf

cd /mnt/weka/dghazaryan/parameter-golf/records/track_10min_16mb/2026-04-05_David_MoE-Bigram4096

BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=96 \
WARMDOWN_ITERS=4000 TARGET_MB=15.9 SEED=42 \
DATA_PATH=/mnt/weka/dghazaryan/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/mnt/weka/dghazaryan/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
2>&1 | tee train_seed42.log
