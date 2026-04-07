#!/bin/bash
#SBATCH --job-name=golf_test
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/weka/dghazaryan/parameter-golf/records/track_10min_16mb/2026-04-05_David_MoE-Bigram4096/slurm_quicktest_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/dghazaryan/conda-pkgs
source /mnt/weka/dghazaryan/miniconda3/etc/profile.d/conda.sh
conda activate golf

cd /mnt/weka/dghazaryan/parameter-golf/records/track_10min_16mb/2026-04-05_David_MoE-Bigram4096

SEED=1337 MAX_WALLCLOCK_SECONDS=120 \
DATA_PATH=/mnt/weka/dghazaryan/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/mnt/weka/dghazaryan/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=1 train_gpt.py \
2>&1 | tee quick_test.log
