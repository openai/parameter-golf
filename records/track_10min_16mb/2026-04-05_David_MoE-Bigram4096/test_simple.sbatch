#!/bin/bash
#SBATCH --job-name=simple_test
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --output=/mnt/weka/dghazaryan/simple_test_%j.out

echo "Job started"
nvidia-smi
echo "Job done"
